from __future__ import annotations
from logger import logger

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
import os
import traceback
import re
from typing import Callable, List, Optional, Dict
from cache_manager import cache

import pandas as pd

from indicators import add_indicators
from persistence import safe_load_json, safe_save_json
from market_safety import run_market_safety, check_high_impact_news, check_dangerous_volatility, check_atr_floor
from confirmation_engine import validate_live_signal
from signal_generator import decide_direction_live
from learning_engine import learning_engine, record_voter_outcome, record_session_outcome
from market_regime import detect_market_regime, MartingaleAdaptiveController

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


TIMEZONE_NAME = "Asia/Kolkata"
MARKET_OPEN = time(13, 0)
MARKET_CLOSE = time(22, 0)
SIGNAL_WINDOW_SECONDS = 75
PRE_SIGNAL_MIN_SECONDS = 60
PRE_SIGNAL_MAX_SECONDS = 120
MARTINGALE_PREALERT_MIN_SECONDS = 45
MARTINGALE_PREALERT_MAX_SECONDS = 150
MARTINGALE_ENTRY_DELAY = timedelta(minutes=2)
GENERATED_SIGNALS_FILE = "generated_signals.json"
SIGNAL_PATTERN = re.compile(r"^(?P<hour>\d{2}):(?P<minute>\d{2})\s+EURUSD\s+(?P<signal>CALL|PUT)$")

# Forced daily signal times (IST) – ALWAYS sent, never skipped
FORCED_DIRECT_TIME = "15:05"
FORCED_MARTINGALE_TIME = "15:10"
FORCED_SIGNAL_LOW_CONF_THRESHOLD = 55  # Below this → LOW CONFIDENCE warning
FORCED_MARTINGALE_MIN_CONFIDENCE = 75  # Below this → HIGH RISK MARTINGALE warning


@dataclass(frozen=True)
class SignalEntry:
    signal_time: datetime
    pair: str
    direction: str
    raw_line: str


# Global trade lock: minimum gap between any two confirmed trades (seconds)
TRADE_LOCK_SECONDS = 5 * 60  # 5 minutes


class SmartSignalManager:
    def __init__(self):
        self.active_signals = []
        self.processed_signals = set()
        self.last_update_id = None
        self.last_signal_update_time = None
        self.evaluated_signal_count = 0
        self.confirmed_signal_count = 0
        self.enable_martingale = True
        self.tracked_trades = []
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.last_daily_report_date = None
        self.storage_file = "smart_signals.json"
        self.last_signal_lines = []
        self._initialized = False
        self.skip_message_times = []
        # Global trade lock – updated whenever any signal type is confirmed
        self.last_confirmed_trade_time: Optional[datetime] = None
        # Overlap tracking: timestamps of every forced-signal trade-lock bypass
        self.forced_overlap_times: List[datetime] = []

    def load(self):
        try:
            data = safe_load_json(self.storage_file, default={})
            if not data:
                return
                
            self.tracked_trades = data.get("tracked_trades", [])
            for t in self.tracked_trades:
                if "signal_time" in t and isinstance(t["signal_time"], str):
                    t["signal_time"] = datetime.fromisoformat(t["signal_time"])
                if "expiry_time" in t and isinstance(t["expiry_time"], str):
                    t["expiry_time"] = datetime.fromisoformat(t["expiry_time"])
            
            self.last_daily_report_date = data.get("last_daily_report_date")
            if self.last_daily_report_date:
                self.last_daily_report_date = date.fromisoformat(self.last_daily_report_date)
            
            self.processed_signals = set(data.get("processed_signals", []))

            raw_lctt = data.get("last_confirmed_trade_time")
            if raw_lctt:
                try:
                    self.last_confirmed_trade_time = datetime.fromisoformat(raw_lctt)
                except Exception as e:
                    self.last_confirmed_trade_time = None
            
            self.recalculate_stats()
            logger.info(f"Loaded {len(self.tracked_trades)} trades from {self.storage_file}")
        except Exception as e:
            logger.exception(f"Error loading {self.storage_file}: {e}")

    def save(self):
        try:
            data = {
                "tracked_trades": [],
                "last_daily_report_date": self.last_daily_report_date.isoformat() if self.last_daily_report_date else None,
                "processed_signals": list(self.processed_signals),
                "last_confirmed_trade_time": self.last_confirmed_trade_time.isoformat() if self.last_confirmed_trade_time else None,
            }
            for t in self.tracked_trades:
                entry = dict(t)
                if "signal_time" in entry and isinstance(entry["signal_time"], datetime):
                    entry["signal_time"] = entry["signal_time"].isoformat()
                if "expiry_time" in entry and isinstance(entry["expiry_time"], datetime):
                    entry["expiry_time"] = entry["expiry_time"].isoformat()
                data["tracked_trades"].append(entry)
            
            safe_save_json(self.storage_file, data)
        except Exception as e:
            logger.exception(f"Error saving {self.storage_file}: {e}")

    def recalculate_stats(self):
        resolved = [t for t in self.tracked_trades if t.get("resolved")]
        self.total_trades = len(resolved)
        self.wins = sum(1 for t in resolved if t.get("result") == "WIN")
        self.losses = self.total_trades - self.wins

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    _PROCESSED_SIGNALS_MAX = 1_000

    def cleanup_processed_signals(self) -> None:
        """
        Prune stale entries from processed_signals to prevent unbounded
        RAM growth during long Railway uptime.

        Strategy:
          1. Remove keys whose HH:MM_DIRECTION pattern includes a date
             prefix that is NOT today (e.g. "2025-01-01_15:05_CALL").
          2. Retain all today's keys unconditionally (they protect against
             duplicate execution on the current trading day).
          3. If the set still exceeds _PROCESSED_SIGNALS_MAX after pruning,
             discard the oldest alphabetical entries until within the cap.
        """
        today_str = date.today().isoformat()   # "YYYY-MM-DD"
        before = len(self.processed_signals)

        # Step 1 – drop dated entries that belong to a different day
        to_remove: set[str] = set()
        for key in self.processed_signals:
            # Keys that carry an explicit date prefix look like:
            # "YYYY-MM-DD_HH:MM_DIRECTION" or "YYYY-MM-DD_martingale_HH:MM_…"
            if len(key) >= 10 and key[4] == "-" and key[7] == "-":
                key_date = key[:10]
                if key_date != today_str:
                    to_remove.add(key)

        self.processed_signals -= to_remove
        after_date_prune = len(self.processed_signals)

        # Step 2 – hard cap: drop the alphabetically earliest entries
        excess = len(self.processed_signals) - self._PROCESSED_SIGNALS_MAX
        if excess > 0:
            sorted_keys = sorted(self.processed_signals)
            self.processed_signals -= set(sorted_keys[:excess])

        after = len(self.processed_signals)
        removed_stale = before - after_date_prune
        removed_cap   = after_date_prune - after

        if removed_stale or removed_cap:
            logger.info(
                f"processed_signals cleanup completed: "
                f"removed {removed_stale} stale-date entries, "
                f"{removed_cap} over-cap entries. "
                f"Remaining: {after}/{before}"
            )
        else:
            logger.debug(
                f"processed_signals cleanup: no action needed "
                f"({after} entries)."
            )

manager = SmartSignalManager()
manager.load()
manager.cleanup_processed_signals()   # prune stale keys from previous day(s) on startup


def _get_timezone():
    if ZoneInfo is None:
        return None
    return ZoneInfo(TIMEZONE_NAME)


def _now() -> datetime:
    tz = _get_timezone()
    if tz is None:
        return datetime.now()
    return datetime.now(tz)


def _signal_key(signal_time: datetime, direction: str) -> str:
    return f"{signal_time:%H:%M}_{direction}"


def _build_signal_state(entries: List[SignalEntry], source: str = "telegram") -> List[dict]:
    return [
        {
            "time": entry.signal_time,
            "pair": entry.pair,
            "direction": entry.direction,
            "source": source,
            "pre_sent": False,
            "confirmed_sent": False,
            "martingale_time": entry.signal_time + MARTINGALE_ENTRY_DELAY,
            "martingale_prealert_sent": False,
            "martingale_confirmed_sent": False,
            "martingale_confidence": 0,
            "raw_line": entry.raw_line,
            # Forced-signal extras (default to None/False for non-forced signals)
            "is_forced": False,
            "forced_type": None,
            "low_confidence": False,
            "stored_confidence": 0,
        }
        for entry in entries
    ]


def _parse_line(line: str, current_day: date) -> Optional[SignalEntry]:
    try:
        if not isinstance(line, str):
            return None

        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            return None

        match = SIGNAL_PATTERN.match(stripped.upper())
        if match is None:
            return None

        signal_hour = int(match.group("hour"))
        signal_minute = int(match.group("minute"))

        if signal_hour < 0 or signal_hour > 23:
            return None
        if signal_minute < 0 or signal_minute > 59:
            return None

        # Critical time conversion is guarded to keep the bot crash-free.
        signal_time = datetime.combine(current_day, time(signal_hour, signal_minute))

        tz = _get_timezone()
        if tz is not None:
            signal_time = signal_time.replace(tzinfo=tz)

        return SignalEntry(
            signal_time=signal_time,
            pair="EURUSD",
            direction=match.group("signal").upper(),
            raw_line=stripped,
        )
    except Exception as e:
        logger.exception(f"Parse error: {e}")
        return None


def load_signal_entries(signal_lines: List[str], current_day: Optional[date] = None) -> List[SignalEntry]:
    if current_day is None:
        current_day = _now().date()

    if not signal_lines:
        return []

    entries: List[SignalEntry] = []
    for line in signal_lines:
        entry = _parse_line(line, current_day)
        if entry is not None:
            entries.append(entry)

    return entries


def load_generated_signals() -> List[SignalEntry]:
    """Load generated_signals.json and convert to SignalEntry list."""
    try:
        generated = safe_load_json(GENERATED_SIGNALS_FILE, default=[])
        if not isinstance(generated, list):
            return []
    except Exception as e:
        logger.exception(f"Error loading generated signals: {e}")
        return []

    tz = _get_timezone()
    now = _now()
    today = now.date()
    file_date = None
    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(GENERATED_SIGNALS_FILE), tz=tz)
        file_date = mtime.date()
    except Exception as e:
        file_date = None

    entries = []
    for sig in generated:
        try:
            time_str = str(sig.get("time", ""))
            pair = str(sig.get("pair", "EURUSD")).upper()
            direction = str(sig.get("direction", "")).upper()
            generated_date = sig.get("generated_date")

            if not time_str or direction not in ("CALL", "PUT"):
                continue
            if generated_date and str(generated_date) != today.isoformat():
                continue
            if not generated_date and file_date is not None and file_date != today:
                continue

            parts = time_str.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            sig_time = datetime.combine(today, time(hour, minute))
            if tz is not None:
                sig_time = sig_time.replace(tzinfo=tz)

            if sig_time.time() < MARKET_OPEN:
                continue
            if now > sig_time + MARTINGALE_ENTRY_DELAY + timedelta(seconds=SIGNAL_WINDOW_SECONDS):
                continue

            raw_line = f"{time_str} {pair} {direction}"
            entries.append(SignalEntry(
                signal_time=sig_time,
                pair=pair,
                direction=direction,
                raw_line=raw_line,
            ))
        except (KeyError, ValueError, IndexError, TypeError) as e:
            print(f"Error parsing generated signal {sig}: {e}")
            continue

    return entries


def _merge_generated_into_manager() -> None:
    """Merge generated signal entries into manager.active_signals, avoiding duplicates."""
    generated_entries = load_generated_signals()
    if not generated_entries:
        return

    existing_keys = set()
    for sig in manager.active_signals:
        try:
            key = _signal_key(sig["time"], sig["direction"])
            existing_keys.add(key)
        except Exception as e:
            logger.exception("Unexpected error during loop")
            continue

    new_entries = []
    for entry in generated_entries:
        key = _signal_key(entry.signal_time, entry.direction)
        if key not in existing_keys:
            existing_keys.add(key)
            new_entries.append(entry)

    if new_entries:
        manager.active_signals.extend(_build_signal_state(new_entries, source="generated"))


def _inject_forced_signals_into_manager(now: Optional[datetime] = None) -> None:
    """
    Ensure the two compulsory daily signals (15:05 direct and 15:10 martingale)
    are always present in manager.active_signals with ``is_forced=True``.

    These are injected independently of the Telegram / generated signal files so
    they can never be accidentally omitted.  The normal processing loop skips
    them (via the ``is_forced`` guard), and ``_process_forced_signals()`` handles
    their PRE + CONFIRMATION messages without any confidence / RSI / momentum
    gating.
    """
    if now is None:
        now = _now()

    today = now.date()
    tz = _get_timezone()

    forced_specs = [
        (FORCED_DIRECT_TIME,     "direct"),
        (FORCED_MARTINGALE_TIME, "martingale"),
    ]

    for time_str, forced_type in forced_specs:
        try:
            h, m = map(int, time_str.split(":"))
            signal_time = datetime.combine(today, time(h, m))
            if tz is not None:
                signal_time = signal_time.replace(tzinfo=tz)

            # Dedup: skip if a forced entry for this slot already exists
            already_present = any(
                sig.get("is_forced")
                and sig.get("forced_type") == forced_type
                and sig["time"].date() == today
                for sig in manager.active_signals
            )
            if already_present:
                continue

            forced_state = {
                "time": signal_time,
                "pair": "EURUSD",
                # Direction will be decided live by _process_forced_signals;
                # use a neutral placeholder so the slot is populated.
                "direction": "CALL",
                "source": "forced",
                "pre_sent": False,
                "confirmed_sent": False,
                "martingale_time": signal_time + MARTINGALE_ENTRY_DELAY,
                "martingale_prealert_sent": False,
                "martingale_confirmed_sent": False,
                "martingale_confidence": 0,
                "raw_line": f"{time_str} EURUSD FORCED",
                # Forced-signal extras
                "is_forced": True,
                "forced_type": forced_type,
                "low_confidence": False,
                "stored_confidence": 0,
                "is_blocked": False,
            }
            manager.active_signals.append(forced_state)
            logger.debug(f"[FORCED] Injected {forced_type} signal slot at {time_str}")
        except Exception as e:
            logger.exception(f"[FORCED] Injection error for {time_str}: {e}")




def _merge_states(primary: List[dict], secondary: List[dict]) -> List[dict]:
    """Merge two lists of signal states, avoiding duplicates based on time and direction."""
    seen = set()
    merged = []
    
    # Primary (Telegram) takes precedence
    for s in primary:
        key = _signal_key(s["time"], s["direction"])
        if key not in seen:
            merged.append(s)
            seen.add(key)
            
    # Secondary (Generated) added if not already present
    for s in secondary:
        key = _signal_key(s["time"], s["direction"])
        if key not in seen:
            merged.append(s)
            seen.add(key)
            
    return merged


def update_signal_list(signal_lines: Optional[List[str]] = None, now: Optional[datetime] = None) -> List[dict]:
    if now is None:
        now = _now()

    current_day = now.date()
    previous_day = manager.last_signal_update_time.date() if manager.last_signal_update_time is not None else None

    if previous_day != current_day:
        manager.active_signals = []
        manager.processed_signals = set()
        manager.last_update_id = None
        manager.evaluated_signal_count = 0
        manager.confirmed_signal_count = 0

    if signal_lines is None:
        manager.last_signal_update_time = now
        _merge_generated_into_manager()
        _inject_forced_signals_into_manager(now)
        return manager.active_signals

    if isinstance(signal_lines, str):
        signal_lines = signal_lines.splitlines()

    if not isinstance(signal_lines, list):
        signal_lines = []

    if not signal_lines:
        manager.active_signals = []
        manager.processed_signals = set()
        manager.last_update_id = None
        manager.evaluated_signal_count = 0
        manager.confirmed_signal_count = 0
        manager.last_signal_update_time = now
        _merge_generated_into_manager()
        _inject_forced_signals_into_manager(now)
        return manager.active_signals

    update_id = f"mem-{current_day.isoformat()}-{len(signal_lines)}-{'|'.join(signal_lines)}"

    if update_id != manager.last_update_id:
        entries = load_signal_entries(signal_lines, current_day)
        manager.active_signals = _build_signal_state(entries)
        manager.processed_signals = set()
        manager.evaluated_signal_count = 0
        manager.confirmed_signal_count = 0
        manager.last_update_id = update_id
        print("Signal list updated")

    _merge_generated_into_manager()
    _inject_forced_signals_into_manager(now)
    manager.last_signal_update_time = now
    return manager.active_signals


def apply_signal_text(signal_text: str, now: Optional[datetime] = None) -> List[dict]:
    if not isinstance(signal_text, str):
        return update_signal_list([], now)

    lines = [line.strip() for line in signal_text.splitlines() if line.strip()]
    return update_signal_list(lines, now)


def should_force_fast_mode(now: Optional[datetime] = None, window_seconds: int = 180) -> bool:
    if now is None:
        now = _now()

    for signal in manager.active_signals:
        try:
            signal_time = signal.get("time")
            direction = signal.get("direction")
            if signal_time is None:
                continue
            if signal_time.date() != now.date():
                continue
            if signal_time.time() < MARKET_OPEN or signal_time.time() > MARKET_CLOSE:
                continue

            if direction is not None and _signal_key(signal_time, direction) in manager.processed_signals:
                continue

            seconds_to_signal = (signal_time - now).total_seconds()
            if 0 <= seconds_to_signal <= window_seconds:
                return True
        except Exception as e:
            logger.exception("Unexpected error during loop")
            continue

    return False


def validate_sniper_signal(df: pd.DataFrame, direction: str) -> bool:
    if df is None or len(df) < 200:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    required_values = [
        last["EMA50"],
        last["EMA200"],
        last["RSI"],
        prev["RSI"],
        last["ATR"],
        df["ATR"].mean(),
    ]
    if any(pd.isna(value) for value in required_values):
        return False

    atr = float(last["ATR"])
    atr_mean = float(df["ATR"].mean())
    candle_size = abs(float(last["Close"]) - float(last["Open"]))
    avg_candle = (df["Close"] - df["Open"]).abs().tail(10).mean()
    distance = abs(float(last["Close"]) - float(last["EMA50"]))

    trend_threshold = max(0.0002, atr * 0.15)
    ema_distance_threshold = max(0.0003, atr * 0.50)

    if direction == "CALL":
        trend_ok = last["EMA50"] > last["EMA200"]
        rsi_ok = last["RSI"] > prev["RSI"]
    else:
        trend_ok = last["EMA50"] < last["EMA200"]
        rsi_ok = last["RSI"] < prev["RSI"]

    if not trend_ok:
        return False
    if abs(float(last["EMA50"]) - float(last["EMA200"])) <= trend_threshold:
        return False
    if not rsi_ok:
        return False
    if atr <= atr_mean:
        return False
    if candle_size <= float(avg_candle):
        return False
    if distance > ema_distance_threshold:
        return False

    return True


def validate_martingale_signal(df: pd.DataFrame, direction: str) -> bool:
    if not validate_sniper_signal(df, direction):
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if direction == "CALL":
        if last["RSI"] <= 57:
            return False
        if last["RSI"] <= prev["RSI"]:
            return False
        if float(last["Close"]) <= float(last["Open"]):
            return False
    else:
        if last["RSI"] >= 43:
            return False
        if last["RSI"] >= prev["RSI"]:
            return False
        if float(last["Close"]) >= float(last["Open"]):
            return False

    return True


def calculate_confidence(df: pd.DataFrame, direction: str) -> int:
    if df is None or len(df) < 3:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    atr = float(last["ATR"]) if not pd.isna(last.get("ATR")) else 0.0
    atr_mean = float(df["ATR"].tail(50).mean()) if not pd.isna(df["ATR"].mean()) else 0.0

    candle_size = abs(float(last["Close"]) - float(last["Open"]))
    prev_candle_size = abs(float(prev["Close"]) - float(prev["Open"]))
    avg_candle = float((df["Close"] - df["Open"]).abs().tail(10).mean())

    distance = abs(float(last["Close"]) - float(last["EMA50"]))
    ema_separation = abs(float(last["EMA50"]) - float(last["EMA200"]))
    ema_separation_mean = abs(df["EMA50"] - df["EMA200"]).tail(20).mean()

    score = 0

    # 1. EMA Trend: 25
    if direction == "CALL":
        if last["EMA50"] > last["EMA200"]:
            score += 25
    else:
        if last["EMA50"] < last["EMA200"]:
            score += 25

    # 2. RSI Momentum: 25
    rsi_now = float(last["RSI"])
    rsi_prev = float(prev["RSI"])
    rsi_prev2 = float(prev2["RSI"])
    
    rsi_strong = False
    if direction == "CALL":
        if rsi_now > 52 and rsi_now > rsi_prev:
            score += 15
            # RSI acceleration
            if (rsi_now - rsi_prev) > (rsi_prev - rsi_prev2) and (rsi_now - rsi_prev) > 1.0:
                score += 10
                rsi_strong = True
    else:
        if rsi_now < 48 and rsi_now < rsi_prev:
            score += 15
            # RSI acceleration
            if (rsi_prev - rsi_now) > (rsi_prev2 - rsi_prev) and (rsi_prev - rsi_now) > 1.0:
                score += 10
                rsi_strong = True

    # 3. ATR Strength: 20
    if atr > atr_mean:
        score += 10
        if atr > (atr_mean * 1.2):
            score += 10

    # 4. Candle Momentum: 20
    candle_strong = False
    if candle_size > avg_candle:
        score += 10
        if direction == "CALL" and last["Close"] > last["Open"] and prev["Close"] > prev["Open"]:
            if candle_size >= prev_candle_size:
                score += 10
                candle_strong = True
        elif direction == "PUT" and last["Close"] < last["Open"] and prev["Close"] < prev["Open"]:
            if candle_size >= prev_candle_size:
                score += 10
                candle_strong = True

    # 5. EMA Position: 10
    if distance <= max(0.0003, atr * 0.50):
        score += 10

    # Confidence Penalties
    penalties = 0
    if candle_size < (avg_candle * 0.5):
        penalties += 10
    if atr < (atr_mean * 0.8):
        penalties += 10
    if abs(rsi_now - rsi_prev) < 0.5:
        penalties += 15
    if ema_separation < (ema_separation_mean * 0.8):
        penalties += 10
    if atr < (atr_mean * 0.6):
        penalties += 30  # Decay during sideways

    confidence = score - penalties
    
    # Cap at 80 unless all strong conditions are met
    ema_strong = ema_separation > ema_separation_mean
    if confidence > 80:
        if not (atr > atr_mean and rsi_strong and candle_strong and ema_strong):
            confidence = 80

    return max(0, min(confidence, 100))


def build_forex_targets(df: pd.DataFrame, direction: str, confidence: int) -> tuple[float, float, float]:
    last = df.iloc[-1]
    atr = float(last["ATR"])
    entry = float(last["Close"])

    if confidence >= 85:
        tp_factor = 2.0
        sl_factor = 0.8
    elif confidence >= 75:
        tp_factor = 1.5
        sl_factor = 1.0
    else:
        tp_factor = 1.0
        sl_factor = 1.2

    if direction == "CALL":
        tp = entry + (tp_factor * atr)
        sl = entry - (sl_factor * atr)
    else:
        tp = entry - (tp_factor * atr)
        sl = entry + (sl_factor * atr)

    return round(entry, 5), round(tp, 5), round(sl, 5)


def _get_ema_trend(df: pd.DataFrame, direction: str) -> str:
    """Determine if EMA trend is Bullish or Bearish."""
    last = df.iloc[-1]
    normalized_direction = str(direction).upper()

    if normalized_direction in {"CALL", "BUY"}:
        if last["EMA50"] > last["EMA200"]:
            return "Bullish 📈"
        else:
            return "Bearish 📉"
    else:
        if last["EMA50"] < last["EMA200"]:
            return "Bearish 📉"
        else:
            return "Bullish 📈"


def _get_rsi_interpretation(rsi: float) -> str:
    """Interpret RSI value."""
    if rsi >= 70:
        return "Overbought"
    elif rsi >= 60:
        return "Moderate Bullish"
    elif rsi > 50:
        return "Slightly Bullish"
    elif rsi == 50:
        return "Neutral"
    elif rsi > 40:
        return "Slightly Bearish"
    elif rsi >= 30:
        return "Moderate Bearish"
    else:
        return "Oversold"


def _get_atr_strength(df: pd.DataFrame) -> str:
    """Determine ATR strength as Low/Medium/High."""
    last = df.iloc[-1]
    atr = float(last["ATR"])
    atr_mean = float(df["ATR"].mean())
    
    if atr < atr_mean:
        return "Low Volatility"
    elif atr < atr_mean * 1.3:
        return "Medium Volatility"
    else:
        return "High Volatility 🔥"


def _get_candle_strength(df: pd.DataFrame) -> str:
    """Determine candle strength as Weak/Strong."""
    last = df.iloc[-1]
    candle_size = abs(float(last["Close"]) - float(last["Open"]))
    avg_candle = float((df["Close"] - df["Open"]).abs().tail(10).mean())
    
    if candle_size < avg_candle:
        return "Weak"
    else:
        return "Strong 💪"


def _build_pre_message(signal: dict, confidence: int, df: pd.DataFrame) -> str:
    """Build pre-signal message with full indicator analysis."""
    last = df.iloc[-1]
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 0
    ema_trend = _get_ema_trend(df, signal['direction'])
    rsi_interp = _get_rsi_interpretation(rsi)
    atr_strength = _get_atr_strength(df)
    candle_strength = _get_candle_strength(df)
    
    return (
        f"📊 *PRE-SIGNAL*\n\n"
        f"Pair: {signal['pair']}\n"
        f"Direction: {signal['direction']}\n"
        f"Time: {signal['time']:%H:%M}\n\n"
        f"Confidence: {confidence}%\n\n"
        f"*Technical Analysis*\n"
        f"EMA Trend: {ema_trend}\n"
        f"RSI: {rsi:.0f} ({rsi_interp})\n"
        f"ATR: {atr_strength}\n"
        f"Candle Strength: {candle_strength}\n\n"
        f"Status: Preparing for entry ⏳"
    )


def _build_confirm_message(signal: dict, confidence: int, expiry: datetime, tp: float, sl: float, df: pd.DataFrame) -> str:
    """Build confirmed signal message with full indicator analysis."""
    last = df.iloc[-1]
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 0
    ema_trend = _get_ema_trend(df, signal['direction'])
    rsi_interp = _get_rsi_interpretation(rsi)
    atr_strength = _get_atr_strength(df)
    candle_strength = _get_candle_strength(df)
    source = signal.get("source", "telegram")
    source_badges = {"telegram": "📨 Telegram", "generated": "🤖 Generated", "auto": "⚡ Auto"}
    source_label = source_badges.get(source, source.title())
    
    # Determine signal quality based on confidence
    if confidence >= 85:
        signal_quality = "HIGH PROBABILITY TRADE ✅"
    elif confidence >= 75:
        signal_quality = "STRONG SIGNAL ✅"
    else:
        signal_quality = "VALID SIGNAL ✓"
    
    return (
        f"📊 *SIGNAL CONFIRMED*\n\n"
        f"Pair: {signal['pair']}\n"
        f"Direction: {signal['direction']}\n"
        f"Source: {source_label}\n\n"
        f"Confidence: {confidence}%\n\n"
        f"*Technical Analysis*\n"
        f"EMA Trend: {ema_trend}\n"
        f"RSI: {rsi:.0f} ({rsi_interp})\n"
        f"ATR: {atr_strength}\n"
        f"Candle Strength: {candle_strength}\n\n"
        f"*Trade Details*\n"
        f"Entry Time: {signal['time']:%H:%M}\n"
        f"Expiry: {expiry:%H:%M}\n"
        f"TP: {tp}\n"
        f"SL: {sl}\n\n"
        f"Decision: {signal_quality}"
    )


def _check_safety_rules(df: pd.DataFrame, direction: str, confidence: int) -> tuple[bool, str, int]:
    """
    Strictly re-check technical indicators before confirmation.
    Re-checks: ATR, RSI, EMA Trend, and Momentum.
    """
    if df is None or len(df) < 200:
        return False, "insufficient data", confidence

    last = df.iloc[-1]
    prev = df.iloc[-2]

    required_values = [
        last["EMA50"],
        last["EMA200"],
        last["RSI"],
        prev["RSI"],
        last["ATR"],
        df["ATR"].mean(),
    ]
    if any(pd.isna(value) for value in required_values):
        return False, "indicator values unavailable", confidence

    atr = float(last["ATR"])
    atr_mean = float(df["ATR"].mean())
    rsi_value = float(last["RSI"])
    rsi_prev = float(prev["RSI"])
    
    # ==========================================
    # 1. ATR / VOLATILITY CHECK (Weak Market)
    # ==========================================
    # Absolute minimum volatility floor
    if atr < 0.00008:
        return False, "Market too dead (Flat ATR)", confidence
        
    # Relative weakness
    if atr < (atr_mean * 0.65):
        return False, "Volatility too weak for reliable signal", confidence

    # ==========================================
    # 2. RSI & MOMENTUM CHECK
    # ==========================================
    # Overbought/Oversold protection (Stricter)
    if direction == "CALL" and rsi_value >= 65:
        return False, "RSI overbought (>65)", confidence
    if direction == "PUT" and rsi_value <= 35:
        return False, "RSI oversold (<35)", confidence

    # Momentum Direction check (Must be moving WITH the trade)
    if direction == "CALL":
        if rsi_value < 52:
            return False, "Weak bullish momentum (RSI < 52)", confidence
        if rsi_value <= rsi_prev:
            return False, "Momentum stalled (RSI not increasing)", confidence
    else:
        if rsi_value > 48:
            return False, "Weak bearish momentum (RSI > 48)", confidence
        if rsi_value >= rsi_prev:
            return False, "Momentum stalled (RSI not decreasing)", confidence

    # ==========================================
    # 3. EMA TREND & ALIGNMENT
    # ==========================================
    # Directional Alignment
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    ema50_prev = float(df.iloc[-3]["EMA50"]) # Check 2 candles back for slope
    
    if direction == "CALL":
        trend_direction_ok = ema50 > ema200
        price_above_ema = float(last["Close"]) > ema50
        ema_slope_ok = ema50 > ema50_prev # Must be rising
    else:
        trend_direction_ok = ema50 < ema200
        price_above_ema = float(last["Close"]) < ema50
        ema_slope_ok = ema50 < ema50_prev # Must be falling

    if not trend_direction_ok:
        return False, f"Opposite EMA trend ({'Bullish' if ema50 > ema200 else 'Bearish'})", confidence
        
    if not price_above_ema:
        return False, "Price on wrong side of EMA50", confidence
        
    if not ema_slope_ok:
        return False, f"EMA50 slope not {direction.lower()}-aligned", confidence

    # ==========================================
    # 4. PRICE ACTION MOMENTUM
    # ==========================================
    candle_size = abs(float(last["Close"]) - float(last["Open"]))
    avg_candle = float((df["Close"] - df["Open"]).abs().tail(10).mean())
    
    # Require current candle body to be at least 60% of average candle (slightly more relaxed but still strict)
    if candle_size < (avg_candle * 0.6):
        return False, "Weak price action (Doji or small body)", confidence

    # --- Apply Penalties for Minor Weaknesses ---
    penalties = 0
    if atr < atr_mean: penalties += 5
    if candle_size < avg_candle: penalties += 5
    
    # If RSI is near boundaries, apply penalty
    if direction == "CALL" and rsi_value > 60: penalties += 10
    if direction == "PUT" and rsi_value < 40: penalties += 10
    
    confidence -= penalties
    if confidence < 68:
        return False, f"Confidence dropped below 68 after penalties (-{penalties})", confidence

    return True, "Safety passed", confidence


def _is_strong_martingale(df: pd.DataFrame, direction: str) -> bool:
    if not validate_martingale_signal(df, direction):
        return False

    confidence = calculate_confidence(df, direction)
    return confidence >= 75


def _should_take_signal(df: pd.DataFrame, direction: str, confidence: int, is_next_signal: bool) -> tuple[bool, str, int]:
    safety_ok, safety_reason, confidence = _check_safety_rules(df, direction, confidence)
    if not safety_ok:
        return False, safety_reason, confidence

    # 2. Market Safety Engine (Session, Volatility, Wicks, Sideways)
    market_ok, market_msg, penalty = run_market_safety(df, direction)
    confidence -= penalty
    if not market_ok:
        return False, f"Market Safety: {market_msg}", confidence

    # 3. Live Confirmation Engine (Sudden Reversals, Momentum, Spikes)
    live_ok, live_msg = validate_live_signal(df, direction)
    if not live_ok:
        return False, f"Live Confirmation: {live_msg}", confidence

    # 4. Signal Confirmation (Confidence Threshold)
    # First signal is strict: only high-confidence entries.
    base_threshold = 70 if is_next_signal else 75
    threshold = get_adaptive_trade_threshold(base_threshold)

    if confidence >= threshold:
        return True, f"confidence >= {threshold}", confidence

    if is_next_signal:
        return False, f"next signal confidence below {threshold}", confidence

    return False, f"first signal confidence below {threshold}", confidence


def _is_trade_direction(direction: str) -> bool:
    return str(direction).upper() in {"CALL", "BUY"}


def _get_resolved_trades() -> List[dict]:
    return [entry for entry in manager.tracked_trades if entry.get("resolved")]


def _get_recent_resolved_trades(limit: int = 10) -> List[dict]:
    resolved = _get_resolved_trades()
    resolved.sort(key=lambda entry: entry.get("signal_time") or datetime.min)
    return resolved[-limit:]


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 80:
        return ">=80"
    if confidence >= 70:
        return "70-79"
    return "<70"


def _win_rate(trades: List[dict]) -> float:
    if not trades:
        return 0.0
    wins_count = sum(1 for trade in trades if trade.get("result") == "WIN")
    return (wins_count / len(trades)) * 100


def get_trade_performance() -> dict:
    resolved = _get_resolved_trades()
    resolved_total = len(resolved)
    resolved_wins = sum(1 for trade in resolved if trade.get("result") == "WIN")
    resolved_losses = sum(1 for trade in resolved if trade.get("result") == "LOSS")
    overall_win_rate = (resolved_wins / resolved_total * 100) if resolved_total else 0.0

    confidence_buckets = {">= 80": [], "70-79": [], "<70": []}
    for trade in resolved:
        confidence = trade.get("confidence")
        if confidence is None:
            continue
        confidence_buckets[_confidence_bucket(float(confidence))].append(trade)

    bucket_stats = {}
    for bucket, trades in confidence_buckets.items():
        bucket_total = len(trades)
        bucket_wins = sum(1 for trade in trades if trade.get("result") == "WIN")
        bucket_stats[bucket] = {
            "total": bucket_total,
            "wins": bucket_wins,
            "losses": bucket_total - bucket_wins,
            "win_rate": (bucket_wins / bucket_total * 100) if bucket_total else 0.0,
        }

    # Per-source breakdown
    source_stats: dict = {}
    for trade in resolved:
        src = trade.get("source", "telegram")
        if src not in source_stats:
            source_stats[src] = {"total": 0, "wins": 0}
        source_stats[src]["total"] += 1
        if trade.get("result") == "WIN":
            source_stats[src]["wins"] += 1
    for src, s in source_stats.items():
        s["win_rate"] = (s["wins"] / s["total"] * 100) if s["total"] else 0.0

    recent_10 = _get_recent_resolved_trades(10)

    return {
        "total_trades": resolved_total,
        "wins": resolved_wins,
        "losses": resolved_losses,
        "win_rate": overall_win_rate,
        "recent_10_win_rate": _win_rate(recent_10),
        "confidence_buckets": bucket_stats,
        "source_stats": source_stats,
    }



def get_adaptive_trade_threshold(base_threshold: int = 70) -> int:
    recent_10 = _get_recent_resolved_trades(10)
    if not recent_10:
        return base_threshold

    recent_win_rate = _win_rate(recent_10)
    if recent_win_rate < 50:
        return base_threshold + 5
    if recent_win_rate > 70:
        return 70
    return base_threshold


def store_tracked_signal(
    signal_time: datetime,
    direction: str,
    entry_price: float,
    expiry_time: datetime,
    signal_type: str,
    pair: str,
    confidence: float,
    df: pd.DataFrame,
    source: str = "telegram",
    overlapped_forced: bool = False,
) -> None:
    last = df.iloc[-1]
    manager.tracked_trades.append({
        "signal_time": signal_time,
        "direction": direction,
        "entry_price": float(entry_price),
        "expiry_time": expiry_time,
        "signal_type": signal_type,
        "pair": pair,
        "confidence": float(confidence),
        "source": source,
        "rsi": float(last["RSI"]) if not pd.isna(last["RSI"]) else None,
        "atr": float(last["ATR"]) if not pd.isna(last["ATR"]) else None,
        "ema_trend": _get_ema_trend(df, direction),
        "resolved": False,
        "overlapped_forced": overlapped_forced,
    })

    # Memory management: keep only last 200 trades
    if len(manager.tracked_trades) > 200:
        manager.tracked_trades[:] = manager.tracked_trades[-200:]
    
    # Save to manager JSON for persistence
    manager.save()



def _build_performance_report() -> str:
    stats = get_trade_performance()
    lines = [
        f"📊 Performance Report",
        f"Total Trades: {stats['total_trades']}",
        f"Wins: {stats['wins']}",
        f"Losses: {stats['losses']}",
        f"Win Rate: {stats['win_rate']:.1f}%",
        f"",
        f"Win Rate >=80: {stats['confidence_buckets'].get('>= 80', {}).get('win_rate', 0):.1f}%",
        f"Win Rate 70-79: {stats['confidence_buckets'].get('70-79', {}).get('win_rate', 0):.1f}%",
        f"Win Rate <70: {stats['confidence_buckets'].get('<70', {}).get('win_rate', 0):.1f}%",
    ]
    source_stats = stats.get("source_stats", {})
    if source_stats:
        lines.append("")
        lines.append("*By Source*")
        badges = {"telegram": "📨", "generated": "🤖", "auto": "⚡", "forced": "🔒"}
        for src, s in source_stats.items():
            badge = badges.get(src, "•")
            lines.append(f"{badge} {src.title()}: {s['wins']}W/{s['total'] - s['wins']}L ({s['win_rate']:.0f}%)")
    return "\n".join(lines)


def _maybe_build_daily_report(now: datetime) -> Optional[str]:
    report_date = now.date() - timedelta(days=1)
    if manager.last_daily_report_date == report_date:
        return None

    resolved_for_day = [
        trade for trade in _get_resolved_trades()
        if trade.get("signal_time") is not None and trade["signal_time"].date() == report_date
    ]

    if not resolved_for_day:
        return None

    total = len(resolved_for_day)
    wins = sum(1 for t in resolved_for_day if t.get("result") == "WIN")
    losses = total - wins
    win_rate = (wins / total * 100) if total else 0.0
    best_trades = sum(1 for t in resolved_for_day if float(t.get("confidence") or 0) >= 80 and t.get("result") == "WIN")

    # Per-source breakdown
    source_lines = []
    source_map: dict = {}
    for t in resolved_for_day:
        src = t.get("source", "telegram")
        if src not in source_map:
            source_map[src] = {"w": 0, "l": 0}
        if t.get("result") == "WIN":
            source_map[src]["w"] += 1
        else:
            source_map[src]["l"] += 1
    badges = {"telegram": "📨", "generated": "🤖", "auto": "⚡"}
    for src, s in source_map.items():
        src_total = s["w"] + s["l"]
        src_rate = (s["w"] / src_total * 100) if src_total else 0.0
        badge = badges.get(src, "•")
        source_lines.append(f"{badge} {src.title()}: {s['w']}W/{s['l']}L ({src_rate:.0f}%)")

    manager.last_daily_report_date = report_date
    manager.save()

    report = (
        f"📊 Performance Report\n"
        f"Total Trades: {total}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Best Trades: {best_trades}"
    )
    if source_lines:
        report += "\n\n*By Source*\n" + "\n".join(source_lines)
    return report


def _build_mg_pre_message(signal: dict, confidence: int, df: pd.DataFrame) -> str:
    """Build martingale pre-alert message with indicator analysis."""
    last = df.iloc[-1]
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 0
    ema_trend = _get_ema_trend(df, signal['direction'])
    rsi_interp = _get_rsi_interpretation(rsi)
    atr_strength = _get_atr_strength(df)
    candle_strength = _get_candle_strength(df)
    
    return (
        f"🎲 *MARTINGALE PRE-ALERT*\n\n"
        f"Pair: {signal['pair']}\n"
        f"Direction: {signal['direction']}\n"
        f"Time: {signal['martingale_time']:%H:%M}\n\n"
        f"Confidence: {confidence}%\n\n"
        f"*Technical Analysis*\n"
        f"EMA Trend: {ema_trend}\n"
        f"RSI: {rsi:.0f} ({rsi_interp})\n"
        f"ATR: {atr_strength}\n"
        f"Candle Strength: {candle_strength}\n\n"
        f"Status: Ready for martingale entry"
    )


def _build_mg_confirm_message(signal: dict, confidence: int, expiry: datetime, tp: float, sl: float, df: pd.DataFrame) -> str:
    """Build martingale confirmed message with indicator analysis."""
    last = df.iloc[-1]
    rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 0
    ema_trend = _get_ema_trend(df, signal['direction'])
    rsi_interp = _get_rsi_interpretation(rsi)
    atr_strength = _get_atr_strength(df)
    candle_strength = _get_candle_strength(df)
    
    return (
        f"🎲 *MARTINGALE CONFIRMED*\n\n"
        f"Pair: {signal['pair']}\n"
        f"Direction: {signal['direction']}\n\n"
        f"Confidence: {confidence}%\n\n"
        f"*Technical Analysis*\n"
        f"EMA Trend: {ema_trend}\n"
        f"RSI: {rsi:.0f} ({rsi_interp})\n"
        f"ATR: {atr_strength}\n"
        f"Candle Strength: {candle_strength}\n\n"
        f"*Trade Details*\n"
        f"Entry Time: {signal['martingale_time']:%H:%M}\n"
        f"Expiry: {expiry:%H:%M}\n"
        f"TP: {tp}\n"
        f"SL: {sl}\n\n"
        f"Decision: ENTERING MARTINGALE ✅"
    )


def _format_result_message(entry: dict) -> str:
    """Format result message for a finished trade."""
    pair = entry.get("pair", "EURUSD")
    direction = entry.get("direction")
    entry_price = entry.get("entry_price")
    final_price = entry.get("final_price")
    result = entry.get("result")

    outcome = "✅ WIN" if result == "WIN" else "❌ LOSS"

    return (
        f"📊 RESULT\n\n"
        f"Pair: {pair}\n"
        f"Direction: {direction}\n\n"
        f"Entry: {entry_price:.5f}\n"
        f"Exit: {final_price:.5f}\n\n"
        f"Result: {outcome}"
    )


def _update_stats(entry: dict):
    """Update global stats based on a finished trade."""
    manager.total_trades += 1
    if entry.get("result") == "WIN":
        manager.wins += 1
    else:
        manager.losses += 1



# ─────────────────────────────────────────────────────────────────────────────
# Forced signal message builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_forced_pre_message(signal_time: datetime, direction: str, confidence: int,
                              df: Optional[pd.DataFrame], low_confidence: bool,
                              is_martingale: bool) -> str:
    """PRE-SIGNAL message for forced 15:05 / 15:10 signals."""
    label = "🎲 MARTINGALE PRE-ALERT" if is_martingale else "📊 PRE-SIGNAL"
    risk_tag = "\n⚠️ *LOW CONFIDENCE / RISKY MARKET*" if low_confidence else ""
    analysis = ""
    if df is not None and len(df) >= 2:
        last = df.iloc[-1]
        rsi = float(last["RSI"]) if not pd.isna(last.get("RSI", float("nan"))) else 0
        analysis = (
            f"\n*Technical Analysis*\n"
            f"EMA Trend: {_get_ema_trend(df, direction)}\n"
            f"RSI: {rsi:.0f} ({_get_rsi_interpretation(rsi)})\n"
            f"ATR: {_get_atr_strength(df)}\n"
            f"Candle: {_get_candle_strength(df)}\n"
        )
    return (
        f"{label} ⚡ *FORCED*\n\n"
        f"Pair: EURUSD\n"
        f"Direction: {direction}\n"
        f"Time: {signal_time:%H:%M}\n\n"
        f"Confidence: {confidence}%{risk_tag}\n"
        f"{analysis}\n"
        f"Status: Preparing for entry ⏳"
    )


def _build_forced_confirm_message(signal_time: datetime, direction: str, confidence: int,
                                   expiry: datetime, tp: float, sl: float,
                                   df: Optional[pd.DataFrame], low_confidence: bool,
                                   is_martingale: bool, direction_flipped: bool) -> str:
    """CONFIRMATION message for forced 15:05 / 15:10 signals."""
    label = "🎲 MARTINGALE CONFIRMED" if is_martingale else "📊 SIGNAL CONFIRMED"
    risk_lines = []
    if low_confidence:
        risk_lines.append("⚠️ *LOW CONFIDENCE / RISKY MARKET*")
    if is_martingale and confidence < FORCED_MARTINGALE_MIN_CONFIDENCE:
        risk_lines.append("🔴 *High Risk Martingale*")
    if direction_flipped:
        risk_lines.append("🔄 Direction updated by live confirmation")
    risk_tag = "\n" + "\n".join(risk_lines) if risk_lines else ""
    analysis = ""
    if df is not None and len(df) >= 2:
        last = df.iloc[-1]
        rsi = float(last["RSI"]) if not pd.isna(last.get("RSI", float("nan"))) else 0
        analysis = (
            f"\n*Technical Analysis*\n"
            f"EMA Trend: {_get_ema_trend(df, direction)}\n"
            f"RSI: {rsi:.0f} ({_get_rsi_interpretation(rsi)})\n"
            f"ATR: {_get_atr_strength(df)}\n"
            f"Candle: {_get_candle_strength(df)}\n"
        )
    return (
        f"{label} ⚡ *FORCED*\n\n"
        f"Pair: EURUSD\n"
        f"Direction: {direction}\n"
        f"Source: ⚡ Forced Daily\n"
        f"{risk_tag}\n"
        f"Confidence: {confidence}%\n"
        f"{analysis}\n"
        f"*Trade Details*\n"
        f"Entry Time: {signal_time:%H:%M}\n"
        f"Expiry: {expiry:%H:%M}\n"
        f"TP: {tp}\n"
        f"SL: {sl}"
    )



# ──────────────────────────────────────────────────────────────
# Forced-signal overlap diagnostics (non-blocking)
# ──────────────────────────────────────────────────────────────

# Window used when counting repeated bypasses
_OVERLAP_WARNING_WINDOW_SECONDS = 15 * 60   # 15 minutes
_OVERLAP_WARNING_THRESHOLD = 2              # warn when >= this many in window


def _log_forced_overlap(
    now: datetime,
    forced_signal_time: datetime,
    last_trade_time: datetime,
    elapsed: float,
) -> None:
    """
    Log detailed diagnostics when a forced signal bypasses the global trade lock.

    Tracks every bypass event in ``manager.forced_overlap_times``.
    If 2 or more bypass events occur within 15 minutes, emits an escalated
    WARNING to signal potential over-trading in the forced window.

    Never blocks or modifies the forced signal itself.
    """
    remaining = int(TRADE_LOCK_SECONDS - elapsed)
    overlap_duration = int(elapsed)  # seconds since the previous trade

    logger.warning(
        f"[FORCED] Trade lock bypassed — forced signal at {forced_signal_time:%H:%M} "
        f"fired while lock active | "
        f"Active trade time: {last_trade_time:%H:%M:%S} | "
        f"Overlap duration: {overlap_duration}s | "
        f"Lock remaining: {remaining}s"
    )

    # Record this bypass
    manager.forced_overlap_times.append(now)

    # Prune events outside the 15-minute window
    cutoff = now - timedelta(seconds=_OVERLAP_WARNING_WINDOW_SECONDS)
    manager.forced_overlap_times = [
        t for t in manager.forced_overlap_times if t >= cutoff
    ]

    recent_count = len(manager.forced_overlap_times)
    if recent_count >= _OVERLAP_WARNING_THRESHOLD:
        logger.warning(
            f"[FORCED] ⚠️ OVERLAP WARNING: {recent_count} forced bypass(es) detected "
            f"within the last {_OVERLAP_WARNING_WINDOW_SECONDS // 60} minutes. "
            f"Potential overlapping trades — review forced signal schedule."
        )


def _process_forced_signals(
    messages: List[str],
    active_signals: List[dict],
    minute_df: Optional[pd.DataFrame],
    df: pd.DataFrame,
    now: datetime,
) -> None:
    """
    Process compulsory 15:05 (direct) and 15:10 (martingale) signals.

    Rules:
    • Always sends PRE-SIGNAL 60-120 s before entry.
    • Always sends CONFIRMATION near entry time.
    • Re-checks direction live at confirmation; may flip if original is invalidated.
    • Bypasses processed_signals dedup using 'forced_' prefixed keys.
    • Adds LOW CONFIDENCE / RISKY MARKET or High Risk Martingale warnings as needed.
    • Never skips due to confidence thresholds – only warns.
    """
    work_df = minute_df if (minute_df is not None and len(minute_df) >= 50) else df

    for signal in active_signals:
        try:
            if not signal.get("is_forced"):
                continue

            signal_time: datetime = signal["time"]
            if signal_time.date() != now.date():
                continue

            is_martingale = signal.get("forced_type") == "martingale"
            if signal.get("is_blocked"):
                continue

            # ── ADAPTIVE MARTINGALE GATE (forced 15:10) ─────────────────
            # The forced martingale at 15:10 is still subject to regime gating.
            # A SIDEWAYS/HIGH_VOL/REVERSAL_HEAVY regime blocks MG even here;
            # the user sees a warning rather than a silent skip.
            if is_martingale and abs(seconds_to_entry) <= PRE_SIGNAL_MAX_SECONDS:
                try:
                    _mg_df = work_df if (work_df is not None and len(work_df) >= 50) else df
                    _mg_regime_report = detect_market_regime(_mg_df)
                    _mg_ctrl_result = MartingaleAdaptiveController.evaluate(_mg_regime_report)
                    if not _mg_ctrl_result.allowed:
                        _mg_block_key = f"forced_mg_blocked_{signal_time:%H:%M}"
                        if _mg_block_key not in manager.processed_signals:
                            manager.processed_signals.add(_mg_block_key)
                            _mg_block_msg = (
                                f"⚠️ *Martingale Blocked by Adaptive Regime Control*\n"
                                f"Time: {signal_time:%H:%M} (forced)\n"
                                f"Regime: {_mg_ctrl_result.regime} "
                                f"(conf={_mg_ctrl_result.regime_confidence}%)\n"
                                f"Reason: {_mg_ctrl_result.reason}"
                            )
                            messages.append(_mg_block_msg)
                            logger.info(
                                "[FORCED-MG] Blocked by adaptive controller: %s",
                                _mg_ctrl_result.reason,
                            )
                        signal["is_blocked"] = True
                        continue
                except Exception as _mg_exc:
                    logger.warning("[FORCED-MG] Adaptive gate error (allowing MG): %s", _mg_exc)

            # Use a dedup key that cannot collide with regular signals
            forced_key = f"forced_{signal_time:%H:%M}_{'MG' if is_martingale else 'DIR'}"

            if manager.processed_signals.issuperset({forced_key}):
                continue  # Already confirmed today

            # ── Direction: start with stored, re-decide at confirmation ───
            stored_direction: str = signal["direction"]
            stored_confidence: int = int(signal.get("stored_confidence", 0))
            low_confidence: bool = signal.get("low_confidence", bool(stored_confidence < FORCED_SIGNAL_LOW_CONF_THRESHOLD))

            # ── PRE-SIGNAL (60-120 s before entry) ───────────────────────
            seconds_to_entry = (signal_time - now).total_seconds()
            if PRE_SIGNAL_MIN_SECONDS <= seconds_to_entry <= PRE_SIGNAL_MAX_SECONDS and not signal.get("pre_sent"):
                live_dir, live_conf = decide_direction_live(work_df)
                
                block_reason = None
                if live_dir is None:
                    block_reason = "Missing live direction data"
                else:
                    # ── FORCED_HARD_BLOCK CHECK ──
                    is_news, news_msg, _ = check_high_impact_news()
                    is_dead, atr_msg, _ = check_atr_floor(work_df)
                    
                    if is_news:
                        block_reason = news_msg
                    elif is_dead:
                        block_reason = atr_msg

                if block_reason:
                    logger.warning(f"[FORCED] Signal blocked: {block_reason}")
                    msg = f"⚠️ **Forced signal blocked due to dangerous market conditions**\nReason: {str(block_reason).replace('_', ' ')}"
                    messages.append(msg)
                    signal["is_blocked"] = True
                    continue

                signal["direction"] = live_dir
                signal["stored_confidence"] = live_conf
                low_confidence = live_conf < FORCED_SIGNAL_LOW_CONF_THRESHOLD
                signal["low_confidence"] = low_confidence

                msg = _build_forced_pre_message(signal_time, live_dir, live_conf, work_df, low_confidence, is_martingale)
                messages.append(msg)
                signal["pre_sent"] = True
                signal["pre_confidence"] = live_conf
                logger.info(f"[FORCED] PRE sent: {'MG' if is_martingale else 'DIR'} {signal_time:%H:%M} {live_dir} conf={live_conf}%")

            # ── CONFIRMATION (within SIGNAL_WINDOW_SECONDS of entry) ──────
            if abs(seconds_to_entry) <= SIGNAL_WINDOW_SECONDS and not signal.get("confirmed_sent"):
                # --- Global trade lock check (forced signals may bypass) ---
                is_overlapped = False
                if manager.last_confirmed_trade_time is not None:
                    elapsed = (now - manager.last_confirmed_trade_time).total_seconds()
                    if elapsed < TRADE_LOCK_SECONDS:
                        is_overlapped = True
                        _log_forced_overlap(
                            now=now,
                            forced_signal_time=signal_time,
                            last_trade_time=manager.last_confirmed_trade_time,
                            elapsed=elapsed,
                        )
                        # Forced signals bypass the lock — log but do NOT skip

                live_dir, live_conf = decide_direction_live(work_df)
                
                block_reason = None
                if live_dir is None:
                    block_reason = "Missing live direction data"
                else:
                    # ── FORCED_HARD_BLOCK CHECK (again at confirmation) ──
                    is_news, news_msg, _ = check_high_impact_news()
                    is_dead, atr_msg, _ = check_atr_floor(work_df)
                    
                    if is_news:
                        block_reason = news_msg
                    elif is_dead:
                        block_reason = atr_msg

                if block_reason:
                    logger.warning(f"[FORCED] Confirmation blocked: {block_reason}")
                    if not signal.get("is_blocked"):
                        msg = f"⚠️ **Forced signal blocked due to dangerous market conditions**\nReason: {str(block_reason).replace('_', ' ')}"
                        messages.append(msg)
                    signal["is_blocked"] = True
                    continue

                direction_flipped = live_dir != signal["direction"]
                if direction_flipped:
                    logger.info(f"[FORCED] Direction flipped {signal['direction']} → {live_dir} at confirmation")
                    signal["direction"] = live_dir

                low_confidence = live_conf < FORCED_SIGNAL_LOW_CONF_THRESHOLD
                signal["low_confidence"] = low_confidence

                # Use work_df for TP/SL even if data is thin
                conf_df = work_df if (work_df is not None and len(work_df) >= 2) else df
                try:
                    _, tp, sl = build_forex_targets(conf_df, live_dir, live_conf)
                except Exception as e:
                    atr_val = float(conf_df.iloc[-1]["ATR"]) if conf_df is not None and len(conf_df) >= 1 else 0.0005
                    close_val = float(conf_df.iloc[-1]["Close"]) if conf_df is not None and len(conf_df) >= 1 else 1.0
                    tp = round(close_val + atr_val, 5) if live_dir == "CALL" else round(close_val - atr_val, 5)
                    sl = round(close_val - atr_val, 5) if live_dir == "CALL" else round(close_val + atr_val, 5)

                expiry = signal_time + timedelta(minutes=5)
                msg = _build_forced_confirm_message(
                    signal_time, live_dir, live_conf, expiry, tp, sl,
                    conf_df, low_confidence, is_martingale, direction_flipped
                )
                messages.append(msg)
                signal["confirmed_sent"] = True
                signal["martingale_confidence"] = live_conf
                manager.processed_signals.add(forced_key)
                manager.last_confirmed_trade_time = now  # Update global trade lock

                # Track trade for performance analytics
                try:
                    entry_price = float(conf_df.iloc[-1]["Close"]) if conf_df is not None and len(conf_df) >= 1 else 0.0
                    store_tracked_signal(
                        signal_time=signal_time,
                        direction=live_dir,
                        entry_price=entry_price,
                        expiry_time=expiry,
                        signal_type="martingale" if is_martingale else "direct",
                        pair="EURUSD",
                        confidence=live_conf,
                        df=conf_df,
                        source="forced",
                        overlapped_forced=is_overlapped,
                    )
                except Exception as te:
                    logger.exception(f"[FORCED] store_tracked_signal error: {te}")

                logger.info(
                    f"[FORCED] CONFIRMED: {'MG' if is_martingale else 'DIR'} {signal_time:%H:%M} "
                    f"{live_dir} conf={live_conf}% low={low_confidence}"
                )
        except Exception as e:
            logger.exception(f"[FORCED] Processing error: {e}")


def process_signal_list(
    df: pd.DataFrame,
    minute_data_fetcher: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
) -> List[str]:
    # Load tracked signals from JSON on first call
    if not manager._initialized:
        manager.load()
        manager._initialized = True

    if df is None or len(df) < 200:
        return []

    now = _now()
    messages: List[str] = []

    daily_report = _maybe_build_daily_report(now)
    if daily_report:
        messages.append(daily_report)
        # New trading day detected – prune yesterday's processed_signals
        manager.cleanup_processed_signals()

    # Safety valve: also clean up every time processed_signals grows large
    # (catches long-running days where no midnight rollover occurs locally)
    if len(manager.processed_signals) > manager._PROCESSED_SIGNALS_MAX:
        manager.cleanup_processed_signals()

    # Fetch 1-minute data ONCE and reuse for all operations
    minute_df = None
    if minute_data_fetcher is not None:
        try:
            minute_df = minute_data_fetcher()
            if minute_df is not None and len(minute_df) >= 200:
                minute_df = add_indicators(minute_df)
        except Exception as e:
            minute_df = None

    # 1) Check for any tracked signals that have expired and report results
    try:
        for entry in list(manager.tracked_trades):
            if entry.get("resolved"):
                continue
            expiry = entry.get("expiry_time")
            if expiry is None:
                continue
            if expiry <= now:
                # Use cached 1min data to determine final price; fallback to provided `df`.
                final_data = None
                if minute_df is not None and len(minute_df) >= 1:
                    final_data = minute_df
                elif df is not None and len(df) >= 1:
                    final_data = df

                # Ensure a fallback to `df` so result will always be calculated
                if final_data is None or len(final_data) < 1:
                    final_data = df

                try:
                    if "CandleTime" in final_data.columns:
                        times_naive = pd.to_datetime(final_data["CandleTime"]).dt.tz_localize(None)
                        exp_naive = pd.to_datetime(expiry).tz_localize(None)
                        
                        time_diffs = (times_naive - exp_naive).abs()
                        closest_idx = time_diffs.idxmin()
                        diff_sec = time_diffs.loc[closest_idx].total_seconds()
                        
                        if diff_sec <= 10:
                            final_price = float(final_data.loc[closest_idx]["Close"])
                            logger.debug(f"Expiry candle selected: {final_data.loc[closest_idx]['CandleTime']} for expiry {expiry} (diff: {diff_sec}s)")
                        else:
                            logger.warning(f"Stale expiry candle ({diff_sec}s off). Fallback to last.")
                            final_price = float(final_data.iloc[-1]["Close"])
                    else:
                        final_price = float(final_data.iloc[-1]["Close"])
                except Exception as e:
                    logger.exception(f"Error selecting expiry candle: {e}")
                    try:
                        final_price = float(final_data.iloc[-1]["Close"])
                    except Exception as e:
                        final_price = float(df.iloc[-1]["Close"])
                entry_price = float(entry.get("entry_price"))
                direction = entry.get("direction")

                if _is_trade_direction(direction):
                    is_win = final_price > entry_price
                else:
                    is_win = final_price < entry_price

                entry["final_price"] = final_price
                entry["resolved"] = True
                entry["result"] = "WIN" if is_win else "LOSS"

                # update stats and prepare message
                _update_stats(entry)
                messages.append(_format_result_message(entry))

                # Feed learning engine (per source)
                try:
                    time_str = entry["signal_time"].strftime("%H:%M") if isinstance(entry["signal_time"], datetime) else str(entry["signal_time"])
                    learning_engine.record_trade(
                        time_of_day=time_str,
                        direction=entry.get("direction", "CALL"),
                        confidence=int(entry.get("confidence") or 0),
                        atr=float(entry.get("atr") or 0),
                        rsi=float(entry.get("rsi") or 50),
                        result=entry["result"],
                        source=entry.get("source", "telegram"),
                        regime=entry.get("regime", "UNKNOWN"),
                        probability_score=float(entry.get("probability_score", 0.0))
                    )
                    
                    if "agreement_votes" in entry:
                        record_voter_outcome(entry["agreement_votes"], entry["result"], entry.get("direction", "CALL"))
                        
                    if isinstance(entry["signal_time"], datetime):
                        ist_minutes = entry["signal_time"].hour * 60 + entry["signal_time"].minute
                        record_session_outcome(ist_minutes, entry.get("direction", "CALL"), entry["result"])
                        
                except Exception as le:
                    logger.exception(f"LearningEngine record error: {le}")

                # Save updated tracked signals to JSON
                manager.save()
    except Exception as e:
        logger.exception(f"Result tracking error: {e}")

    # ── CLEAN UP STALE SIGNALS ───────────────────────────────────────────────
    fresh_signals = []
    stale_count = 0
    for sig in manager.active_signals:
        sig_time = sig.get("time")
        stale_after = sig.get("martingale_time") or sig_time
        if stale_after is not None and now > stale_after + timedelta(seconds=SIGNAL_WINDOW_SECONDS + 30):
            stale_count += 1
        else:
            fresh_signals.append(sig)

    if stale_count > 0:
        logger.info(f"Cleaned up {stale_count} stale signal(s) from active queue.")
    manager.active_signals = fresh_signals

    # ── FORCED DAILY SIGNALS (15:05 direct + 15:10 martingale) ──────────────
    # These ALWAYS send PRE + CONFIRMATION regardless of confidence or safety
    # thresholds. Low confidence signals carry a warning, not a skip.
    _process_forced_signals(messages, manager.active_signals, minute_df, df, now)
    # ── END FORCED DAILY SIGNALS ─────────────────────────────────────────────

    for signal in manager.active_signals:
        try:
            # Forced signals (15:05 / 15:10) are handled exclusively by
            # _process_forced_signals() above.  Skip them here to prevent
            # confidence / RSI / momentum filters from rejecting them.
            if signal.get("is_forced"):
                continue

            signal_time = signal["time"]
            direction = signal["direction"]

            if signal_time.date() != now.date():
                continue
            if signal_time.time() < MARKET_OPEN or signal_time.time() > MARKET_CLOSE:
                continue

            base_key = _signal_key(signal_time, direction)

            # Use cached 1-minute data for PRE-SIGNAL confidence when available.
            # Determine whether this is a 'next' signal for thresholding.
            is_next_signal = manager.evaluated_signal_count > 0
            base_threshold = 70 if is_next_signal else 75
            minute_df_local = minute_df if (minute_df is not None and len(minute_df) >= 200) else df

            if base_key not in manager.processed_signals:
                confidence = calculate_confidence(minute_df_local, direction)
                signal["stored_confidence"] = confidence
                seconds_to_signal = (signal_time - now).total_seconds()

                if PRE_SIGNAL_MIN_SECONDS <= seconds_to_signal <= PRE_SIGNAL_MAX_SECONDS and not signal.get("pre_sent"):
                    should_take, skip_reason, confidence = _should_take_signal(
                        minute_df_local, direction, confidence, is_next_signal
                    )
                    if should_take:
                        messages.append(_build_pre_message(signal, confidence, minute_df_local))
                        signal["pre_sent"] = True
                        signal["pre_confidence"] = confidence
                        logger.info(f"PRE-SIGNAL sent: {signal_time:%H:%M} {direction} conf={confidence}%")
                    else:
                        logger.info(f"PRE skipped: {signal_time:%H:%M} {direction} — {skip_reason}")

                if abs(seconds_to_signal) <= SIGNAL_WINDOW_SECONDS and not signal.get("confirmed_sent"):
                    # --- Global trade lock check ---
                    if manager.last_confirmed_trade_time is not None:
                        elapsed = (now - manager.last_confirmed_trade_time).total_seconds()
                        if elapsed < TRADE_LOCK_SECONDS:
                            remaining = int(TRADE_LOCK_SECONDS - elapsed)
                            logger.warning(
                                f"Trade lock active — blocking {signal_time:%H:%M} {direction} "
                                f"({remaining}s remaining). Trade skipped."
                            )
                            manager.processed_signals.add(base_key)
                            manager.evaluated_signal_count += 1
                            continue

                    confidence = calculate_confidence(minute_df_local, direction)
                    signal["stored_confidence"] = confidence
                    logger.debug(f"FINAL CONFIDENCE USED: {confidence}%")

                    should_take, skip_reason, confidence = _should_take_signal(
                        minute_df_local, direction, confidence, is_next_signal
                    )
                    if should_take:
                        _, tp, sl = build_forex_targets(minute_df_local, direction, confidence)
                        expiry = signal_time + timedelta(minutes=5)
                        messages.append(
                            _build_confirm_message(signal, confidence, expiry, tp, sl, minute_df_local)
                        )
                        # Store confirmed signal
                        try:
                            entry_price = float(minute_df_local.iloc[-1]["Close"])
                            store_tracked_signal(
                                signal_time=signal_time,
                                direction=direction,
                                entry_price=entry_price,
                                expiry_time=expiry,
                                signal_type="direct",
                                pair=signal.get("pair", "EURUSD"),
                                confidence=confidence,
                                df=minute_df_local,
                                source=signal.get("source", "telegram"),
                            )
                        except Exception as e:
                            logger.exception(f"store_tracked_signal error: {e}")
                        signal["confirmed_sent"] = True
                        manager.last_confirmed_trade_time = now
                        logger.info(f"Signal confirmed: {signal_time:%H:%M} {direction} conf={confidence}%")
                    else:
                        logger.info(f"Signal skipped: {signal_time:%H:%M} {direction} — {skip_reason}")
                        if len(manager.skip_message_times) < 5:
                            skip_msg = (
                                f"⚠️ *SIGNAL SKIPPED*\n"
                                f"Reason: {skip_reason}"
                            )
                            messages.append(skip_msg)
                            manager.skip_message_times.append(now)
                        else:
                            logger.info(f"Skip message suppressed (rate limit reached) for {signal_time:%H:%M} {direction}")

                manager.processed_signals.add(base_key)
                manager.evaluated_signal_count += 1

            if not manager.enable_martingale:
                continue

            mg_time = signal["martingale_time"]
            mg_key = _signal_key(mg_time, direction)
            seconds_to_mg = (mg_time - now).total_seconds()

            if seconds_to_mg > MARTINGALE_PREALERT_MAX_SECONDS:
                continue

            # ── ADAPTIVE MARTINGALE GATE (regular signals) ────────────────────
            # Evaluate live regime before allowing any martingale pre-alert or
            # confirmation.  Only TRENDING with strong confidence passes through.
            try:
                _reg_df = minute_df if (minute_df is not None and len(minute_df) >= 50) else df
                _mg_result = MartingaleAdaptiveController.evaluate_from_df(_reg_df)
                if not _mg_result.allowed:
                    _mg_skip_key = f"mg_regime_blocked_{signal_time:%H:%M}_{direction}"
                    if _mg_skip_key not in manager.processed_signals:
                        manager.processed_signals.add(_mg_skip_key)
                        logger.info(
                            "[MG-Gate] Martingale skipped for %s %s — %s",
                            signal_time.strftime("%H:%M"), direction, _mg_result.reason,
                        )
                    continue
            except Exception as _mg_gate_exc:
                logger.warning("[MG-Gate] Adaptive check error (allowing MG): %s", _mg_gate_exc)

            if (
                MARTINGALE_PREALERT_MIN_SECONDS <= seconds_to_mg <= MARTINGALE_PREALERT_MAX_SECONDS
                and not signal.get("martingale_prealert_sent")
            ):
                # Prefer 1-minute data for martingale checks; fallback to main df
                mg_df = minute_df if (minute_df is not None and len(minute_df) >= 200) else df
                if _is_strong_martingale(mg_df, direction):
                    mg_confidence = calculate_confidence(mg_df, direction)
                    signal["martingale_confidence"] = mg_confidence
                    messages.append(_build_mg_pre_message(signal, mg_confidence, mg_df))
                    logger.debug(f"FINAL CONFIDENCE USED: {mg_confidence}%")
                    signal["martingale_prealert_sent"] = True

            if abs((now - mg_time).total_seconds()) <= SIGNAL_WINDOW_SECONDS:
                # Prefer 1-minute data for martingale confirmation; fallback to main df
                mg_df = minute_df if (minute_df is not None and len(minute_df) >= 200) else df
                if mg_key not in manager.processed_signals and _is_strong_martingale(mg_df, direction):
                    # --- Global trade lock check (martingale) ---
                    if manager.last_confirmed_trade_time is not None:
                        elapsed = (now - manager.last_confirmed_trade_time).total_seconds()
                        if elapsed < TRADE_LOCK_SECONDS:
                            remaining = int(TRADE_LOCK_SECONDS - elapsed)
                            logger.warning(
                                f"Trade lock active — blocking martingale {mg_time:%H:%M} {direction} "
                                f"({remaining}s remaining). Trade skipped."
                            )
                            manager.processed_signals.add(mg_key)
                            continue

                    mg_confidence = calculate_confidence(mg_df, direction)
                    signal["martingale_confidence"] = mg_confidence
                    logger.debug(f"FINAL CONFIDENCE USED: {mg_confidence}%")
                    _, tp, sl = build_forex_targets(mg_df, direction, mg_confidence)
                    expiry = mg_time + timedelta(minutes=5)
                    messages.append(
                        _build_mg_confirm_message(
                            signal,
                            mg_confidence,
                            expiry,
                            tp,
                            sl,
                            mg_df,
                        )
                    )
                    # Store martingale confirmed signal
                    try:
                        entry_price = float(mg_df.iloc[-1]["Close"])
                        store_tracked_signal(
                            signal_time=mg_time,
                            direction=direction,
                            entry_price=entry_price,
                            expiry_time=expiry,
                            signal_type="martingale",
                            pair=signal.get("pair", "EURUSD"),
                            confidence=mg_confidence,
                            df=mg_df,
                            source=signal.get("source", "telegram"),
                        )
                    except Exception as e:
                        logger.exception(f"store_tracked_signal error: {e}")
                    manager.processed_signals.add(mg_key)
                    signal["martingale_confirmed_sent"] = True
                    manager.last_confirmed_trade_time = now  # Update global trade lock
                    logger.info(f"Signal confirmed: {mg_time:%H:%M} {direction}")
        except Exception as e:
            logger.exception(f"Processing error: {e}")
            continue

    return messages

