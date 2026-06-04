"""
session_intelligence.py
───────────────────────
Advanced Session Intelligence for the trading bot.

Detects the current Forex trading session from IST (UTC+5:30) time and
provides adaptive session metrics that feed directly into the probability
scoring formula via session_strength.

Sessions (UTC-based, converted to IST):
  ┌──────────────────────┬─────────────────┬─────────────────────┐
  │ Session              │ UTC             │ IST (UTC+5:30)      │
  ├──────────────────────┼─────────────────┼─────────────────────┤
  │ Asian                │ 00:00 – 09:00   │ 05:30 – 14:30       │
  │ London               │ 07:00 – 16:00   │ 12:30 – 21:30       │
  │ London–NY Overlap    │ 13:00 – 16:00   │ 18:30 – 21:30       │
  │ New York             │ 13:00 – 22:00   │ 18:30 – 03:30+1     │
  └──────────────────────┴─────────────────┴─────────────────────┘

  Note: Sessions overlap — a time in 18:30–21:30 IST is BOTH London,
  New York, AND the Overlap.  The overlap is the primary label for
  that period because it is historically the most liquid window.

Performance tracking:
  - Rolling window (last 60 trades) per session × direction
  - Tracks win rate, volatility quality, continuation strength
  - Persisted to session_performance.json (atomic writes, 3 backups)
  - Daily summary updated after each EOD result batch

Confidence formula:
  session_strength (0-100) =
      base_strength(session)           × 0.40   (structural liquidity)
    + adaptive_win_rate(session, dir)  × 0.35   (historical accuracy)
    + volatility_quality_bonus         × 0.15   (vol quality)
    + continuation_bonus               × 0.10   (trend continuation)

  Clamped to [20, 100] so even weak sessions remain tradeable.

Usage:
    from session_intelligence import session_intel
    info = session_intel.get_session_info(ist_minutes)
    strength = session_intel.compute_session_strength(ist_minutes, direction, df)
    session_intel.record_session_outcome(ist_minutes, direction, result, df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from logger import logger
from persistence import safe_load_json, safe_save_json

# ─── Constants ────────────────────────────────────────────────────────────────

SESSION_PERFORMANCE_FILE = "session_performance.json"

# Session labels
SESSION_ASIAN    = "Asian"
SESSION_LONDON   = "London"
SESSION_OVERLAP  = "London_NY_Overlap"
SESSION_NEW_YORK = "New_York"
SESSION_OFF      = "Off_Hours"

ALL_SESSIONS = [SESSION_ASIAN, SESSION_LONDON, SESSION_OVERLAP, SESSION_NEW_YORK, SESSION_OFF]

# Rolling window per session
ROLLING_WINDOW = 60

# Minimum trades before adaptive adjustment kicks in
MIN_TRADES_FOR_ADAPTATION = 8

# Session windows in IST minutes (from midnight)
# IST = UTC + 5:30  →  UTC 00:00 = IST 05:30 = 330 min from midnight IST
_IST_OFFSET = 5 * 60 + 30   # 330 minutes

def _utc_to_ist_min(h: int, m: int = 0) -> int:
    """Convert UTC hours+minutes to IST minutes-from-midnight."""
    return (h * 60 + m + _IST_OFFSET) % (24 * 60)

# Session windows: (start_ist_min, end_ist_min)  — end is exclusive
_ASIAN_START    = _utc_to_ist_min(0,  0)    # 05:30 IST
_ASIAN_END      = _utc_to_ist_min(9,  0)    # 14:30 IST
_LONDON_START   = _utc_to_ist_min(7,  0)    # 12:30 IST
_LONDON_END     = _utc_to_ist_min(16, 0)    # 21:30 IST
_OVERLAP_START  = _utc_to_ist_min(13, 0)    # 18:30 IST
_OVERLAP_END    = _utc_to_ist_min(16, 0)    # 21:30 IST
_NY_START       = _utc_to_ist_min(13, 0)    # 18:30 IST
_NY_END         = _utc_to_ist_min(22, 0)    # 03:30 IST next day → 330+1320 = handled mod

# Base structural strength for each session (reflects typical EURUSD liquidity)
# These are priors that adapt based on actual trade performance.
_BASE_STRENGTH: dict[str, float] = {
    SESSION_ASIAN:    42.0,   # Quiet — low EURUSD liquidity
    SESSION_LONDON:   80.0,   # Active — EU market hours
    SESSION_OVERLAP:  96.0,   # Most liquid — both EU+US open
    SESSION_NEW_YORK: 78.0,   # Active — but after EU close it quiets
    SESSION_OFF:      20.0,   # No major session
}

# Max adaptive drift from base strength (±25 pts)
_MAX_ADAPTIVE_DRIFT = 25.0


# ─── Session detection ────────────────────────────────────────────────────────

def _in_window(ist_minutes: int, start: int, end: int) -> bool:
    """Check if `ist_minutes` is within [start, end), handling midnight wrap."""
    if start <= end:
        return start <= ist_minutes < end
    else:
        # wraps past midnight (e.g. NY end at 03:30 IST next day)
        return ist_minutes >= start or ist_minutes < end


def detect_session(ist_minutes: int) -> str:
    """
    Return the primary session label for an IST minutes-from-midnight value.

    Priority order (highest liquidity wins when overlapping):
      Overlap > New York > London > Asian > Off_Hours
    """
    if _in_window(ist_minutes, _OVERLAP_START, _OVERLAP_END):
        return SESSION_OVERLAP
    if _in_window(ist_minutes, _NY_START, _NY_END):
        return SESSION_NEW_YORK
    if _in_window(ist_minutes, _LONDON_START, _LONDON_END):
        return SESSION_LONDON
    if _in_window(ist_minutes, _ASIAN_START, _ASIAN_END):
        return SESSION_ASIAN
    return SESSION_OFF


def get_session_windows_ist() -> dict[str, tuple[str, str]]:
    """Return human-readable session windows in IST for display/logging."""
    def _fmt(m: int) -> str:
        return f"{m // 60:02d}:{m % 60:02d}"
    return {
        SESSION_ASIAN:    (_fmt(_ASIAN_START),   _fmt(_ASIAN_END)),
        SESSION_LONDON:   (_fmt(_LONDON_START),  _fmt(_LONDON_END)),
        SESSION_OVERLAP:  (_fmt(_OVERLAP_START), _fmt(_OVERLAP_END)),
        SESSION_NEW_YORK: (_fmt(_NY_START),       "03:30 (+1)"),
        SESSION_OFF:      ("—",                  "—"),
    }


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    """Single trade outcome stored in session history."""
    timestamp:   str
    session:     str
    direction:   str
    result:      str       # "WIN" or "LOSS"
    vol_quality: float     # 0-100 (ATR quality at time of trade)
    continuation: float    # 0-100 (trend continuation metric)


@dataclass
class SessionStats:
    """Aggregated statistics for one session."""
    session: str
    history: list = field(default_factory=list)   # list of dicts

    @property
    def _window(self) -> list:
        return self.history[-ROLLING_WINDOW:]

    def rolling_win_rate(self, direction: Optional[str] = None) -> float:
        """Win rate over last ROLLING_WINDOW, optionally filtered by direction."""
        window = self._window
        if direction:
            window = [e for e in window if e.get("direction") == direction]
        if not window:
            return 50.0
        wins = sum(1 for e in window if e.get("result") == "WIN")
        return round(wins / len(window) * 100.0, 2)

    def avg_vol_quality(self) -> float:
        window = self._window
        if not window:
            return 70.0
        return round(float(np.mean([e.get("vol_quality", 70.0) for e in window])), 2)

    def avg_continuation(self) -> float:
        window = self._window
        if not window:
            return 60.0
        return round(float(np.mean([e.get("continuation", 60.0) for e in window])), 2)

    def sample_size(self, direction: Optional[str] = None) -> int:
        window = self._window
        if direction:
            window = [e for e in window if e.get("direction") == direction]
        return len(window)

    def as_summary_dict(self) -> dict:
        return {
            "session":               self.session,
            "win_rate_call":         self.rolling_win_rate("CALL"),
            "win_rate_put":          self.rolling_win_rate("PUT"),
            "win_rate_overall":      self.rolling_win_rate(),
            "avg_vol_quality":       self.avg_vol_quality(),
            "avg_continuation":      self.avg_continuation(),
            "sample_call":           self.sample_size("CALL"),
            "sample_put":            self.sample_size("PUT"),
            "total_trades":          len(self.history),
        }


# ─── Main class ───────────────────────────────────────────────────────────────

class SessionIntelligence:
    """
    Advanced Session Intelligence Engine.

    Detects current session, computes adaptive session_strength, and
    tracks per-session performance across rolling trade history.
    """

    def __init__(self, perf_file: str = SESSION_PERFORMANCE_FILE):
        self.perf_file = perf_file
        self._stats: dict[str, SessionStats] = {
            s: SessionStats(session=s) for s in ALL_SESSIONS
        }
        self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = safe_load_json(self.perf_file, default={})
        for session in ALL_SESSIONS:
            sv = data.get(session, {})
            self._stats[session].history = sv.get("history", [])
        self._trim_history()
        logger.debug("[SessionIntel] Loaded — sessions: %s",
                     {s: self._stats[s].sample_size() for s in ALL_SESSIONS})

    def _save(self) -> None:
        payload = {
            session: {
                "history":  stats.history[-ROLLING_WINDOW * 2:],  # keep 2× window max
                "summary":  stats.as_summary_dict(),
            }
            for session, stats in self._stats.items()
        }
        payload["_meta"] = {
            "last_saved":    datetime.now().isoformat(),
            "rolling_window": ROLLING_WINDOW,
        }
        safe_save_json(self.perf_file, payload, indent=2)

    def _trim_history(self) -> None:
        """Remove entries older than 30 days."""
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        for stats in self._stats.values():
            stats.history = [
                e for e in stats.history
                if e.get("timestamp", "") >= cutoff
            ]

    # ── Session detection ─────────────────────────────────────────────────────

    def get_session_info(self, ist_minutes: int) -> dict:
        """
        Return full session context for a given IST time.

        Returns:
            session:        primary session label
            base_strength:  structural liquidity strength (0-100)
            is_overlap:     True if London-NY overlap
            sessions_active: list of all active sessions at this time
        """
        primary = detect_session(ist_minutes)
        active = []
        if _in_window(ist_minutes, _ASIAN_START, _ASIAN_END):
            active.append(SESSION_ASIAN)
        if _in_window(ist_minutes, _LONDON_START, _LONDON_END):
            active.append(SESSION_LONDON)
        if _in_window(ist_minutes, _OVERLAP_START, _OVERLAP_END):
            active.append(SESSION_OVERLAP)
        if _in_window(ist_minutes, _NY_START, _NY_END):
            active.append(SESSION_NEW_YORK)
        if not active:
            active.append(SESSION_OFF)

        return {
            "session":         primary,
            "base_strength":   _BASE_STRENGTH[primary],
            "is_overlap":      primary == SESSION_OVERLAP,
            "sessions_active": active,
        }

    # ── Adaptive strength computation ─────────────────────────────────────────

    def compute_session_strength(
        self,
        ist_minutes: int,
        direction: str,
        df: Optional[pd.DataFrame] = None,
    ) -> tuple[float, dict]:
        """
        Compute adaptive session_strength (0–100) for a candidate signal.

        Formula:
            strength = base_strength(session)         × 0.40
                     + adaptive_win_rate(sess, dir)   × 0.35
                     + vol_quality_bonus               × 0.15
                     + continuation_bonus              × 0.10

        The adaptive component shifts based on the rolling session win rate.
        With no history (cold start) the adaptive component = 50 (neutral).

        Args:
            ist_minutes: signal time in IST minutes from midnight
            direction:   "CALL" or "PUT"
            df:          optional enriched DataFrame (for live vol/continuation)

        Returns:
            (strength_0_100, detail_dict)
        """
        info    = self.get_session_info(ist_minutes)
        session = info["session"]
        stats   = self._stats.get(session, SessionStats(session=session))

        # ── Component 1: Base structural strength ────────────────────
        base = _BASE_STRENGTH[session]

        # ── Component 2: Adaptive win rate (direction-aware) ─────────
        n_dir  = stats.sample_size(direction)
        n_all  = stats.sample_size()

        if n_dir >= MIN_TRADES_FOR_ADAPTATION:
            adaptive_wr = stats.rolling_win_rate(direction)
        elif n_all >= MIN_TRADES_FOR_ADAPTATION:
            # Fall back to overall session win rate
            adaptive_wr = stats.rolling_win_rate()
        else:
            adaptive_wr = 50.0   # neutral cold start

        # ── Component 3: Volatility quality ──────────────────────────
        if n_all >= MIN_TRADES_FOR_ADAPTATION:
            vol_bonus = stats.avg_vol_quality()
        elif df is not None:
            vol_bonus = _compute_live_vol_quality(df, ist_minutes)
        else:
            vol_bonus = 70.0   # assume decent by default

        # ── Component 4: Continuation strength ───────────────────────
        if n_all >= MIN_TRADES_FOR_ADAPTATION:
            cont_bonus = stats.avg_continuation()
        elif df is not None:
            cont_bonus = _compute_live_continuation(df, ist_minutes)
        else:
            cont_bonus = 60.0

        # ── Blended strength ─────────────────────────────────────────
        strength_raw = (
            base           * 0.40
          + adaptive_wr    * 0.35
          + vol_bonus      * 0.15
          + cont_bonus     * 0.10
        )

        # Adaptive drift cap: adaptive component can shift base ±MAX_DRIFT
        # so very poor session history can reduce but not zero-out
        drift = (adaptive_wr - 50.0) * 0.35   # max ±17.5 from adaptive
        clamped_drift = max(-_MAX_ADAPTIVE_DRIFT, min(_MAX_ADAPTIVE_DRIFT, drift))
        strength_final = max(20.0, min(100.0, strength_raw))

        detail = {
            "session":            session,
            "sessions_active":    info["sessions_active"],
            "is_overlap":         info["is_overlap"],
            "base_strength":      round(base, 1),
            "adaptive_win_rate":  round(adaptive_wr, 1),
            "vol_bonus":          round(vol_bonus, 1),
            "continuation_bonus": round(cont_bonus, 1),
            "adaptive_drift":     round(clamped_drift, 2),
            "sample_size_dir":    n_dir,
            "sample_size_all":    n_all,
            "strength":           round(strength_final, 2),
        }

        logger.debug(
            "[SessionIntel] %s %s — session=%s base=%.0f wr=%.0f vol=%.0f "
            "cont=%.0f → strength=%.1f",
            _fmt_ist(ist_minutes), direction, session,
            base, adaptive_wr, vol_bonus, cont_bonus, strength_final,
        )

        return strength_final, detail

    # ── Trade outcome recording ───────────────────────────────────────────────

    def record_session_outcome(
        self,
        ist_minutes: int,
        direction: str,
        result: str,
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Record a completed trade outcome for session-level adaptive learning.

        Args:
            ist_minutes: signal IST time in minutes from midnight
            direction:   "CALL" or "PUT"
            result:      "WIN" or "LOSS"
            df:          optional enriched DataFrame (for vol/cont metrics)
        """
        if result not in ("WIN", "LOSS"):
            return

        session    = detect_session(ist_minutes)
        vol_quality = _compute_live_vol_quality(df, ist_minutes) if df is not None else 70.0
        cont_str    = _compute_live_continuation(df, ist_minutes) if df is not None else 60.0

        entry = {
            "timestamp":    datetime.now().isoformat(),
            "session":      session,
            "direction":    direction,
            "result":       result,
            "vol_quality":  round(vol_quality, 2),
            "continuation": round(cont_str, 2),
            "ist_time":     _fmt_ist(ist_minutes),
        }
        self._stats[session].history.append(entry)
        self._save()

        logger.info(
            "[SessionIntel] Recorded %s %s %s → session=%s (n=%d, wr=%.0f%%)",
            _fmt_ist(ist_minutes), direction, result,
            session,
            self._stats[session].sample_size(),
            self._stats[session].rolling_win_rate(direction),
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_performance_report(self) -> str:
        """Return formatted performance report across all sessions."""
        lines = ["Session Intelligence Performance Report"]
        lines.append(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
        lines.append("")

        windows = get_session_windows_ist()
        for session in ALL_SESSIONS:
            stats  = self._stats[session]
            wr_all = stats.rolling_win_rate()
            wr_c   = stats.rolling_win_rate("CALL")
            wr_p   = stats.rolling_win_rate("PUT")
            n_all  = stats.sample_size()
            n_c    = stats.sample_size("CALL")
            n_p    = stats.sample_size("PUT")
            start, end = windows.get(session, ("?", "?"))
            base   = _BASE_STRENGTH[session]

            lines.append(f"  [{session}]  IST {start}–{end}")
            lines.append(f"    Base={base:.0f}  WR_all={wr_all:.0f}%  "
                         f"WR_CALL={wr_c:.0f}%(n={n_c})  "
                         f"WR_PUT={wr_p:.0f}%(n={n_p})  "
                         f"Total={n_all}")
        return "\n".join(lines)

    def get_telegram_summary(self) -> str:
        """Short Telegram summary of session win rates."""
        lines = ["Session Performance"]
        emoji = {
            SESSION_ASIAN:    "🌏",
            SESSION_LONDON:   "🇬🇧",
            SESSION_OVERLAP:  "⚡",
            SESSION_NEW_YORK: "🗽",
            SESSION_OFF:      "💤",
        }
        for session in ALL_SESSIONS:
            if session == SESSION_OFF:
                continue
            stats = self._stats[session]
            n     = stats.sample_size()
            wr    = stats.rolling_win_rate()
            e     = emoji.get(session, "")
            lines.append(
                f"  {e} {session.replace('_', ' ')}: "
                f"{wr:.0f}% (n={n})"
            )
        return "\n".join(lines)

    def get_all_stats(self) -> dict:
        """Return all session stats as a dict for JSON serialisation."""
        return {s: self._stats[s].as_summary_dict() for s in ALL_SESSIONS}


# ─── Live df helpers ──────────────────────────────────────────────────────────

def _compute_live_vol_quality(df: pd.DataFrame, ist_minutes: int) -> float:
    """
    Estimate volatility quality from the most recent candles in df.
    Returns 0-100.
    """
    try:
        if df is None or len(df) < 10:
            return 70.0
        atr = pd.to_numeric(df["ATR"], errors="coerce").dropna()
        if len(atr) < 5:
            return 70.0
        atr_now    = float(atr.iloc[-1])
        atr_median = float(atr.tail(96).median())
        if atr_median <= 0:
            return 70.0
        ratio = atr_now / atr_median
        # Ideal: 0.75–1.55 → 100, tails → 0
        if 0.75 <= ratio <= 1.55:
            return 100.0
        if ratio < 0.45:
            return 0.0
        if ratio > 2.50:
            return 0.0
        if ratio < 0.75:
            return ((ratio - 0.45) / 0.30) * 100
        return max(0.0, ((2.50 - ratio) / 0.95) * 100)
    except Exception:
        return 70.0


def _compute_live_continuation(df: pd.DataFrame, ist_minutes: int) -> float:
    """
    Estimate trend continuation strength from recent candles.
    Returns 0-100.
    """
    try:
        if df is None or len(df) < 10:
            return 60.0
        tail = df.tail(20)
        direction_col = (tail["Close"] > tail["Open"]).astype(int)
        cont = (direction_col == direction_col.shift(1)) & direction_col.shift(1).notna()
        return round(float(cont.mean()) * 100, 2)
    except Exception:
        return 60.0


def _fmt_ist(ist_minutes: int) -> str:
    """Format IST minutes-from-midnight as HH:MM string."""
    h = (ist_minutes // 60) % 24
    m = ist_minutes % 60
    return f"{h:02d}:{m:02d}"


# ─── Global singleton ─────────────────────────────────────────────────────────
session_intel = SessionIntelligence()
