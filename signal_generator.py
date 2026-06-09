from logger import logger
import os
import json
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from indicators import add_indicators
from learning_engine import learning_engine, dynamic_weight_optimizer
from market_safety import run_market_safety
from market_regime import detect_market_regime, get_regime_behavior, classify_volatility_zone
from probability_engine import (
    probability_engine,
    ProbabilityInputs,
    TIER_STRONG,
    TIER_MODERATE,
    TIER_SKIP,
    _regime_min_score,
)
from persistence import safe_load_json, safe_save_json
from signal_manager import timing_db
from agreement_engine import agreement_engine, AGREEMENT_SKIP, AGREEMENT_STRONG, AGREEMENT_MODERATE
from backtesting_engine import BacktestingEngine
from sequence_engine import sequence_engine, get_sequence_momentum_bonus
from strategy_weight_tracker import strategy_weight_tracker
from session_intelligence import session_intel, detect_session, SESSION_OVERLAP, SESSION_LONDON, SESSION_NEW_YORK, SESSION_ASIAN

# ── Configuration ─────────────────────────────────────────
PAIR = "EURUSD"
SYMBOL = "EUR/USD"
INTERVAL = "5min"
SIGNAL_FILE = "generated_signals.json"
STATE_FILE = ".generator_state.json"
DF_CACHE_FILE = ".df_cache.pkl"

FORCE_DIRECT_TIME = "15:05"
FORCE_MARTINGALE_TIME = "15:10"
FORCE_SIGNAL_CONFIDENCE_THRESHOLD = 55

# Pattern engine thresholds
MIN_SLOT_OCCURRENCES = 8        # minimum candles per time slot across 14 days
BASE_CONFIDENCE_THRESHOLD = 72  # minimum composite score to qualify
PATTERN_STRENGTH_THRESHOLD = 60 # minimum pattern strength from timing_db
MIN_VOLATILITY_CLUSTER_SCORE = 55
MIN_RECURRING_SIGNAL_GAP_MINUTES = 10
UNSTABLE_SIGNAL_GAP_MINUTES = 20

# ── Probability scoring (centralized via probability_engine.py) ───────
# Formula (v2 — balanced safe-profit):
#   score = win_rate*0.32 + direction_consistency*0.25 + momentum_strength*0.15
#         + atr_quality*0.12 + session_strength*0.08
#         + volatility_quality*0.08 - reversal_risk*0.06
#         × regime_multiplier
#
# Dynamic min score: TRENDING=70 | SIDEWAYS=78 | HIGH_VOL/REVERSAL=80
# Tiers: STRONG_SIGNAL >= 80 | MODERATE_SIGNAL >= 70 | SKIP < 70
#
# All routing via probability_engine.compute(ProbabilityInputs(...))
_SCORE_TIERS = {TIER_STRONG: "STRONG", TIER_MODERATE: "MODERATE", TIER_SKIP: "SKIP"}


# ── Signal Diagnostics ────────────────────────────────────────────────────────────
# Stored after each generation run; consumed by bot.py for Telegram summary.
from dataclasses import dataclass, field as dc_field

# Rejection reason categories (used for grouping in the Telegram summary)
REJECT_VOL_ZONE         = "vol_zone"
REJECT_ATR_RATIO        = "atr_ratio"
REJECT_PROB_SCORE       = "prob_score"
REJECT_REVERSAL_HEAVY   = "reversal_heavy"
REJECT_AGREEMENT        = "agreement"
REJECT_LEARNING_VETO    = "learning_veto"
REJECT_DB_STRENGTH      = "db_strength"
REJECT_LIVE_CONF        = "live_confirmation"
REJECT_ADJUSTED_CONF    = "adjusted_conf"
REJECT_OTHER            = "other"

_REJECT_CATEGORY_LABELS = {
    REJECT_VOL_ZONE:       "Volatility Zone",
    REJECT_ATR_RATIO:      "ATR Ratio",
    REJECT_PROB_SCORE:     "Prob Score (regime floor)",
    REJECT_REVERSAL_HEAVY: "Reversal-Heavy Gate",
    REJECT_AGREEMENT:      "Agreement Engine",
    REJECT_LEARNING_VETO:  "Learning Engine Veto",
    REJECT_DB_STRENGTH:    "DB Strength + Confidence",
    REJECT_LIVE_CONF:      "Live Confirmation",
    REJECT_ADJUSTED_CONF:  "Adjusted Confidence Low",
    REJECT_OTHER:          "Other",
}


@dataclass
class SignalDiagnostics:
    """
    Captures full telemetry from one `calculate_recurring_strength()` run.
    Stored in `_last_diagnostics` and consumed by bot.py for Telegram.
    """
    date:                  str   = ""
    regime:                str   = ""
    regime_confidence:     float = 0.0
    currency_bias:         str   = "NEUTRAL"
    currency_eur:          float = 50.0
    currency_usd:          float = 50.0
    sweep_detected:        bool  = False
    sweep_direction:       str   = ""
    generated_candidates:  int   = 0   # total slot × direction pairs evaluated
    accepted_count:        int   = 0   # passed all gates
    final_count:           int   = 0   # after cooldown + ranking
    rejection_counts: dict = dc_field(default_factory=dict)  # {category: count}
    rejection_details: list = dc_field(default_factory=list) # [(time,dir,reason)]
    strong_count:      int  = 0
    moderate_count:    int  = 0
    avg_prob_score:    float = 0.0
    top_times:         list = dc_field(default_factory=list)  # [(time, dir, score)]

    def rejection_rate(self) -> float:
        if self.generated_candidates == 0:
            return 0.0
        return round(
            (self.generated_candidates - self.accepted_count)
            / self.generated_candidates * 100, 1
        )

    def dominant_reject_reason(self) -> str:
        if not self.rejection_counts:
            return "None"
        key = max(self.rejection_counts, key=self.rejection_counts.get)
        return _REJECT_CATEGORY_LABELS.get(key, key)


def _categorise_rejection(reason: str) -> str:
    """Map a free-text rejection reason to a canonical category key."""
    r = reason.lower()
    if "vol_zone" in r:              return REJECT_VOL_ZONE
    if "atr_ratio" in r:             return REJECT_ATR_RATIO
    if "prob_score" in r or "regime_floor" in r: return REJECT_PROB_SCORE
    if "reversal-heavy" in r:        return REJECT_REVERSAL_HEAVY
    if "agreement" in r:             return REJECT_AGREEMENT
    if "learning veto" in r:         return REJECT_LEARNING_VETO
    if "db strength" in r:           return REJECT_DB_STRENGTH
    if "live confirmation" in r:     return REJECT_LIVE_CONF
    if "adjusted_conf" in r:         return REJECT_ADJUSTED_CONF
    return REJECT_OTHER


# Module-level store: set by calculate_recurring_strength(), read by bot.py
_last_diagnostics: SignalDiagnostics | None = None


def get_last_diagnostics() -> "SignalDiagnostics | None":
    """Return the diagnostics from the most recent generation run, or None."""
    return _last_diagnostics


def format_diagnostic_summary(d: "SignalDiagnostics") -> str:
    """
    Format a Telegram-ready diagnostic summary for one generation run.
    Shows the full funnel: candidates → accepted → final, rejection
    breakdown by category, and top signals by probability score.
    """
    lines: list[str] = []
    sep = "━" * 26

    # ── Header ───────────────────────────────────────────────────────────
    lines.append(sep)
    lines.append("\U0001f4ca *Signal Diagnostics* — " + d.date)
    lines.append(sep)

    # ── Market context ───────────────────────────────────────────────
    regime_conf_pct = int(d.regime_confidence)
    lines.append(f"Regime: *{str(d.regime).replace('_', ' ')}* ({regime_conf_pct}% conf)")
    # Currency strength
    bias_arrow = "↑" if d.currency_bias == "CALL" else "↓" if d.currency_bias == "PUT" else "→"
    lines.append(
        f"Currency: EUR={d.currency_eur:.1f} USD={d.currency_usd:.1f} {bias_arrow}{d.currency_bias}"
    )
    # Sweep
    if d.sweep_detected:
        lines.append(f"Sweep: \U0001f30a {d.sweep_direction} detected")
    else:
        lines.append("Sweep: None")

    lines.append("")

    # ── Signal funnel ──────────────────────────────────────────────────
    rejected_count = d.generated_candidates - d.accepted_count
    lines.append("*Signal Funnel*")
    lines.append(f"  Generated candidates : {d.generated_candidates}")
    lines.append(f"  Passed all gates     : {d.accepted_count}")
    lines.append(f"  Rejected             : {rejected_count} ({d.rejection_rate()}%)")
    lines.append(f"  Final (post-rank)    : *{d.final_count}*")
    if d.accepted_count > 0:
        lines.append(f"  Avg prob score       : {d.avg_prob_score:.1f}")
        strong_pct = int(d.strong_count / d.accepted_count * 100)
        lines.append(f"  STRONG / MODERATE    : {d.strong_count} / {d.moderate_count} ({strong_pct}% strong)")

    lines.append("")

    # ── Rejection breakdown ────────────────────────────────────────────
    if d.rejection_counts:
        lines.append("*Rejection Breakdown*")
        sorted_rejects = sorted(d.rejection_counts.items(), key=lambda x: x[1], reverse=True)
        for cat_key, count in sorted_rejects:
            label = _REJECT_CATEGORY_LABELS.get(cat_key, cat_key)
            bar = "█" * min(10, count)
            lines.append(f"  {label}: {count} {bar}")
        lines.append(f"  [Dominant filter: {d.dominant_reject_reason()}]")
        lines.append("")

    # ── Top 3 accepted signals (by prob score) ─────────────────────────
    if d.top_times:
        lines.append("*Top Accepted Signals*")
        for t, direction, score in d.top_times[:3]:
            emoji = "\U0001f4c8" if direction == "CALL" else "\U0001f4c9"
            lines.append(f"  {emoji} {t} {direction} — prob={score:.1f}")
        lines.append("")

    # ── Over-filtering alert ───────────────────────────────────────────
    if d.final_count == 0:
        lines.append("\u26a0\ufe0f *ALERT: 0 signals today — check thresholds*")
    elif d.rejection_rate() > 90:
        lines.append(
            f"\u26a0\ufe0f *High rejection rate {d.rejection_rate()}%* — "
            f"possible over-filtering via {d.dominant_reject_reason()}"
        )
    elif d.final_count >= 3:
        lines.append(f"\u2705 Quality: {d.final_count} signals passed (≥ 3 target)")

    lines.append(sep)
    return "\n".join(lines)

# Load API key
if os.getenv("RAILWAY_ENVIRONMENT"):
    from config_prod import TD_API_KEY
else:
    try:
        from config_local import TD_API_KEY
    except ImportError:
        TD_API_KEY = os.getenv("TD_API_KEY")

# ── DataFrame cache ───────────────────────────────────────
_df_memory_cache: pd.DataFrame | None = None
_df_memory_cache_time: datetime | None = None
_DF_CACHE_MAX_AGE_SECONDS = 30 * 60


def _save_df_cache(df: pd.DataFrame) -> None:
    global _df_memory_cache, _df_memory_cache_time
    _df_memory_cache = df
    _df_memory_cache_time = datetime.utcnow()
    try:
        df.to_pickle(DF_CACHE_FILE)
        with open(DF_CACHE_FILE + ".ts", "w") as f:
            f.write(_df_memory_cache_time.isoformat())
    except Exception as e:
        logger.warning(f"DF cache save failed: {e}")


def _get_df_cache_age_seconds() -> float | None:
    if _df_memory_cache_time is None:
        return None
    return (datetime.utcnow() - _df_memory_cache_time).total_seconds()


def is_df_cache_fresh() -> bool:
    age = _get_df_cache_age_seconds()
    if age is None:
        return False
    return age <= _DF_CACHE_MAX_AGE_SECONDS


def _load_df_cache() -> pd.DataFrame | None:
    global _df_memory_cache, _df_memory_cache_time
    if _df_memory_cache is not None:
        if is_df_cache_fresh():
            return _df_memory_cache
        logger.warning("Cached market data too old (>30 min) — rejecting in-memory cache.")

    if os.path.exists(DF_CACHE_FILE):
        try:
            _ts_file = DF_CACHE_FILE + ".ts"
            if os.path.exists(_ts_file):
                with open(_ts_file) as f:
                    saved_at = datetime.fromisoformat(f.read().strip())
                age_sec = (datetime.utcnow() - saved_at).total_seconds()
                if age_sec > _DF_CACHE_MAX_AGE_SECONDS:
                    logger.warning(f"Disk cache too old ({int(age_sec//60)}m) — rejecting.")
                    return None
            else:
                saved_at = datetime.utcfromtimestamp(os.path.getmtime(DF_CACHE_FILE))
                age_sec = (datetime.utcnow() - saved_at).total_seconds()
                if age_sec > _DF_CACHE_MAX_AGE_SECONDS:
                    logger.warning(f"Disk cache too old ({int(age_sec//60)}m) — rejecting.")
                    return None
            df = pd.read_pickle(DF_CACHE_FILE)
            _df_memory_cache = df
            _df_memory_cache_time = saved_at
            logger.info("Loaded DataFrame from disk cache.")
            return df
        except Exception as e:
            logger.warning(f"DF cache load failed: {e}")
    return None


# ── Data fetching ─────────────────────────────────────────
def get_historical_data(outputsize=4500) -> pd.DataFrame | None:
    if not TD_API_KEY:
        logger.error("TD_API_KEY not found.")
        return _load_df_cache()

    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "apikey": TD_API_KEY, "outputsize": outputsize}

    try:
        res = requests.get(url, params=params, timeout=15).json()
        if "values" not in res:
            logger.error(f"API Error: {res}")
            return _load_df_cache()

        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].astype(float)
        df = df.sort_values("datetime").reset_index(drop=True)
        df = add_indicators(df)
        df = _enrich_df(df)
        _save_df_cache(df)
        return df

    except Exception as e:
        logger.error(f"Fetch Error: {e}")
        cached = _load_df_cache()
        if cached is not None:
            logger.info("Using cached DataFrame.")
        return cached


def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns needed by the pattern engine."""
    dt = pd.to_datetime(df["datetime"])
    if getattr(dt.dt, "tz", None) is None:
        dt_utc = dt.dt.tz_localize("UTC")
    else:
        dt_utc = dt.dt.tz_convert("UTC")
    df["datetime_ist"] = dt_utc.dt.tz_convert("Asia/Kolkata")
    df["TimeOfDay"] = df["datetime_ist"].dt.strftime("%H:%M")
    df["TradeDateIST"] = df["datetime_ist"].dt.strftime("%Y-%m-%d")

    # Candle result
    df["Result_CALL"] = (df["Close"] > df["Open"]).astype(int)
    df["Result_PUT"]  = (df["Close"] < df["Open"]).astype(int)

    # EMA trend alignment
    df["EMA_Trend_CALL"] = (df["EMA50"] > df["EMA200"]).astype(int)
    df["EMA_Trend_PUT"]  = (df["EMA50"] < df["EMA200"]).astype(int)

    # RSI continuation
    df["RSI_Cont_CALL"] = (df["RSI"] > 50).astype(int)
    df["RSI_Cont_PUT"]  = (df["RSI"] < 50).astype(int)

    # Candle body / strength
    df["Body"]     = (df["Close"] - df["Open"]).abs()
    df["Range"]    = (df["High"] - df["Low"]).replace(0, 0.00001)
    df["Strength"] = df["Body"] / df["Range"]

    # Momentum continuation (close vs previous close)
    df["Mom_Cont_CALL"] = (df["Close"] > df.shift(1)["Close"]).astype(int)
    df["Mom_Cont_PUT"]  = (df["Close"] < df.shift(1)["Close"]).astype(int)

    # Reversal flag: candle closes opposite to previous
    df["Reversal"] = ((df["Result_CALL"] != df.shift(1)["Result_CALL"]) & df.shift(1)["Result_CALL"].notna()).astype(int)

    atr = pd.to_numeric(df["ATR"], errors="coerce").astype(float).mask(lambda s: s <= 0)
    atr_median = atr.rolling(96, min_periods=20).median().bfill().ffill()
    df["ATR_Ratio"] = (atr / atr_median).replace([float("inf"), -float("inf")], pd.NA).fillna(1.0)
    df["Healthy_Volatility"] = ((df["ATR_Ratio"] >= 0.65) & (df["ATR_Ratio"] <= 1.80)).astype(int)
    df["Noisy_Sideways"] = ((df["Strength"] < 0.35) & (df["ATR_Ratio"] < 0.80)).astype(int)

    return df


# ── Live direction decision ───────────────────────────────
def decide_direction_live(df: pd.DataFrame) -> tuple[str, int]:
    if df is None or len(df) < 50:
        logger.warning("Insufficient data for live direction decision.")
        return None, 0

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    call_score = put_score = 0

    # EMA Trend (25)
    try:
        if not pd.isna(last["EMA50"]) and not pd.isna(last["EMA200"]):
            if last["EMA50"] > last["EMA200"]:
                call_score += 25
            else:
                put_score += 25
    except Exception:
        pass

    # RSI Momentum (25)
    try:
        rsi = float(last["RSI"]) if not pd.isna(last["RSI"]) else 50.0
        rsi_prev = float(prev["RSI"]) if not pd.isna(prev["RSI"]) else 50.0
        if rsi > 50 and rsi > rsi_prev:
            call_score += 25
        elif rsi < 50 and rsi < rsi_prev:
            put_score += 25
        elif rsi > 50:
            call_score += 12
        else:
            put_score += 12
    except Exception:
        pass

    # ATR Strength (15)
    try:
        atr = float(last["ATR"]) if not pd.isna(last["ATR"]) else 0
        atr_mean = float(df["ATR"].mean()) if not pd.isna(df["ATR"].mean()) else 0
        if atr > atr_mean:
            if call_score >= put_score:
                call_score += 15
            else:
                put_score += 15
    except Exception:
        pass

    # Candle Momentum (15)
    try:
        close = float(last["Close"])
        open_ = float(last["Open"])
        prev_close = float(prev["Close"])
        if close > open_ and close > prev_close:
            call_score += 15
        elif close < open_ and close < prev_close:
            put_score += 15
        elif close > open_:
            call_score += 7
        else:
            put_score += 7
    except Exception:
        pass

    # Price vs EMA50 (10)
    try:
        ema50 = float(last["EMA50"]) if not pd.isna(last["EMA50"]) else float(last["Close"])
        if float(last["Close"]) > ema50:
            call_score += 10
        else:
            put_score += 10
    except Exception:
        pass

    if call_score > put_score:
        direction = "CALL"
        confidence = int((call_score / 100) * 100)
    elif put_score > call_score:
        direction = "PUT"
        confidence = int((put_score / 100) * 100)
    else:
        direction = None
        confidence = 0

    confidence = max(0, min(confidence, 99))
    if direction:
        logger.info(f"Live direction: {direction} | CALL={call_score} PUT={put_score} | Confidence={confidence}%")
    else:
        logger.warning(f"No clear direction: CALL={call_score} PUT={put_score}")

    return direction, confidence


# ── Advanced recurring pattern analysis ──────────────────
def _analyse_slot(slot_data: pd.DataFrame, atr_mean: float, direction: str) -> dict:
    """
    Compute all 8 metrics for a single time-slot + direction.

    Returns a dict with named metrics and a composite score (0-100).
    """
    n = len(slot_data)
    if direction == "CALL":
        wr          = slot_data["Result_CALL"].mean() * 100        # historical success %
        ema_align   = slot_data["EMA_Trend_CALL"].mean() * 100
        rsi_cont    = slot_data["RSI_Cont_CALL"].mean() * 100
        mom_cont    = slot_data["Mom_Cont_CALL"].mean() * 100
    else:
        wr          = slot_data["Result_PUT"].mean() * 100
        ema_align   = slot_data["EMA_Trend_PUT"].mean() * 100
        rsi_cont    = slot_data["RSI_Cont_PUT"].mean() * 100
        mom_cont    = slot_data["Mom_Cont_PUT"].mean() * 100

    # ATR stability: how close is slot ATR to the mean (low spread = stable)
    atr_avg       = slot_data["ATR"].mean()
    atr_stability = 100 if atr_avg > atr_mean * 0.6 else 60  # active but not extreme

    # Bullish / Bearish consistency
    bullish_pct   = slot_data["Result_CALL"].mean() * 100
    bearish_pct   = slot_data["Result_PUT"].mean() * 100

    # Reversal frequency (low = better)
    reversal_freq = slot_data["Reversal"].mean() * 100 if "Reversal" in slot_data.columns else 50.0
    reversal_score = max(0, 100 - reversal_freq)          # invert: low reversals → high score

    # Average candle continuation strength
    candle_strength = slot_data["Strength"].mean() * 100

    # Session strength: weight London (13-18 IST) higher
    # (will be applied in caller after UTC→IST conversion)
    session_strength = 100.0  # placeholder; refined by caller

    # ── Composite score with rebalanced weights (v2 — balanced safe-profit) ──
    # Priorities:
    #   win_rate: 0.35         ↑ historical track record is the primary signal
    #   ema_align: 0.15        ↑ trend alignment (direction consistency proxy)
    #   mom_cont: 0.15         ↑ momentum continuation priority
    #   reversal_score: 0.10   → less aggressive reversal penalty vs v1
    #   rsi_cont: 0.10         unchanged — RSI momentum quality
    #   atr_stability: 0.08    ↓ reduced ATR strictness
    #   candle_strength: 0.07  ↓ less weight on body quality alone
    composite = (
        wr              * 0.35 +   # historical direction success (primary)
        ema_align       * 0.15 +   # EMA trend alignment (direction bias)
        mom_cont        * 0.15 +   # candle momentum continuation (priority)
        rsi_cont        * 0.10 +   # RSI momentum continuation
        reversal_score  * 0.10 +   # low reversal probability (less aggressive)
        atr_stability   * 0.08 +   # ATR stability / activity
        candle_strength * 0.07     # candle body strength
    )

    # ── Recurring timing stability bonus ───────────────────────
    # Reward timings with very strong win rates and stable direction
    # (these are the historically repeating profitable timings we want to prioritize)
    if wr >= 70.0 and reversal_freq <= 30.0:
        composite += 5.0   # strong recurring profitable timing bonus
    elif wr >= 65.0 and reversal_freq <= 35.0:
        composite += 3.0   # moderate recurring timing bonus

    # ── Penalize weak/noisy timings (v2 — softened) ────────────
    # Threshold raised from 60.0 to 58.0 to avoid penalizing borderline valid slots
    if wr < 58.0:
        composite -= 10.0  # was 15.0 — reduced penalty

    return {
        "direction": direction,
        "n": n,
        "historical_success_rate": round(wr, 1),
        "bullish_pct": round(bullish_pct, 1),
        "bearish_pct": round(bearish_pct, 1),
        "ema_alignment": round(ema_align, 1),
        "rsi_continuation": round(rsi_cont, 1),
        "atr_avg": round(atr_avg, 6),
        "atr_stability": atr_stability,
        "momentum_consistency": round(mom_cont, 1),
        "reversal_frequency": round(reversal_freq, 1),
        "candle_strength": round(candle_strength, 1),
        "composite": round(composite, 2),
    }


def _calculate_recurring_strength_legacy(df: pd.DataFrame) -> list[dict]:
    """
    Main pattern analysis.  Scans every 5-min time slot over the last
    14 days and selects the strongest recurring directional patterns.

    Returns a list of signal dicts (sorted by time) ready for
    generated_signals.json, each including:
        time, pair, direction, confidence, pattern_strength,
        historical_success_rate, source="generated"
    """
    if df is None or len(df) < 500:
        logger.warning("Insufficient data for recurring pattern analysis.")
        return []

    # Restrict to last 14 days
    cutoff = df["datetime"].max() - timedelta(days=14)
    df = df[df["datetime"] >= cutoff].copy()

    atr_mean = df["ATR"].mean()
    unique_times = sorted(df["TimeOfDay"].unique())

    call_candidates: list[dict] = []
    put_candidates:  list[dict] = []

    for t_utc in unique_times:
        slot_data = df[df["TimeOfDay"] == t_utc]
        if len(slot_data) < MIN_SLOT_OCCURRENCES:
            continue

        # Convert UTC time → IST (+5:30)
        try:
            h, m = map(int, t_utc.split(":"))
        except ValueError:
            continue
        ist_minutes = (h * 60 + m + 330) % 1440

        # Only generate for 13:00 – 22:00 IST session
        if not (13 * 60 <= ist_minutes <= 22 * 60):
            continue

        ist_time_str = f"{ist_minutes // 60:02d}:{ist_minutes % 60:02d}"

        # Session strength weight (London/NY prime = 13:30–21:30 IST)
        london_start = 13 * 60 + 30
        ny_close     = 21 * 60 + 30
        session_str  = 100.0 if london_start <= ist_minutes <= ny_close else 75.0

        for direction in ("CALL", "PUT"):
            metrics = _analyse_slot(slot_data, atr_mean, direction)
            metrics["composite"] += (session_str - 100.0) * 0.05  # adjust for session
            metrics["session_strength"] = session_str

            base_conf = metrics["composite"]

            # Skip obviously weak slots early
            if base_conf < BASE_CONFIDENCE_THRESHOLD - 10:
                continue

            # Get legacy learning engine adjustment
            rsi_avg = float(slot_data["RSI"].mean()) if "RSI" in slot_data.columns else 50.0
            adj_legacy = learning_engine.get_adaptive_adjustment(
                ist_time_str, direction, int(base_conf),
                metrics["atr_avg"], rsi_avg, source="generated"
            )
            if adj_legacy <= -3:
                continue   # learning engine veto
            base_conf += adj_legacy

            # Get timing_db adaptive adjustment
            adj_timing = timing_db.get_adaptive_adjustment(ist_time_str, direction)
            base_conf += adj_timing

            # Pattern strength from timing_db
            pattern_strength = timing_db.get_pattern_strength(ist_time_str, direction)

            # Reject if pattern strength too low AND confidence borderline
            if pattern_strength < PATTERN_STRENGTH_THRESHOLD and base_conf < BASE_CONFIDENCE_THRESHOLD:
                continue

            if base_conf < BASE_CONFIDENCE_THRESHOLD:
                continue

            signal = {
                "time": ist_time_str,
                "pair": PAIR,
                "direction": direction,
                "confidence": int(min(99, base_conf)),
                "pattern_strength": pattern_strength,
                "historical_success_rate": metrics["historical_success_rate"],
                "source": "generated",
                # Extra context (for logging / stats)
                "_metrics": metrics,
            }

            if direction == "CALL":
                call_candidates.append(signal)
            else:
                put_candidates.append(signal)

    call_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    put_candidates.sort(key=lambda x: x["confidence"], reverse=True)

    # ── Balanced selection ────────────────────────────────
    final = _select_balanced(call_candidates, put_candidates)

    # Strip internal metrics key
    for s in final:
        s.pop("_metrics", None)

    final.sort(key=lambda x: x["time"])

    # Log summary
    for s in final:
        logger.info(
            f"[Pattern] {s['time']} {s['direction']} | "
            f"Confidence={s['confidence']}% | "
            f"Pattern Strength={s['pattern_strength']} | "
            f"Historical Success={s['historical_success_rate']}%"
        )




    return final



def _select_balanced(
    call_candidates: list[dict],
    put_candidates:  list[dict],
    total_target: int = 12,
    max_dominance: float = 0.70,
    balance_min_conf: int = 63,
) -> list[dict]:
    """Select up to total_target signals keeping direction balance ≤ max_dominance."""
    combined = call_candidates + put_candidates
    combined.sort(key=lambda x: x["confidence"], reverse=True)

    final: list[dict] = []
    c_count = p_count = 0
    avail_c = len(call_candidates)
    avail_p = len(put_candidates)

    # Pass 1: standard selection
    for s in combined:
        if len(final) >= total_target:
            break
        if s["direction"] == "CALL":
            max_p = min(avail_p, total_target - (c_count + 1))
            max_allowed = max(2, (c_count + 1 + max_p) * max_dominance)
            if (c_count + 1) > max_allowed:
                continue
            final.append(s)
            c_count += 1
        else:
            max_c = min(avail_c, total_target - (p_count + 1))
            max_allowed = max(2, (p_count + 1 + max_c) * max_dominance)
            if (p_count + 1) > max_allowed:
                continue
            final.append(s)
            p_count += 1

    # Pass 2: rebalance if one direction dominates
    total_selected = len(final)
    if total_selected >= 2:
        dominant  = "CALL" if c_count > p_count else "PUT"
        minority  = "PUT"  if dominant == "CALL" else "CALL"
        dom_count = c_count if dominant == "CALL" else p_count

        if dom_count / total_selected > max_dominance:
            selected_times = {s["time"] for s in final}
            minority_pool = [
                s for s in (put_candidates if minority == "PUT" else call_candidates)
                if s["confidence"] >= balance_min_conf and s["time"] not in selected_times
            ]
            minority_pool.sort(key=lambda x: x["confidence"], reverse=True)

            max_dom_allowed  = int(total_selected * max_dominance)
            swaps_needed     = dom_count - max_dom_allowed

            for extra in minority_pool:
                if swaps_needed <= 0:
                    break
                dom_sigs = [s for s in final if s["direction"] == dominant]
                if not dom_sigs:
                    break
                weakest = min(dom_sigs, key=lambda x: x["confidence"])
                if weakest["confidence"] > extra["confidence"]:
                    break
                final.remove(weakest)
                final.append(extra)
                if dominant == "CALL":
                    c_count -= 1; p_count += 1
                else:
                    p_count -= 1; c_count += 1
                swaps_needed -= 1
                logger.info(
                    f"[Balance] Swapped {dominant} conf={weakest['confidence']}% "
                    f"→ {minority} conf={extra['confidence']}% @ {extra['time']}"
                )




    return final



# ── State helpers ─────────────────────────────────────────
def _score_in_range(value: float, low: float, high: float, ideal_low: float, ideal_high: float) -> float:
    """Return a 0-100 quality score for values best kept in a healthy middle band."""
    if pd.isna(value):
        return 50.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        if value <= low:
            return 0.0
        return ((value - low) / max(ideal_low - low, 0.000001)) * 100
    if value >= high:
        return 0.0
    return ((high - value) / max(high - ideal_high, 0.000001)) * 100


def _session_strength(minutes_ist: int, direction: str = "CALL", df=None) -> float:
    """
    Compute adaptive session strength using the SessionIntelligence engine.

    Replaces the old 3-tier static lookup (100/78/35) with a fully adaptive
    score driven by:
      - Session structural liquidity (Asian/London/Overlap/NY)
      - Rolling win rate for this session × direction
      - Historical volatility quality in this session
      - Historical continuation strength in this session

    Returns a value in [20, 100].
    """
    strength, _ = session_intel.compute_session_strength(minutes_ist, direction, df)
    return strength


def _analyse_probability_slot(
    slot_data: pd.DataFrame,
    market_data: pd.DataFrame,
    direction: str,
    session_strength: float,
) -> dict:
    """
    Compute all raw sub-metrics for a time slot.

    Does NOT apply the centralized scoring formula — that is done by
    ProbabilityEngine.compute() in the caller.  This function returns a
    flat dict of normalised metrics (each in [0, 100]) plus derived
    helper fields (atr_avg, volatility_zone, etc.).

    Eight metrics fed into ProbabilityEngine:
        win_rate              (0-100)
        direction_consistency (0-100)
        atr_quality           (0-100)
        momentum_strength     (0-100)
        session_strength      (0-100)
        volatility_quality    (0-100)  ← NEW: replaces old non-formula vol check
        reversal_risk         (0-100)  ← subtracted
        [regime / regime_confidence fed separately from regime_report]
    """
    if direction == "CALL":
        historical_success    = slot_data["Result_CALL"].mean() * 100
        direction_consistency = slot_data["Result_CALL"].mean() * 100
        rsi_cont              = slot_data["RSI_Cont_CALL"].mean() * 100
        mom_cont              = slot_data["Mom_Cont_CALL"].mean() * 100
        trend_reliability     = slot_data["EMA_Trend_CALL"].mean() * 100
    else:
        historical_success    = slot_data["Result_PUT"].mean() * 100
        direction_consistency = slot_data["Result_PUT"].mean() * 100
        rsi_cont              = slot_data["RSI_Cont_PUT"].mean() * 100
        mom_cont              = slot_data["Mom_Cont_PUT"].mean() * 100
        trend_reliability     = slot_data["EMA_Trend_PUT"].mean() * 100

    bullish_pct = slot_data["Result_CALL"].mean() * 100
    bearish_pct = slot_data["Result_PUT"].mean() * 100

    # ── ATR quality (activity × stability) ───────────────
    atr_avg           = float(slot_data["ATR"].mean())
    atr_std           = float(slot_data["ATR"].std() or 0.0)
    atr_market_median = float(market_data["ATR"].median() or atr_avg or 0.0001)
    atr_cv            = atr_std / max(atr_avg, 0.000001)
    atr_activity_score = _score_in_range(
        atr_avg / max(atr_market_median, 0.000001), 0.45, 2.20, 0.75, 1.55
    )
    atr_stability = max(0.0, min(100.0, 100.0 - (atr_cv * 140.0)))
    atr_quality   = (atr_activity_score * 0.60) + (atr_stability * 0.40)

    # ── Volatility clustering & quality ──────────────────
    volatility_cluster_score = float(slot_data["Healthy_Volatility"].mean() * 100)
    noisy_sideways_pct       = float(slot_data["Noisy_Sideways"].mean() * 100)

    # volatility_quality is an 8th metric input to the centralized formula.
    # It combines healthy-volatility presence and low noisy-sideways fraction.
    volatility_quality = max(
        0.0,
        min(100.0, (volatility_cluster_score * 0.70) + ((100.0 - noisy_sideways_pct) * 0.30)),
    )

    # volatility_consistency kept for backward compat (used in cooldown logic)
    volatility_consistency = max(
        0.0,
        min(100.0, (volatility_cluster_score * 0.75) + ((100.0 - noisy_sideways_pct) * 0.25)),
    )

    # Volatility zone (healthy / dead / noisy / unstable_spike)
    slot_atr_ratio  = atr_avg / max(atr_market_median, 1e-8)
    slot_noisy_frac = noisy_sideways_pct / 100.0
    volatility_zone = classify_volatility_zone(slot_atr_ratio, slot_noisy_frac)

    # ── Reversal risk ─────────────────────────────────────
    reversal_probability = (
        float(slot_data["Reversal"].mean() * 100)
        if "Reversal" in slot_data.columns
        else 50.0
    )
    reversal_risk = reversal_probability  # high reversal freq = high risk

    # ── Momentum strength (composite) ────────────────────
    candle_strength   = float(slot_data["Strength"].mean() * 100)
    momentum_strength = (
        mom_cont          * 0.45
        + rsi_cont        * 0.25
        + candle_strength * 0.15
        + trend_reliability * 0.15
    )

    return {
        # ── 8 centralized formula inputs ──────────────────
        "win_rate":                       round(historical_success, 1),
        "direction_consistency":          round(direction_consistency, 1),
        "atr_quality":                    round(atr_quality, 1),
        "momentum_strength":              round(momentum_strength, 1),
        "session_strength":               round(session_strength, 1),
        "volatility_quality":             round(volatility_quality, 1),
        "reversal_risk":                  round(reversal_risk, 1),
        # ── Aliases / backward compat keys ───────────────
        "direction":                      direction,
        "n":                              len(slot_data),
        "bullish_frequency_pct":          round(bullish_pct, 1),
        "bearish_frequency_pct":          round(bearish_pct, 1),
        "historical_win_rate_pct":        round(historical_success, 1),
        "historical_success_rate":        round(historical_success, 1),
        "atr_avg":                        round(atr_avg, 6),
        "atr_stability":                  round(atr_stability, 1),
        "atr_activity_score":             round(atr_activity_score, 1),
        "momentum_continuation_strength": round(momentum_strength, 1),
        "reversal_probability":           round(reversal_probability, 1),
        "volatility_cluster_score":       round(volatility_cluster_score, 1),
        "volatility_consistency":         round(volatility_consistency, 1),
        "volatility_zone":                volatility_zone,
        "trend_continuation_reliability": round(trend_reliability, 1),
        "candle_strength":                round(candle_strength, 1),
    }


def _classify_live_market(df: pd.DataFrame) -> dict:
    """
    Market classification driven by Market Regime Detection.

    Calls detect_market_regime() and translates its output into the
    legacy-compatible dict: {profile, target, threshold, score, regime_report}.

    profile values (for backward compat):
        "strong"   → TRENDING
        "high_vol" → HIGH_VOLATILITY
        "reversal" → REVERSAL_HEAVY
        "moderate" → (fallback moderate)
        "weak"     → SIDEWAYS or insufficient data
    """
    if df is None or len(df) < 80:
        return {
            "profile":       "weak",
            "target":        3,
            "threshold":     78,
            "score":         0,
            "regime":        "SIDEWAYS",
            "regime_report": None,
        }

    regime_report = detect_market_regime(df)
    regime        = regime_report["regime"]
    behavior      = regime_report["behavior"]
    conf          = regime_report["confidence"]

    # Map regime → legacy profile string
    profile_map = {
        "TRENDING":        "strong",
        "HIGH_VOLATILITY": "high_vol",
        "REVERSAL_HEAVY":  "reversal",
        "SIDEWAYS":        "weak",
    }
    profile = profile_map.get(regime, "weak")

    return {
        "profile":       profile,
        "target":        behavior["target"],
        "threshold":     behavior["threshold"],
        "score":         conf,
        "regime":        regime,
        "regime_report": regime_report,
    }


def _live_confirmation_ok(
    df: pd.DataFrame,
    direction: str,
    candidate_confidence: float,
    live_direction: str | None = None,
) -> tuple[bool, str, float]:
    if df is None or len(df) < 80:
        return False, "insufficient live data", candidate_confidence

    if live_direction is None:
        live_direction, _ = decide_direction_live(df)
    if live_direction != direction:
        return False, f"live direction mismatch ({live_direction})", candidate_confidence

    market_ok, market_msg, safety_penalty = run_market_safety(df, direction)
    adjusted_confidence = candidate_confidence - safety_penalty
    if not market_ok:
        return False, f"market safety rejected: {market_msg}", adjusted_confidence

    last = df.iloc[-1]
    prev = df.iloc[-2]
    atr = float(last["ATR"]) if not pd.isna(last.get("ATR")) else 0.0
    atr_mean = float(df["ATR"].tail(80).mean() or 0.0)
    rsi_now = float(last["RSI"])
    rsi_prev = float(prev["RSI"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])

    if atr_mean <= 0 or atr < atr_mean * 0.60:
        return False, "live ATR too weak", adjusted_confidence
    if atr > atr_mean * 3.00:
        return False, "live ATR spike too unstable", adjusted_confidence
    if atr > atr_mean * 2.20:
        adjusted_confidence -= 8
    if direction == "CALL":
        if not (ema50 > ema200 and rsi_now > 50 and rsi_now >= rsi_prev):
            return False, "live EMA/RSI not bullish", adjusted_confidence
    else:
        if not (ema50 < ema200 and rsi_now < 50 and rsi_now <= rsi_prev):
            return False, "live EMA/RSI not bearish", adjusted_confidence

    return adjusted_confidence >= 65, "live confirmation passed", adjusted_confidence


def _minutes_from_time(time_str: str) -> int:
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def _metric_float(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return round(float(metrics.get(key, default)), 3)
    except Exception:
        return default


def _select_ranked_with_cooldown(candidates: list[dict], total_target: int) -> list[dict]:
    """
    Deduplicate by time slot (keep highest probability_score), rank by
    probability_score descending, then apply minimum-gap cooldown.
    STRONG_SIGNAL candidates have priority over MODERATE_SIGNAL candidates.
    """
    best_by_time: dict[str, dict] = {}
    for candidate in candidates:
        existing = best_by_time.get(candidate["time"])
        if existing is None or (
            candidate.get("probability_score", 0) > existing.get("probability_score", 0)
        ):
            best_by_time[candidate["time"]] = candidate

    # Sort: STRONG first, then MODERATE, within each tier by probability_score desc
    def _rank_key(x: dict) -> tuple:
        tier_order = {TIER_STRONG: 2, TIER_MODERATE: 1, TIER_SKIP: 0}
        return (
            tier_order.get(x.get("signal_tier", TIER_SKIP), 0),
            x.get("probability_score", 0.0),
            x.get("pattern_strength", 0),
        )

    ranked = sorted(best_by_time.values(), key=_rank_key, reverse=True)

    final: list[dict] = []
    for candidate in ranked:
        if len(final) >= total_target:
            break
        candidate_minute = _minutes_from_time(candidate["time"])
        # Unstable signals need a wider cooldown gap
        unstable = (
            candidate.get("volatility_consistency", 0) < 70
            or candidate.get("reversal_probability", 100) > 42
        )
        required_gap = UNSTABLE_SIGNAL_GAP_MINUTES if unstable else MIN_RECURRING_SIGNAL_GAP_MINUTES
        if any(
            abs(candidate_minute - _minutes_from_time(selected["time"])) < required_gap
            for selected in final
        ):
            continue
        final.append(candidate)

    if len(final) < 3:
        logger.warning(f"Only {len(final)} generated timings passed quality filters today.")




    return final



def calculate_recurring_strength(df: pd.DataFrame) -> list[dict]:
    """
    Advanced recurring probability engine.

    Pipeline:
      1. Detect market regime (TRENDING / SIDEWAYS / HIGH_VOLATILITY / REVERSAL_HEAVY)
      2. Apply regime-adaptive targets and thresholds
      3. Score every IST time slot with weighted probability formula
      4. Classify volatility zone per slot
      5. Apply learning-engine and timing-DB adjustments
      6. Gate each candidate through live confirmation
      7. Rank and return the top signals for the day
    """
    if df is None or len(df) < 500:
        logger.warning("Insufficient data for recurring pattern analysis.")
        return []

    if "datetime_ist" not in df.columns or "Healthy_Volatility" not in df.columns:
        df = _enrich_df(df.copy())

    cutoff = df["datetime_ist"].max() - pd.Timedelta(days=14)
    df14   = df[df["datetime_ist"] >= cutoff].copy()
    unique_times = sorted(df14["TimeOfDay"].unique())

    # ── Step 1: Market Regime Detection ──────────────────
    market_profile  = _classify_live_market(df)
    regime          = market_profile["regime"]
    regime_report   = market_profile["regime_report"]
    behavior        = get_regime_behavior(regime)

    # Regime-adaptive parameters
    total_target  = behavior["target"]
    target_min    = behavior.get("target_min", max(2, total_target - 3))
    target_max    = behavior.get("target_max", total_target + 3)
    threshold     = max(BASE_CONFIDENCE_THRESHOLD, behavior["threshold"])
    min_pat_str   = behavior["min_pattern_str"]
    rev_prob_max  = behavior["reversal_prob_max"]
    atr_ratio_min = behavior["atr_ratio_min"]
    atr_ratio_max = behavior["atr_ratio_max"]

    from probability_engine import TIER_MODERATE_MIN
    fallback_threshold = TIER_MODERATE_MIN

    logger.info(
        f"[Regime] Using regime={regime} | target={total_target} (range {target_min}–{target_max}) | "
        f"threshold={threshold} (fallback={fallback_threshold}) | min_pat_str={min_pat_str} | "
        f"rev_prob_max={rev_prob_max}"
    )

    today_ist      = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")
    live_direction, _ = decide_direction_live(df)

    # ── Step 1.5: Market Structure ───────────────────────
    from market_structure import market_structure_engine
    market_structure = market_structure_engine.analyse(df)

    # Log sweep detections for the main pipeline
    if market_structure.has_strong_sweep:
        logger.info(
            "🌊 STRONG SWEEP DETECTED [signal_gen]: dir=%s | conf=%.1f | %s",
            market_structure.sweep_direction,
            market_structure.sweep_confidence,
            market_structure.sweep_result.sweep_summary if market_structure.sweep_result else "",
        )
    elif market_structure.sweep_result and market_structure.sweep_result.detected:
        logger.info(
            "〰️ Moderate sweep [signal_gen]: dir=%s | conf=%.1f",
            market_structure.sweep_direction,
            market_structure.sweep_confidence,
        )
    # ── Step 1.6: Currency Strength ─────────────────────────────────
    # Computed once per run (cached 30 min). Uses Tier 2 basket if API key
    # is available, otherwise Tier 1 (df only). Confirmation-only — never
    # generates a signal alone; only modifies regime_mult in ProbabilityEngine.
    from currency_strength import currency_strength_engine
    currency_strength_engine.invalidate_cache()  # fresh result each daily run
    cs_result = currency_strength_engine.compute(df, api_key=TD_API_KEY)
    logger.info(
        "💱 Currency Strength: EUR=%.1f USD=%.1f bias=%s conf=%.1f tier=%s method=%s",
        cs_result.eur_strength, cs_result.usd_strength,
        cs_result.bias, cs_result.bias_confidence,
        cs_result.tier, cs_result.method,
    )

    # ATR median for volatility zone classification
    atr_market_median = float(
        pd.to_numeric(df14["ATR"], errors="coerce").tail(96).median() or 0.0001
    )

    regime_confidence = market_profile["score"]  # 0-100 regime detection confidence
    current_regime_floor = None
    
    for attempt in range(2):
        candidates: list[dict] = []
        generated_candidates_count = 0
        accepted_signals_list = []
        rejected_signals_list = []
        for ist_time_str in unique_times:
            slot_data = df14[df14["TimeOfDay"] == ist_time_str]
            if len(slot_data) < MIN_SLOT_OCCURRENCES:
                continue

            try:
                h, m = map(int, ist_time_str.split(":"))
            except ValueError:
                continue
            ist_minutes = h * 60 + m

            if not (13 * 60 <= ist_minutes <= 22 * 60):
                continue

            session_str = _session_strength(ist_minutes)   # kept for legacy callers


            # ── Session Intelligence (adaptive, direction-aware) ─────────────────
            # compute_session_strength() returns an adaptive score based on:
            #   a) session structural liquidity (Asian/London/Overlap/NY)
            #   b) rolling win rate for this session × direction
            #   c) historical volatility quality in this session
            #   d) historical continuation strength
            # The second return value (detail dict) is stored on the signal
            # for downstream display and recording.
            for direction in ("CALL", "PUT"):
                generated_candidates_count += 1
                session_str, session_detail = session_intel.compute_session_strength(
                    ist_minutes, direction, df
                )
                metrics  = _analyse_probability_slot(slot_data, df14, direction, session_str)
                metrics["_session_detail"] = session_detail   # stash for signal dict
                vol_zone = metrics["volatility_zone"]

                # ── Pre-score hard gates (fast rejection before scoring) ──
                # 1. Extreme volatility zones: always skip regardless of score
                if vol_zone in ("dead", "noisy", "unstable_spike"):
                    logger.debug(
                        "[Slot] Skip %s %s — vol_zone=%s", ist_time_str, direction, vol_zone
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"vol_zone={vol_zone}"))
                    continue

                # 2. ATR ratio vs regime bounds
                slot_atr_ratio = metrics["atr_avg"] / max(atr_market_median, 1e-8)
                if slot_atr_ratio < atr_ratio_min or slot_atr_ratio > atr_ratio_max:
                    logger.debug(
                        "[Slot] Skip %s %s — atr_ratio=%.2f out of bounds [%.2f,%.2f]",
                        ist_time_str, direction, slot_atr_ratio, atr_ratio_min, atr_ratio_max,
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"atr_ratio={slot_atr_ratio:.2f} out of bounds [{atr_ratio_min:.2f},{atr_ratio_max:.2f}]"))
                    continue

                # ── Sequence Pattern Analysis ──────────────────────────────
                seq_result = sequence_engine.analyse(df, direction_hint=direction, market_structure=market_structure)
                metrics["_sequence_result"] = seq_result

                # Sequence momentum bonus: +-0 to +-20 based on pattern match/conflict
                seq_bonus = get_sequence_momentum_bonus(seq_result, direction)
                # Apply bonus to the momentum_strength metric (clamped to 0-100)
                # before passing to probability engine so the formula benefits.
                metrics["momentum_strength"] = round(
                    max(0.0, min(100.0, metrics["momentum_strength"] + seq_bonus * 0.5)), 1
                )

                # ── Centralized probability scoring (with dynamic voter weights) ────
                prob_inputs = ProbabilityInputs(
                    win_rate              = metrics["win_rate"],
                    direction_consistency = metrics["direction_consistency"],
                    atr_quality           = metrics["atr_quality"],
                    momentum_strength     = metrics["momentum_strength"],
                    session_strength      = metrics["session_strength"],
                    volatility_quality    = metrics["volatility_quality"],
                    reversal_risk         = metrics["reversal_risk"],
                    regime                = regime,
                    regime_confidence     = float(regime_confidence),
                    volatility_zone       = vol_zone,
                    time_str              = ist_time_str,
                    direction             = direction,
                    market_structure      = market_structure,
                    currency_bias         = cs_result.bias,
                    currency_strength_score = cs_result.bias_confidence,
                )
                # Use dynamic voter weights (auto-fetches from strategy_weight_tracker)
                prob_result = probability_engine.compute_with_voter_weights(prob_inputs)

                # ── Tier gate: use regime-aware acceptability (dynamic min score) ─────
                # is_acceptable_for_regime() applies the regime-specific floor:
                #   TRENDING=70, SIDEWAYS=78, HIGH_VOL/REVERSAL=80
                # This allows quality signals through in trending markets without
                # forcing weak trades in sideways/volatile conditions.
                if not probability_engine.is_acceptable_for_regime(prob_result, regime, current_regime_floor):
                    logger.debug(
                        "[Score] SKIP %s %s — prob_score=%.1f tier=%s regime_floor=%.0f",
                        ist_time_str, direction,
                        prob_result.probability_score, prob_result.signal_tier,
                        (current_regime_floor if current_regime_floor is not None else _regime_min_score(regime)),
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"prob_score={prob_result.probability_score:.1f} < regime_floor={(current_regime_floor if current_regime_floor is not None else _regime_min_score(regime)):.0f}"))
                    continue

                # ── REVERSAL_HEAVY extra gate ─────────────────────────────
                # In reversal-heavy markets only STRONG signals are allowed
                if regime == "REVERSAL_HEAVY" and prob_result.signal_tier != TIER_STRONG:
                    logger.info(
                        "[Regime] %s %s paused — reversal-heavy requires STRONG "
                        "(score=%.1f tier=%s)",
                        ist_time_str, direction,
                        prob_result.probability_score, prob_result.signal_tier,
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"reversal-heavy requires STRONG (score={prob_result.probability_score:.1f})"))
                    continue

                # ── Multi-Strategy Agreement Gate ─────────────────────────────────
                # Evaluate all 8 strategy voters. Gate: 6/8=STRONG, 5/8=MODERATE, <5=SKIP.
                # This is an ADDITIONAL filter on top of the probability gate — it never
                # replaces market safety, stale cleanup, martingale, or timezone logic.
                agreement = agreement_engine.compute(
                    df, direction, metrics, regime, prob_result, live_direction, market_structure=market_structure
                )
                if agreement.tier == AGREEMENT_SKIP:
                    logger.info(
                        "[Agreement] SKIP %s %s — score=%d/%d (%s)",
                        ist_time_str, direction,
                        agreement.agreement_score, agreement.total_voters, agreement.tier,
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"agreement={agreement.agreement_score}/{agreement.total_voters} ({agreement.tier})"))
                    continue

                logger.info(
                    "[Agreement] %s %s — score=%d/%d tier=%s",
                    ist_time_str, direction,
                    agreement.agreement_score, agreement.total_voters, agreement.tier,
                )

                # ── Adaptive learning adjustments ────────────────────────
                rsi_avg    = float(slot_data["RSI"].mean()) if "RSI" in slot_data.columns else 50.0
                base_conf  = prob_result.probability_score  # use prob score as base confidence

                adj_legacy = learning_engine.get_adaptive_adjustment(
                    ist_time_str, direction, int(base_conf),
                    metrics["atr_avg"], rsi_avg,
                    source="generated",
                    regime=regime,
                )
                if adj_legacy <= -3:
                    logger.debug(
                        "[Learn] Veto %s %s — learning adj=%d",
                        ist_time_str, direction, adj_legacy,
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"learning veto (adj={adj_legacy})"))
                    continue

                # Blend learning and timing-db adjustments into confidence
                adj_timing       = timing_db.get_adaptive_adjustment(ist_time_str, direction, regime=regime)
                learned_strength = timing_db.get_regime_pattern_strength(ist_time_str, direction, regime)

                # Final blended confidence = prob_score (primary) + learning tweaks
                blended_conf = probability_engine.score_to_confidence(
                    prob_result.probability_score,
                    base_conf + adj_legacy + adj_timing,
                )

                if learned_strength < PATTERN_STRENGTH_THRESHOLD and blended_conf < fallback_threshold:
                    logger.debug(
                        "[DB] Skip %s %s — learned_str=%d blended_conf=%.1f < fallback_threshold=%.1f",
                        ist_time_str, direction, learned_strength, blended_conf, fallback_threshold,
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"DB strength={learned_strength} & blended_conf={blended_conf:.1f} < {fallback_threshold:.1f}"))
                    continue

                # ── Live confirmation gate ────────────────────────────────
                live_ok, live_reason, adjusted_conf = _live_confirmation_ok(
                    df, direction, blended_conf, live_direction
                )
                if not live_ok:
                    logger.info(
                        "[Pattern] Reject %s %s: %s", ist_time_str, direction, live_reason
                    )
                    rejected_signals_list.append((ist_time_str, direction, f"live confirmation failed: {live_reason}"))
                    continue

                if adjusted_conf < fallback_threshold:
                    rejected_signals_list.append((ist_time_str, direction, f"adjusted_conf={adjusted_conf:.1f} < {fallback_threshold:.1f}"))
                    continue

                # ── Final pattern strength (blended) ─────────────────────
                # 70% probability score + 30% timing-DB learned strength
                pattern_strength = int(round(
                    (prob_result.probability_score * 0.70) + (learned_strength * 0.30)
                ))
                pattern_strength = max(0, min(100, pattern_strength))

                logger.info(
                    "[Slot] %s %s | tier=%s | prob=%.1f | conf=%d | "
                    "pat_str=%d | win=%.1f%% | rev=%.1f | zone=%s",
                    ist_time_str, direction,
                    prob_result.signal_tier, prob_result.probability_score,
                    int(adjusted_conf), pattern_strength,
                    metrics["win_rate"], metrics["reversal_risk"], vol_zone,
                )

                accepted_signals_list.append((ist_time_str, direction))

                candidates.append({
                    "time":                           ist_time_str,
                    "pair":                           PAIR,
                    "direction":                      direction,
                    # ── Centralized score (primary authority) ────────
                    "probability_score":              round(prob_result.probability_score, 2),
                    "signal_tier":                    prob_result.signal_tier,
                    # ── Agreement score ───────────────────────────────
                    "agreement_score":                agreement.agreement_score,
                    "agreement_total":                agreement.total_voters,
                    "agreement_tier":                 agreement.tier,
                    "agreement_votes":                agreement.as_dict().get("votes", {}),
                    # ── Sequence pattern ──────────────────────────
                    "sequence_direction":             seq_result.sequence_direction,
                    "sequence_confidence":            round(seq_result.sequence_confidence, 2),
                    "sequence_continuation_prob":     round(seq_result.continuation_prob, 2),
                    "sequence_patterns":              seq_result.patterns_detected,
                    "sequence_bonus_applied":         round(seq_bonus, 2),
                    # ── Session Intelligence ──────────────────────────────────────
                    "session_name":                   session_detail.get("session", detect_session(ist_minutes)),
                    "session_strength":               _metric_float(metrics, "session_strength"),
                    "session_is_overlap":             session_detail.get("is_overlap", False),
                    "session_adaptive_win_rate":      session_detail.get("adaptive_win_rate", 50.0),
                    "session_vol_quality":            session_detail.get("vol_bonus", 70.0),
                    "session_continuation":           session_detail.get("continuation_bonus", 60.0),
                    "session_sample_size":            session_detail.get("sample_size_all", 0),
                    # ── Published confidence (blended) ───────────────
                    "confidence":                     int(min(99, adjusted_conf)),
                    # ── Pattern strength ─────────────────────────────
                    "pattern_strength":               pattern_strength,
                    "pattern_strength_score":         pattern_strength,
                    # ── Sub-scores (8 formula inputs) ────────────────
                    "historical_success_rate":        _metric_float(metrics, "win_rate"),
                    "historical_win_rate_pct":        _metric_float(metrics, "win_rate"),
                    "bullish_frequency_pct":          _metric_float(metrics, "bullish_frequency_pct"),
                    "bearish_frequency_pct":          _metric_float(metrics, "bearish_frequency_pct"),
                    "direction_consistency":          _metric_float(metrics, "direction_consistency"),
                    "atr_quality":                    _metric_float(metrics, "atr_quality"),
                    "atr_stability":                  _metric_float(metrics, "atr_stability"),
                    "momentum_strength":              _metric_float(metrics, "momentum_strength"),
                    "momentum_continuation_strength": _metric_float(metrics, "momentum_strength"),
                    "session_strength":               _metric_float(metrics, "session_strength"),
                    "volatility_quality":             _metric_float(metrics, "volatility_quality"),
                    "volatility_cluster_score":       _metric_float(metrics, "volatility_cluster_score"),
                    "volatility_consistency":         _metric_float(metrics, "volatility_consistency"),
                    "reversal_probability":           _metric_float(metrics, "reversal_probability"),
                    "reversal_risk":                  _metric_float(metrics, "reversal_risk"),
                    "trend_continuation_reliability": _metric_float(metrics, "trend_continuation_reliability"),
                    # ── Score breakdown (for analysis / logging) ─────
                    "score_breakdown": prob_result.as_dict(),
                    # ── Confidence breakdown (formatted) ─────────────
                    "confidence_breakdown": prob_result.get_breakdown_items(),
                    # ── Context ──────────────────────────────────────
                    "volatility_zone":                vol_zone,
                    "market_profile":                 market_profile["profile"],
                    "market_regime":                  regime,
                    "regime_confidence":              regime_confidence,
                    "generated_date":                 today_ist,
                    "timezone":                       "Asia/Kolkata",
                    "source":                         "generated",
                    # ── Liquidity Sweep ───────────────────────────────
                    "liquidity_sweep":                market_structure.sweep_direction or "NONE",
                    "sweep_confidence":               round(market_structure.sweep_confidence, 1),
                    "has_strong_sweep":               market_structure.has_strong_sweep,
                    # ── Currency Strength ─────────────────────────────
                    "currency_bias":                  cs_result.bias,
                    "currency_eur_strength":          round(cs_result.eur_strength, 1),
                    "currency_usd_strength":          round(cs_result.usd_strength, 1),
                    "currency_strength_score":        round(cs_result.bias_confidence, 1),
                    "currency_strength_tier":         cs_result.tier,
                })

        # ── Rank and select (adaptive count within target_min–target_max) ───
        
        acceptance_rate = len(accepted_signals_list) / generated_candidates_count if generated_candidates_count > 0 else 0
        if attempt == 0 and acceptance_rate < 0.10:
            logger.info(f"Acceptance rate {acceptance_rate*100:.1f}% < 10%. Moderately reducing thresholds for retry.")
            if regime == "TRENDING":
                current_regime_floor = 60.0
            elif regime == "SIDEWAYS":
                current_regime_floor = 70.0
            else:
                current_regime_floor = 62.0  # MODERATE / HIGH_VOLATILITY / REVERSAL_HEAVY
            fallback_threshold = current_regime_floor
            continue
        
        break
    
    final = _select_ranked_with_cooldown(candidates, total_target=total_target)

    # ── Adaptive count adjustment ─────────────────────────────────
    # If we have enough strong candidates, allow up to target_max.
    # If we have fewer candidates than target_min, that's ok — quality over quantity.
    strong_count = sum(1 for s in candidates if s.get("signal_tier") == TIER_STRONG)
    if strong_count >= target_max and len(final) < target_max:
        extra = _select_ranked_with_cooldown(candidates, total_target=target_max)
        # Only extend to target_max if we have genuinely strong signals to fill it
        if len(extra) > len(final):
            final = extra
            logger.info(
                f"[Adaptive] Extended to target_max={target_max} — "
                f"{strong_count} STRONG candidates available"
            )

    final.sort(key=lambda x: x["time"])

    # Log final selected set
    for signal in final:
        logger.info(
            "[Pattern] %s %s | tier=%s | prob=%.1f | conf=%d%% | "
            "pat_str=%d | win=%.1f%% | regime=%s | zone=%s",
            signal["time"], signal["direction"],
            signal.get("signal_tier", "?"),
            signal.get("probability_score", 0.0),
            signal["confidence"],
            signal["pattern_strength"],
            signal["historical_success_rate"],
            signal.get("market_regime", "?"),
            signal.get("volatility_zone", "?"),
        )

    # ── Build & store SignalDiagnostics ──────────────────────────────────
    global _last_diagnostics
    try:
        # Rejection counts by category
        reject_counts: dict[str, int] = {}
        for _t, _d, _reason in rejected_signals_list:
            cat = _categorise_rejection(_reason)
            reject_counts[cat] = reject_counts.get(cat, 0) + 1

        # Accepted signal quality stats
        _accepted_scores = [
            s.get("probability_score", 0.0) for s in candidates
        ]
        _avg_score = round(sum(_accepted_scores) / len(_accepted_scores), 1) if _accepted_scores else 0.0
        _strong_n  = sum(1 for s in candidates if s.get("signal_tier") == TIER_STRONG)
        _mod_n     = sum(1 for s in candidates if s.get("signal_tier") == TIER_MODERATE)

        # Top 5 by prob score for the Telegram message
        _top = sorted(
            [(s["time"], s["direction"], s.get("probability_score", 0.0)) for s in candidates],
            key=lambda x: x[2], reverse=True,
        )[:5]

        _last_diagnostics = SignalDiagnostics(
            date                = today_ist,
            regime              = regime,
            regime_confidence   = float(regime_confidence),
            currency_bias       = cs_result.bias,
            currency_eur        = cs_result.eur_strength,
            currency_usd        = cs_result.usd_strength,
            sweep_detected      = bool(market_structure.has_strong_sweep or (
                market_structure.sweep_result and market_structure.sweep_result.detected
            )),
            sweep_direction     = market_structure.sweep_direction or "",
            generated_candidates= generated_candidates_count,
            accepted_count      = len(accepted_signals_list),
            final_count         = len(final),
            rejection_counts    = reject_counts,
            rejection_details   = list(rejected_signals_list),
            strong_count        = _strong_n,
            moderate_count      = _mod_n,
            avg_prob_score      = _avg_score,
            top_times           = _top,
        )
        logger.debug("[Diagnostics] Stored: %s", _last_diagnostics)
    except Exception as _diag_err:
        logger.warning("[Diagnostics] Failed to build diagnostics (non-critical): %s", _diag_err)

    # ── Summary log ────────────────────────────────────────────────────────
    logger.info(
        "\n--- Signal Generation Summary (%s) ---\n"
        "  Generated candidates : %d\n"
        "  Accepted (all gates) : %d\n"
        "  Rejected             : %d (%.1f%%)\n"
        "  Final (post-rank)    : %d\n"
        "  Dominant reject gate : %s",
        today_ist,
        generated_candidates_count,
        len(accepted_signals_list),
        len(rejected_signals_list),
        (_last_diagnostics.rejection_rate() if _last_diagnostics else 0.0),
        len(final),
        (_last_diagnostics.dominant_reject_reason() if _last_diagnostics else "N/A"),
    )
    
    logger.info("--- Generated Candidates ---")
    logger.info(f"Total candidates generated: {generated_candidates_count}")
    
    logger.info("--- Accepted Signals ---")
    if accepted_signals_list:
        for a in accepted_signals_list:
            logger.info("  Accepted %s %s", a[0], a[1])
    else:
        logger.info("  No signals accepted.")
        
    logger.info("--- Rejected Signals ---")
    if rejected_signals_list:
        for r in rejected_signals_list:
            logger.info("  Rejected %s %s → %s", r[0], r[1], r[2])
    else:
        logger.info("  No signals rejected.")

    return final



def _today_ist_str() -> str:
    return pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d")


def has_run_today() -> bool:
    try:
        state = safe_load_json(STATE_FILE, default={})
        return state.get("last_run_date") == _today_ist_str()
    except Exception:
        return False


def update_run_state() -> None:
    safe_save_json(STATE_FILE, {"last_run_date": _today_ist_str(), "timezone": "Asia/Kolkata"})


# ── Main daily generation ─────────────────────────────────
def generate_daily_signals() -> bool:
    if has_run_today():
        logger.info("Signals already generated for today. Skipping.")
        return False

    logger.info(f"--- Generating Daily Signals for {PAIR} ---")
    df = get_historical_data()

    if df is None:
        logger.error("Failed to fetch data.")
        return False

    if not is_df_cache_fresh():
        logger.warning("Cached market data too old — skipping generation.")
        return False

    # ── Run 14-day backtest before signal selection ─────────────────────────
    # Backtesting uses the already-fetched df — no extra API calls.
    # Results are used to:
    #   1. Update dynamic probability weights (backtest performance)
    #   2. Update dynamic voter weights (rolling win-rate per voter)
    #   3. Seed timing_db with historically-grounded pattern strengths
    try:
        bt_engine = BacktestingEngine()
        bt_results = bt_engine.run(df)
        # Feed results to dynamic weight optimizer
        dynamic_weight_optimizer.update_from_backtest(bt_results)
        # Seed timing_db with backtest win rates for unseen slots
        timing_db.update_from_backtest(bt_results.timing_win_rates)
        # ── Daily voter weight update ────────────────────────────────────────
        # Recompute per-voter multipliers from their rolling prediction records.
        # These multipliers scale formula weights in compute_with_voter_weights().
        try:
            new_voter_weights = strategy_weight_tracker.run_daily_update()
            logger.info(
                "[StrategyWeights] Voter multipliers updated: %s",
                " ".join(f"{k[:8]}={v:.2f}" for k, v in new_voter_weights.items()),
            )
        except Exception as we:
            logger.warning("[StrategyWeights] Daily update failed (non-critical): %s", we)
        logger.info(
            "[Backtest] Integrated: overall_wr=%.1f%% top_regime=%s",
            bt_results.overall_win_rate, bt_results.top_regime,
        )
    except Exception as e:
        logger.warning("[Backtest] Failed (non-critical): %s", e)

    signals = calculate_recurring_strength(df)

    if signals:
        safe_save_json(SIGNAL_FILE, signals)
        logger.info(f"Generated {len(signals)} strong recurring signals.")
        update_run_state()
        return True
    else:
        logger.info("No signals met the pattern threshold today.")
        return False


# ── Forced daily signals (15:05 + 15:10) ─────────────────
def generate_forced_daily_signals(df: pd.DataFrame | None = None) -> list[dict]:
    logger.info("--- Generating FORCED daily signals (15:05 + 15:10) ---")

    if df is None or len(df) < 50:
        df = _load_df_cache()
    if df is None or len(df) < 50:
        df = get_historical_data(outputsize=500)
    if df is None or len(df) < 50:
        df = None

    direction, confidence = decide_direction_live(df)

    if direction is None:
        logger.warning("Skipping forced signals — no live direction (data unavailable).")
        return []

    low_confidence = confidence < FORCE_SIGNAL_CONFIDENCE_THRESHOLD
    if low_confidence:
        logger.warning(f"Forced signal LOW confidence ({confidence}%) — tagging as RISKY")

    direct_signal = {
        "time": FORCE_DIRECT_TIME,
        "pair": PAIR,
        "direction": direction,
        "confidence": confidence,
        "source": "forced",
        "signal_type": "direct",
        "low_confidence": low_confidence,
    }
    martingale_signal = {
        "time": FORCE_MARTINGALE_TIME,
        "pair": PAIR,
        "direction": direction,
        "confidence": confidence,
        "source": "forced",
        "signal_type": "martingale",
        "low_confidence": low_confidence,
    }
    forced_signals = [direct_signal, martingale_signal]

    existing = safe_load_json(SIGNAL_FILE, default=[])
    if not isinstance(existing, list):
        existing = []

    forced_times = {FORCE_DIRECT_TIME, FORCE_MARTINGALE_TIME}
    existing = [s for s in existing if s.get("time") not in forced_times]

    all_signals = existing + forced_signals
    filtered = []
    for s in all_signals:
        t_str = s.get("time")
        if not t_str:
            continue
        try:
            h, m = map(int, t_str.split(":"))
            if 13 * 60 <= h * 60 + m <= 22 * 60:
                filtered.append(s)
        except Exception:
            continue

    filtered.sort(key=lambda x: x.get("time", ""))
    try:
        safe_save_json(SIGNAL_FILE, filtered)
        logger.info(
            f"Forced signals saved: {direction} @ {FORCE_DIRECT_TIME} & "
            f"{FORCE_MARTINGALE_TIME} | confidence={confidence}%"
        )
    except Exception as e:
        logger.error(f"Could not save forced signals: {e}")

    return forced_signals


if __name__ == "__main__":
    generate_daily_signals()
    generate_forced_daily_signals()
