from logger import logger
import os
import json
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from indicators import add_indicators
from learning_engine import learning_engine
from persistence import safe_load_json, safe_save_json
from signal_manager import timing_db

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
            df = pd.read_pickle(DF_CACHE_FILE)
            _df_memory_cache = df
            _df_memory_cache_time = saved_at if os.path.exists(_ts_file) else datetime.utcnow()
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
    df["TimeOfDay"] = df["datetime"].dt.strftime("%H:%M")

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

    # Composite score with weights
    composite = (
        wr            * 0.30 +   # historical direction success
        ema_align     * 0.15 +   # EMA trend alignment
        rsi_cont      * 0.10 +   # RSI momentum continuation
        atr_stability * 0.10 +   # ATR stability / activity
        mom_cont      * 0.10 +   # candle momentum continuation
        reversal_score* 0.10 +   # low reversal probability
        candle_strength*0.10 +   # candle body strength
        session_strength*0.05    # session weighting (5% placeholder)
    )

    # Penalize weak/noisy timings directly in composite
    if wr < 60.0:
        composite -= 15.0  # heavy penalty for weak historical success

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


def calculate_recurring_strength(df: pd.DataFrame) -> list[dict]:
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
def has_run_today() -> bool:
    try:
        state = safe_load_json(STATE_FILE, default={})
        return state.get("last_run_date") == datetime.now().strftime("%Y-%m-%d")
    except Exception:
        return False


def update_run_state() -> None:
    safe_save_json(STATE_FILE, {"last_run_date": datetime.now().strftime("%Y-%m-%d")})


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
