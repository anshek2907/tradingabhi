"""
market_regime.py
────────────────
Market Regime Detection Engine for the Advanced Signal Bot.

Detects one of four market regimes from recent price/indicator data:
  • TRENDING        – clear directional bias, healthy ATR, consistent momentum
  • SIDEWAYS        – compressed EMAs, low ATR, price ranging, frequent reversals
  • HIGH_VOLATILITY – ATR spikes, large candle bodies, unstable price swings
  • REVERSAL_HEAVY  – frequent direction changes, wick-heavy candles, inconsistent bias

Returns a full regime report used by signal_generator.py to:
  - Adjust signal count targets
  - Adapt confidence thresholds
  - Apply regime-aware filtering per slot

Usage:
    from market_regime import detect_market_regime, get_regime_behavior
    regime_report = detect_market_regime(df)
    behavior      = get_regime_behavior(regime_report["regime"])
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from logger import logger


# ── Regime labels ─────────────────────────────────────────
REGIME_TRENDING        = "TRENDING"
REGIME_SIDEWAYS        = "SIDEWAYS"
REGIME_HIGH_VOLATILITY = "HIGH_VOLATILITY"
REGIME_REVERSAL_HEAVY  = "REVERSAL_HEAVY"

ALL_REGIMES = [REGIME_TRENDING, REGIME_SIDEWAYS, REGIME_HIGH_VOLATILITY, REGIME_REVERSAL_HEAVY]


# ── Regime behavior presets (v2 — balanced safe-profit) ──────────────
# Signal count targets:
#   TRENDING        : 8–15 signals  (allow quality continuation signals)
#   MODERATE/MIXED  : 5–10 signals  (balanced selection)
#   SIDEWAYS        : 2–5 signals   (strict — avoid choppy noise)
#   HIGH_VOL/REV    : 2–5 signals   (maximum caution)
_REGIME_BEHAVIOR = {
    REGIME_TRENDING: {
        "target":            15,
        "target_min":        10,
        "target_max":        15,
        "threshold":         68,
        "min_pattern_str":   70,
        "reversal_prob_max": 40.0,
        "atr_ratio_min":     0.80,
        "atr_ratio_max":     3.00,
        "allow_weak_setups": False,
        "description":       "Strong trend — target 10-15 momentum-aligned signals",
    },
    REGIME_SIDEWAYS: {
        "target":            10,
        "target_min":        6,
        "target_max":        10,
        "threshold":         68,
        "min_pattern_str":   75,
        "reversal_prob_max": 35.0,
        "atr_ratio_min":     0.50,
        "atr_ratio_max":     2.00,
        "allow_weak_setups": False,
        "description":       "Ranging — target 6-10 high-probability boundary reversion signals",
    },
    REGIME_HIGH_VOLATILITY: {
        "target":            6,
        "target_min":        3,
        "target_max":        6,
        "threshold":         68,
        "min_pattern_str":   80,
        "reversal_prob_max": 25.0,
        "atr_ratio_min":     1.20,
        "atr_ratio_max":     4.00,
        "allow_weak_setups": False,
        "description":       "High volatility — target 3-6 ultra-safe extreme setups",
    },
    REGIME_REVERSAL_HEAVY: {
        "target":            6,
        "target_min":        3,
        "target_max":        6,
        "threshold":         78,
        "min_pattern_str":   75,
        "reversal_prob_max": 50.0,
        "atr_ratio_min":     0.70,
        "atr_ratio_max":     2.10,
        "allow_weak_setups": False,
        "description":       "Reversal-heavy — target 3-6 proven signals with strong confirmation",
    },
}



def get_regime_behavior(regime: str) -> dict:
    """Return behavior config dict for the given regime label."""
    return dict(_REGIME_BEHAVIOR.get(regime, _REGIME_BEHAVIOR[REGIME_SIDEWAYS]))


# ── Internal helpers ───────────────────────────────────────

def _safe_float(series: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(series.dropna().iloc[-1] if len(series.dropna()) > 0 else default)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _compute_ema_slope(ema50: pd.Series, lookback: int = 20, atr_now: float = 0.0001) -> float:
    """
    Normalised EMA50 slope over the last `lookback` candles.
    Positive = bullish trend, negative = bearish trend.
    Normalised by ATR so it's scale-independent.
    """
    try:
        valid = ema50.dropna()
        if len(valid) < lookback:
            return 0.0
        slope_raw = float(valid.iloc[-1]) - float(valid.iloc[-lookback])
        return slope_raw / max(atr_now * lookback, 1e-8)
    except Exception:
        return 0.0


def _compute_reversal_freq(df: pd.Series, window: int = 40) -> float:
    """Fraction of candles (0-1) that reversed the previous candle's direction."""
    try:
        tail = df.tail(window)
        if len(tail) < 4:
            return 0.5
        result_call = (tail["Close"] > tail["Open"]).astype(int)
        reversal = (result_call != result_call.shift(1)) & result_call.shift(1).notna()
        return float(reversal.mean())
    except Exception:
        return 0.5


def _compute_trend_continuation(df: pd.DataFrame, window: int = 40) -> float:
    """Fraction of candles that continue the previous candle's direction."""
    try:
        tail = df.tail(window)
        if len(tail) < 4:
            return 0.5
        result = (tail["Close"] > tail["Open"]).astype(int)
        continuation = (result == result.shift(1)) & result.shift(1).notna()
        return float(continuation.mean())
    except Exception:
        return 0.5


def _compute_atr_ratio(df: pd.DataFrame, window: int = 96) -> tuple[float, float]:
    """
    Returns (current_atr_ratio, spike_fraction).
    atr_ratio = ATR[-1] / median(ATR[-window:])
    spike_fraction = fraction of last 40 candles where atr_ratio > 2.0
    """
    try:
        atr = pd.to_numeric(df["ATR"], errors="coerce")
        atr_now = float(atr.dropna().iloc[-1]) if len(atr.dropna()) > 0 else 0.0
        atr_median = float(atr.tail(window).median())
        if atr_median <= 0:
            return 1.0, 0.0
        ratio = atr_now / atr_median
        # Spike fraction over last 40
        tail_ratios = (atr.tail(40) / atr_median).dropna()
        spike_frac = float((tail_ratios > 2.0).mean()) if len(tail_ratios) > 0 else 0.0
        return ratio, spike_frac
    except Exception:
        return 1.0, 0.0


def _compute_ema_compression(df: pd.DataFrame, window: int = 20) -> float:
    """
    EMA compression score (0-1): how tightly EMA50 and EMA200 are squeezed.
    Returns higher value when EMAs are compressed.
    """
    try:
        ema50  = pd.to_numeric(df["EMA50"],  errors="coerce").dropna()
        ema200 = pd.to_numeric(df["EMA200"], errors="coerce").dropna()
        atr    = pd.to_numeric(df["ATR"],    errors="coerce").dropna()
        if len(ema50) < 2 or len(ema200) < 2 or len(atr) < 2:
            return 0.5
        sep = abs(float(ema50.iloc[-1]) - float(ema200.iloc[-1]))
        atr_now = float(atr.iloc[-1])
        # If separation is less than 0.5× ATR → highly compressed
        ratio = sep / max(atr_now, 1e-8)
        # Map: ratio=0 → compression=1.0, ratio=2 → compression=0.0
        compression = max(0.0, min(1.0, 1.0 - ratio / 2.0))
        return compression
    except Exception:
        return 0.5


def _compute_wick_ratio(df: pd.DataFrame, window: int = 30) -> float:
    """Average ratio of total wick length to body size over last N candles."""
    try:
        tail = df.tail(window)
        body = (tail["Close"] - tail["Open"]).abs().clip(lower=1e-8)
        upper_wick = tail["High"] - tail[["Open", "Close"]].max(axis=1)
        lower_wick = tail[["Open", "Close"]].min(axis=1) - tail["Low"]
        total_wick = upper_wick + lower_wick
        wick_ratio = (total_wick / body).clip(upper=10).mean()
        return float(wick_ratio)
    except Exception:
        return 1.0


def _compute_direction_consistency(df: pd.DataFrame, window: int = 30) -> float:
    """
    Fraction (0-1) of candles that agree with the dominant direction.
    High = market has a strong directional bias; low = choppy.
    """
    try:
        tail = df.tail(window)
        bullish = float((tail["Close"] > tail["Open"]).mean())
        return max(bullish, 1.0 - bullish)  # always return the dominant side's fraction
    except Exception:
        return 0.5


def _candle_body_strength(df: pd.DataFrame, window: int = 30) -> float:
    """Average body-to-range ratio over last N candles (0-1)."""
    try:
        tail = df.tail(window)
        body  = (tail["Close"] - tail["Open"]).abs()
        range_ = (tail["High"] - tail["Low"]).clip(lower=1e-8)
        return float((body / range_).mean())
    except Exception:
        return 0.5


# ── Volatility zone classifier ─────────────────────────────

def classify_volatility_zone(slot_atr_ratio: float, slot_noisy_pct: float) -> str:
    """
    Classify the volatility quality of a time slot.

    Returns one of:
        "healthy"        – good ATR, not noisy
        "dead"           – very low ATR (dead market)
        "noisy"          – low ATR but choppy/sideways
        "unstable_spike" – ATR spike, unreliable

    Args:
        slot_atr_ratio  : slot ATR / market ATR median
        slot_noisy_pct  : fraction of candles flagged as Noisy_Sideways (0-1)
    """
    if slot_atr_ratio < 0.45:
        return "dead"
    if slot_atr_ratio > 2.50:
        return "unstable_spike"
    if slot_noisy_pct > 0.40:
        return "noisy"
    return "healthy"


# ── Main regime detection ──────────────────────────────────

def detect_market_regime(
    df: pd.DataFrame,
    lookback_candles: int = 80,
    reversal_window: int  = 40,
) -> dict:
    """
    Analyse recent price/indicator data and return a full regime report.

    Args:
        df               : enriched DataFrame (with EMA50, EMA200, ATR, RSI cols)
        lookback_candles : number of recent candles to use for regime analysis
        reversal_window  : window for reversal/continuation stats

    Returns:
        {
            "regime": str,              # dominant regime label
            "confidence": int,          # 0-100 confidence in dominant regime
            "regime_scores": dict,      # raw score per regime
            "ema_slope": float,
            "atr_ratio": float,
            "atr_spike_frac": float,
            "reversal_freq": float,
            "trend_continuation": float,
            "ema_compression": float,
            "wick_ratio": float,
            "direction_consistency": float,
            "candle_body_strength": float,
            "behavior": dict,           # regime behavior config
        }
    """
    if df is None or len(df) < max(lookback_candles, 50):
        logger.warning("[Regime] Insufficient data for regime detection — defaulting to SIDEWAYS")
        return _default_regime()

    # Work on a tail slice for efficiency
    recent = df.tail(lookback_candles).copy()

    # ── Raw metrics ────────────────────────────────────────
    atr_ratio, atr_spike_frac = _compute_atr_ratio(df)
    atr_now = _safe_float(pd.to_numeric(recent["ATR"], errors="coerce"), default=0.0001)
    ema_slope         = _compute_ema_slope(pd.to_numeric(df["EMA50"], errors="coerce"), lookback=20, atr_now=atr_now)
    ema_slope_abs     = abs(ema_slope)
    ema_compression   = _compute_ema_compression(recent)
    reversal_freq     = _compute_reversal_freq(recent, window=reversal_window)
    trend_cont        = _compute_trend_continuation(recent, window=reversal_window)
    wick_ratio        = _compute_wick_ratio(recent)
    dir_consistency   = _compute_direction_consistency(recent)
    body_strength     = _candle_body_strength(recent)

    # ATR health: healthy if 0.65 ≤ ratio ≤ 1.80
    atr_healthy       = 1.0 if 0.65 <= atr_ratio <= 1.80 else max(0.0, 1.0 - abs(atr_ratio - 1.20) / 1.20)
    low_atr           = max(0.0, 1.0 - atr_ratio / 0.65) if atr_ratio < 0.65 else 0.0

    # Normalise EMA slope for scoring (sigmoid-like clamp)
    slope_score = min(1.0, ema_slope_abs * 5)  # 0-1, strong at |slope| >= 0.2

    # ── Per-regime composite scores (0-100) ───────────────

    trending_score = (
        slope_score       * 0.35 +
        trend_cont        * 0.35 +
        atr_healthy       * 0.20 +
        dir_consistency   * 0.10
    ) * 100

    sideways_score = (
        ema_compression   * 0.40 +
        low_atr           * 0.20 +
        (1.0 - dir_consistency) * 0.20 +
        reversal_freq     * 0.20
    ) * 100

    high_vol_score = (
        atr_spike_frac    * 0.50 +
        min(1.0, max(0.0, (atr_ratio - 1.80) / 1.20)) * 0.30 +
        (1.0 - body_strength) * 0.20            # spike but chaotic bodies
    ) * 100

    reversal_score = (
        reversal_freq     * 0.45 +
        min(1.0, wick_ratio / 3.0) * 0.30 +
        (1.0 - dir_consistency) * 0.25
    ) * 100

    regime_scores = {
        REGIME_TRENDING:        round(trending_score,  1),
        REGIME_SIDEWAYS:        round(sideways_score,  1),
        REGIME_HIGH_VOLATILITY: round(high_vol_score,  1),
        REGIME_REVERSAL_HEAVY:  round(reversal_score,  1),
    }

    # ── Dominant regime ────────────────────────────────────
    dominant_regime = max(regime_scores, key=regime_scores.get)

    # Confidence: margin between dominant and second-best score
    sorted_scores = sorted(regime_scores.values(), reverse=True)
    top_score    = sorted_scores[0]
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin       = top_score - second_score  # 0-100

    # Confidence = top_score * 0.60 + margin * 0.40
    raw_confidence = top_score * 0.60 + margin * 0.40
    confidence = max(0, min(100, int(round(raw_confidence))))

    report = {
        "regime":               dominant_regime,
        "confidence":           confidence,
        "regime_scores":        regime_scores,
        "ema_slope":            round(ema_slope, 6),
        "atr_ratio":            round(atr_ratio, 3),
        "atr_spike_frac":       round(atr_spike_frac, 3),
        "reversal_freq":        round(reversal_freq, 3),
        "trend_continuation":   round(trend_cont, 3),
        "ema_compression":      round(ema_compression, 3),
        "wick_ratio":           round(wick_ratio, 3),
        "direction_consistency": round(dir_consistency, 3),
        "candle_body_strength": round(body_strength, 3),
        "behavior":             get_regime_behavior(dominant_regime),
    }

    logger.info(
        f"[Regime] Detected: {dominant_regime} | confidence={confidence}% | "
        f"Scores: TREND={regime_scores[REGIME_TRENDING]:.0f} "
        f"SIDE={regime_scores[REGIME_SIDEWAYS]:.0f} "
        f"HVOL={regime_scores[REGIME_HIGH_VOLATILITY]:.0f} "
        f"REV={regime_scores[REGIME_REVERSAL_HEAVY]:.0f}"
    )

    return report


def _default_regime() -> dict:
    """Return a safe default regime when data is insufficient."""
    return {
        "regime":               REGIME_SIDEWAYS,
        "confidence":           30,
        "regime_scores":        {r: 25 for r in ALL_REGIMES},
        "ema_slope":            0.0,
        "atr_ratio":            1.0,
        "atr_spike_frac":       0.0,
        "reversal_freq":        0.5,
        "trend_continuation":   0.5,
        "ema_compression":      0.5,
        "wick_ratio":           1.0,
        "direction_consistency": 0.5,
        "candle_body_strength": 0.5,
        "behavior":             get_regime_behavior(REGIME_SIDEWAYS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Martingale Controller
# ─────────────────────────────────────────────────────────────────────────────

class MartingaleAdaptiveController:
    """
    Regime-aware martingale gate.

    Rules (per spec):
    ┌────────────────────┬────────────────────────────────────────────────────┐
    │ Regime             │ Martingale Decision                                │
    ├────────────────────┼────────────────────────────────────────────────────┤
    │ TRENDING           │ ALLOW MG1 — only when regime confidence ≥ MIN_CONF │
    │ SIDEWAYS           │ BLOCK — choppy markets invalidate MG logic         │
    │ HIGH_VOLATILITY    │ BLOCK — unpredictable swings, avoid doubling       │
    │ REVERSAL_HEAVY     │ BLOCK — frequent flips destroy martingale chains   │
    │ Unknown / default  │ BLOCK (safe default)                               │
    └────────────────────┴────────────────────────────────────────────────────┘

    Confidence gate:
        Even in TRENDING, martingale is blocked if regime_confidence < MIN_CONF.
        This prevents MG on a weakly-detected trend that may actually be noise.

    Usage:
        from market_regime import MartingaleAdaptiveController
        result = MartingaleAdaptiveController.evaluate(regime_report)
        if result.allowed:
            # proceed with martingale
        else:
            logger.info(result.reason)
    """

    # Minimum regime confidence required to allow martingale (even in TRENDING)
    MIN_CONFIDENCE: int = 55

    # ── Decision result ──────────────────────────────────────────────────

    class Result:
        """Immutable martingale decision result."""
        __slots__ = ("allowed", "regime", "regime_confidence", "reason")

        def __init__(
            self,
            allowed:           bool,
            regime:            str,
            regime_confidence: int,
            reason:            str,
        ) -> None:
            self.allowed           = allowed
            self.regime            = regime
            self.regime_confidence = regime_confidence
            self.reason            = reason

        def __repr__(self) -> str:
            status = "ALLOWED" if self.allowed else "BLOCKED"
            return (
                f"MartingaleResult({status} | {self.regime} "
                f"conf={self.regime_confidence}% | {self.reason})"
            )

        def as_dict(self) -> dict:
            return {
                "martingale_allowed":    self.allowed,
                "regime":                self.regime,
                "regime_confidence":     self.regime_confidence,
                "martingale_reason":     self.reason,
            }

    # ── Core API ─────────────────────────────────────────────────────────

    @classmethod
    def evaluate(
        cls,
        regime_report: dict,
    ) -> "MartingaleAdaptiveController.Result":
        """
        Evaluate whether martingale is permitted given the current regime.

        Args:
            regime_report: dict returned by detect_market_regime() containing
                           at minimum 'regime' (str) and 'confidence' (int).

        Returns:
            MartingaleAdaptiveController.Result
        """
        regime     = regime_report.get("regime", REGIME_SIDEWAYS)
        confidence = int(regime_report.get("confidence", 0))

        # ── Rule 1: only TRENDING allows martingale ───────────────────────
        if regime == REGIME_SIDEWAYS:
            reason = (
                f"Martingale blocked — SIDEWAYS regime (conf={confidence}%): "
                "choppy price action invalidates martingale logic"
            )
            logger.info("[MG-Ctrl] %s", reason)
            return cls.Result(False, regime, confidence, reason)

        if regime == REGIME_HIGH_VOLATILITY:
            reason = (
                f"Martingale blocked — HIGH_VOLATILITY regime (conf={confidence}%): "
                "unpredictable swings make doubling dangerous"
            )
            logger.info("[MG-Ctrl] %s", reason)
            return cls.Result(False, regime, confidence, reason)

        if regime == REGIME_REVERSAL_HEAVY:
            reason = (
                f"Martingale blocked — REVERSAL_HEAVY regime (conf={confidence}%): "
                "frequent direction flips destroy martingale chains"
            )
            logger.info("[MG-Ctrl] %s", reason)
            return cls.Result(False, regime, confidence, reason)

        if regime != REGIME_TRENDING:
            # Unknown/unexpected regime — block by default (safety-first)
            reason = (
                f"Martingale blocked — unknown regime '{regime}' (conf={confidence}%): "
                "defaulting to block for safety"
            )
            logger.warning("[MG-Ctrl] %s", reason)
            return cls.Result(False, regime, confidence, reason)

        # ── Rule 2: TRENDING — check confidence gate ───────────────────────
        if confidence < cls.MIN_CONFIDENCE:
            reason = (
                f"Martingale blocked — TRENDING regime confidence too low "
                f"({confidence}% < {cls.MIN_CONFIDENCE}% required): "
                "trend may be noise, not a reliable continuation"
            )
            logger.info("[MG-Ctrl] %s", reason)
            return cls.Result(False, regime, confidence, reason)

        # ── Allow ─────────────────────────────────────────────────────────
        reason = (
            f"Martingale ALLOWED — TRENDING regime, confidence={confidence}% "
            f"(>= {cls.MIN_CONFIDENCE}% required): MG1 permitted"
        )
        logger.info("[MG-Ctrl] %s", reason)
        return cls.Result(True, regime, confidence, reason)

    @classmethod
    def evaluate_from_df(
        cls,
        df,
        lookback_candles: int = 80,
    ) -> "MartingaleAdaptiveController.Result":
        """
        Convenience wrapper: detect regime from DataFrame then evaluate.

        Args:
            df:               enriched DataFrame (with EMA50, EMA200, ATR, RSI)
            lookback_candles: passed through to detect_market_regime()

        Returns:
            MartingaleAdaptiveController.Result
        """
        try:
            regime_report = detect_market_regime(df, lookback_candles=lookback_candles)
        except Exception as exc:
            logger.warning("[MG-Ctrl] Regime detection failed, blocking MG: %s", exc)
            return cls.Result(
                False, REGIME_SIDEWAYS, 0,
                f"Martingale blocked — regime detection error: {exc}",
            )
        return cls.evaluate(regime_report)

    @classmethod
    def is_allowed(cls, regime_report: dict) -> bool:
        """
        Thin boolean shortcut.  Returns True iff martingale is permitted.

        Args:
            regime_report: dict from detect_market_regime()
        """
        return cls.evaluate(regime_report).allowed

    @classmethod
    def is_allowed_from_df(cls, df) -> bool:
        """
        Thin boolean shortcut using DataFrame directly.

        Args:
            df: enriched DataFrame
        """
        return cls.evaluate_from_df(df).allowed


# ── Module-level singleton helper ─────────────────────────────────────────────
martingale_controller = MartingaleAdaptiveController()
