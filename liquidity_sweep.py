"""
liquidity_sweep.py
──────────────────
Liquidity Sweep Detection Engine.

Detects when price breaks through a key liquidity level and then rejects
back, indicating a stop-hunt / liquidity grab — a high-probability signal
for a reversal or continuation trap.

Sweep Types Detected
────────────────────
  BSL       – Buy-Side Liquidity sweep  (swing high broken, price rejects down)
  SSL       – Sell-Side Liquidity sweep (swing low broken, price rejects up)
  PDH_SWEEP – Previous Day High swept
  PDL_SWEEP – Previous Day Low swept
  WH_SWEEP  – Weekly High swept
  WL_SWEEP  – Weekly Low swept

Detection Rule
──────────────
  For a BUY-SIDE sweep (BSL / PDH / WH):
    1. Candle High > liquidity level  (price pierces above)
    2. Candle Close < liquidity level  (price rejects back below)
    3. Wick prominence: upper_wick >= body * 0.5  (wick is meaningful)
    4. Body confirmation: body >= candle_range * 0.15  (not a doji)
    → Signal direction: PUT  (price is likely to continue down after trap)

  For a SELL-SIDE sweep (SSL / PDL / WL):
    1. Candle Low < liquidity level   (price pierces below)
    2. Candle Close > liquidity level  (price rejects back above)
    3. Wick prominence: lower_wick >= body * 0.5
    4. Body confirmation: body >= candle_range * 0.15
    → Signal direction: CALL  (price is likely to continue up after trap)

Sweep Strength (0–100)
──────────────────────
  strength = (wick_pct_of_atr * 50) + (rejection_pct * 50)

  wick_pct_of_atr  = wick_into_level / ATR   (clamped 0–1)
  rejection_pct    = close_recovery / wick_total  (how much of wick was recovered)

Usage
─────
  from liquidity_sweep import detect_liquidity_sweeps, LiquiditySweepResult
  result = detect_liquidity_sweeps(df, liquidity_zones, lookback=5)
  if result.has_strong_sweep:
      print(result.sweep_summary)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from logger import logger

# ── Sweep type constants ───────────────────────────────────────────────────
SWEEP_BSL      = "BSL"        # Buy-side liquidity (swing highs)
SWEEP_SSL      = "SSL"        # Sell-side liquidity (swing lows)
SWEEP_PDH      = "PDH_SWEEP"  # Previous Day High
SWEEP_PDL      = "PDL_SWEEP"  # Previous Day Low
SWEEP_WH       = "WH_SWEEP"   # Weekly High
SWEEP_WL       = "WL_SWEEP"   # Weekly Low

# Sweep direction: what direction trade is expected AFTER the sweep
# BSL/PDH/WH sweeps trap longs → expect price to drop → PUT
# SSL/PDL/WL sweeps trap shorts → expect price to rise → CALL
_SWEEP_DIRECTION_MAP = {
    SWEEP_BSL: "PUT",
    SWEEP_PDH: "PUT",
    SWEEP_WH:  "PUT",
    SWEEP_SSL: "CALL",
    SWEEP_PDL: "CALL",
    SWEEP_WL:  "CALL",
}

# ── Tunable thresholds ─────────────────────────────────────────────────────
STRONG_SWEEP_THRESHOLD  = 65.0   # strength >= 65 → has_strong_sweep = True
MIN_WICK_BODY_RATIO     = 0.50   # wick must be >= 50% of body
MIN_BODY_RANGE_RATIO    = 0.15   # body must be >= 15% of candle range (filter doji)
DEFAULT_LOOKBACK        = 5      # candles to scan for sweeps


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class SweepEvent:
    """Single detected sweep event."""
    sweep_type:    str    = ""      # SWEEP_* constant
    direction:     str    = ""      # "CALL" | "PUT" — expected post-sweep direction
    level:         float  = 0.0    # the liquidity level that was swept
    sweep_high:    float  = 0.0    # candle high (highest point of sweep candle)
    sweep_low:     float  = 0.0    # candle low
    close:         float  = 0.0    # close price (rejected back)
    sweep_strength: float = 0.0   # 0–100 decisiveness score
    candles_ago:   int    = 0      # 0 = current candle, 1 = previous, etc.
    is_strong:     bool   = False  # True if strength >= STRONG_SWEEP_THRESHOLD

    def short_str(self) -> str:
        age_str = "current" if self.candles_ago == 0 else f"{self.candles_ago}c ago"
        return (
            f"{self.sweep_type}({self.direction}) "
            f"lvl={self.level:.5f} str={self.sweep_strength:.1f} [{age_str}]"
        )


@dataclass
class LiquiditySweepResult:
    """Aggregate result of sweep detection across a lookback window."""
    detected:           bool         = False
    sweeps:             list         = field(default_factory=list)   # list[SweepEvent]
    dominant_direction: str          = "NEUTRAL"   # "CALL" | "PUT" | "NEUTRAL"
    sweep_confidence:   float        = 0.0          # 0–100 composite confidence
    has_strong_sweep:   bool         = False
    sweep_summary:      str          = ""

    def as_dict(self) -> dict:
        return {
            "detected":           self.detected,
            "dominant_direction": self.dominant_direction,
            "sweep_confidence":   round(self.sweep_confidence, 1),
            "has_strong_sweep":   self.has_strong_sweep,
            "sweep_count":        len(self.sweeps),
            "sweep_summary":      self.sweep_summary,
            "sweeps": [
                {
                    "type":      e.sweep_type,
                    "direction": e.direction,
                    "level":     round(e.level, 5),
                    "strength":  round(e.sweep_strength, 1),
                    "is_strong": e.is_strong,
                    "candles_ago": e.candles_ago,
                }
                for e in self.sweeps
            ],
        }


# ── Internal helpers ───────────────────────────────────────────────────────

def _sweep_strength(
    wick_into_level: float,
    close_recovery:  float,
    total_wick:      float,
    atr:             float,
) -> float:
    """
    Compute sweep decisiveness score [0–100].

    wick_pct_of_atr  – how far price stabbed into the zone (relative to ATR)
    rejection_pct    – how much of the wick was recovered by close
    """
    if atr <= 0 or total_wick <= 0:
        return 0.0

    wick_pct_of_atr = min(1.0, wick_into_level / max(atr, 1e-8))
    rejection_pct   = min(1.0, close_recovery  / max(total_wick, 1e-8))

    strength = (wick_pct_of_atr * 50.0) + (rejection_pct * 50.0)
    return round(max(0.0, min(100.0, strength)), 2)


def _check_bsl_sweep(
    candle: pd.Series,
    level:  float,
    atr:    float,
    sweep_type: str,
    candles_ago: int,
) -> Optional[SweepEvent]:
    """
    Check if a candle is a Buy-Side Liquidity sweep above `level`.

    BSL criteria:
      - High > level (pierced above)
      - Close < level (rejected back below)
      - Body is meaningful (not doji)
      - Upper wick is prominent relative to body
    """
    try:
        high  = float(candle["High"])
        low   = float(candle["Low"])
        open_ = float(candle["Open"])
        close = float(candle["Close"])

        # Must pierce above level and close below it
        if not (high > level and close < level):
            return None

        candle_range = high - low
        body         = abs(close - open_)
        upper_wick   = high - max(open_, close)
        wick_into    = high - level           # how much above level
        recovery     = level - close          # how much below level close ended

        # Filters
        if candle_range <= 0:
            return None
        if body < candle_range * MIN_BODY_RANGE_RATIO:
            return None    # doji — not a clear rejection candle
        if upper_wick < body * MIN_WICK_BODY_RATIO:
            return None    # wick too small relative to body

        strength = _sweep_strength(wick_into, recovery, upper_wick, atr)

        return SweepEvent(
            sweep_type    = sweep_type,
            direction     = _SWEEP_DIRECTION_MAP[sweep_type],
            level         = level,
            sweep_high    = high,
            sweep_low     = low,
            close         = close,
            sweep_strength = strength,
            candles_ago   = candles_ago,
            is_strong     = strength >= STRONG_SWEEP_THRESHOLD,
        )
    except Exception:
        return None


def _check_ssl_sweep(
    candle: pd.Series,
    level:  float,
    atr:    float,
    sweep_type: str,
    candles_ago: int,
) -> Optional[SweepEvent]:
    """
    Check if a candle is a Sell-Side Liquidity sweep below `level`.

    SSL criteria:
      - Low < level (pierced below)
      - Close > level (rejected back above)
      - Body is meaningful
      - Lower wick is prominent relative to body
    """
    try:
        high  = float(candle["High"])
        low   = float(candle["Low"])
        open_ = float(candle["Open"])
        close = float(candle["Close"])

        # Must pierce below level and close above it
        if not (low < level and close > level):
            return None

        candle_range = high - low
        body         = abs(close - open_)
        lower_wick   = min(open_, close) - low
        wick_into    = level - low        # how much below level
        recovery     = close - level      # how much above level close ended

        # Filters
        if candle_range <= 0:
            return None
        if body < candle_range * MIN_BODY_RANGE_RATIO:
            return None
        if lower_wick < body * MIN_WICK_BODY_RATIO:
            return None

        strength = _sweep_strength(wick_into, recovery, lower_wick, atr)

        return SweepEvent(
            sweep_type    = sweep_type,
            direction     = _SWEEP_DIRECTION_MAP[sweep_type],
            level         = level,
            sweep_high    = high,
            sweep_low     = low,
            close         = close,
            sweep_strength = strength,
            candles_ago   = candles_ago,
            is_strong     = strength >= STRONG_SWEEP_THRESHOLD,
        )
    except Exception:
        return None


def _detect_swing_levels(df: pd.DataFrame, pivot_window: int = 5) -> tuple[float | None, float | None]:
    """
    Identify the most recent swing high and swing low from the DataFrame
    using a simple rolling pivot detection.

    Returns (swing_high, swing_low) — either may be None if insufficient data.
    """
    if df is None or len(df) < pivot_window * 2 + 1:
        return None, None

    window = pivot_window * 2 + 1
    try:
        rolling_max = df["High"].rolling(window=window, center=True).max()
        rolling_min = df["Low"].rolling(window=window, center=True).min()

        pivot_highs = df[df["High"] == rolling_max]["High"].dropna()
        pivot_lows  = df[df["Low"]  == rolling_min]["Low"].dropna()

        swing_high = float(pivot_highs.iloc[-1]) if not pivot_highs.empty else None
        swing_low  = float(pivot_lows.iloc[-1])  if not pivot_lows.empty  else None
        return swing_high, swing_low
    except Exception:
        return None, None


# ── Main public function ───────────────────────────────────────────────────

def detect_liquidity_sweeps(
    df:               pd.DataFrame,
    liquidity_zones:  dict,
    lookback:         int = DEFAULT_LOOKBACK,
) -> LiquiditySweepResult:
    """
    Scan the last `lookback` candles for liquidity sweep events.

    Args:
        df:               Enriched OHLCV DataFrame with ATR column.
        liquidity_zones:  Dict from MarketStructureResult.liquidity_zones.
                          Expected keys: "PDH", "PDL", "WH", "WL" (may be None).
        lookback:         Number of recent candles to inspect (default=5).

    Returns:
        LiquiditySweepResult with all detected events, dominant direction,
        composite confidence, and sweep summary string.
    """
    empty = LiquiditySweepResult()

    if df is None or len(df) < max(lookback + 1, 20):
        return empty

    # ── Get average ATR for strength normalisation ─────────────────────────
    try:
        atr_series = pd.to_numeric(df["ATR"], errors="coerce")
        atr = float(atr_series.tail(50).mean())
        if atr <= 0 or pd.isna(atr):
            atr = float(df["Close"].iloc[-1]) * 0.001
    except Exception:
        atr = 0.001

    # ── Collect liquidity levels to check ─────────────────────────────────
    # Each entry: (level_value, bsl_sweep_type, ssl_sweep_type)
    zones = liquidity_zones or {}
    pdh = zones.get("PDH")
    pdl = zones.get("PDL")
    wh  = zones.get("WH")
    wl  = zones.get("WL")

    # Swing high/low from the df itself
    swing_high, swing_low = _detect_swing_levels(df)

    # Levels to check for BSL sweeps (price breaks above, rejects)
    bsl_levels: list[tuple[float, str]] = []   # (level, sweep_type)
    if pdh:
        bsl_levels.append((float(pdh), SWEEP_PDH))
    if wh:
        bsl_levels.append((float(wh), SWEEP_WH))
    if swing_high:
        bsl_levels.append((float(swing_high), SWEEP_BSL))

    # Levels to check for SSL sweeps (price breaks below, rejects)
    ssl_levels: list[tuple[float, str]] = []
    if pdl:
        ssl_levels.append((float(pdl), SWEEP_PDL))
    if wl:
        ssl_levels.append((float(wl), SWEEP_WL))
    if swing_low:
        ssl_levels.append((float(swing_low), SWEEP_SSL))

    if not bsl_levels and not ssl_levels:
        logger.debug("[Sweep] No liquidity levels available to check.")
        return empty

    # ── Scan last `lookback` candles ───────────────────────────────────────
    detected_sweeps: list[SweepEvent] = []
    scan_df = df.tail(lookback)

    for i, (_, candle) in enumerate(scan_df.iterrows()):
        candles_ago = lookback - 1 - i   # 0 = most recent

        for level, stype in bsl_levels:
            evt = _check_bsl_sweep(candle, level, atr, stype, candles_ago)
            if evt is not None:
                detected_sweeps.append(evt)
                logger.debug(
                    "[Sweep] %s | level=%.5f | str=%.1f | %d candles ago",
                    stype, level, evt.sweep_strength, candles_ago,
                )

        for level, stype in ssl_levels:
            evt = _check_ssl_sweep(candle, level, atr, stype, candles_ago)
            if evt is not None:
                detected_sweeps.append(evt)
                logger.debug(
                    "[Sweep] %s | level=%.5f | str=%.1f | %d candles ago",
                    stype, level, evt.sweep_strength, candles_ago,
                )

    if not detected_sweeps:
        return empty

    # ── Aggregate results ──────────────────────────────────────────────────
    # Recency weighting: candles_ago=0 gets weight 1.0, each +1 reduces by 0.15
    call_weight = 0.0
    put_weight  = 0.0
    max_strength = 0.0
    has_strong   = False

    for evt in detected_sweeps:
        recency_weight = max(0.2, 1.0 - evt.candles_ago * 0.15)
        weighted_str   = evt.sweep_strength * recency_weight
        if evt.direction == "CALL":
            call_weight += weighted_str
        else:
            put_weight  += weighted_str

        if evt.sweep_strength > max_strength:
            max_strength = evt.sweep_strength
        if evt.is_strong:
            has_strong = True

    # Dominant direction
    if call_weight > put_weight:
        dominant_dir = "CALL"
    elif put_weight > call_weight:
        dominant_dir = "PUT"
    else:
        dominant_dir = "NEUTRAL"

    # Composite confidence: weighted average of sweep strengths (recency-weighted)
    total_weight = call_weight + put_weight
    dominant_weight = max(call_weight, put_weight)
    sweep_confidence = min(100.0, dominant_weight * 0.75 + max_strength * 0.25)

    # Human-readable summary
    strong_sweeps = [e for e in detected_sweeps if e.is_strong]
    all_types     = [e.sweep_type for e in detected_sweeps]
    summary_parts = []
    if strong_sweeps:
        summary_parts.append(f"STRONG: {', '.join(e.short_str() for e in strong_sweeps[:2])}")
    elif detected_sweeps:
        summary_parts.append(f"{', '.join(e.short_str() for e in detected_sweeps[:2])}")
    sweep_summary = " | ".join(summary_parts) if summary_parts else "No sweeps"

    result = LiquiditySweepResult(
        detected           = True,
        sweeps             = detected_sweeps,
        dominant_direction = dominant_dir,
        sweep_confidence   = round(sweep_confidence, 2),
        has_strong_sweep   = has_strong,
        sweep_summary      = sweep_summary,
    )

    logger.info(
        "[Sweep] Detected %d sweep(s) | dominant=%s | confidence=%.1f | strong=%s | %s",
        len(detected_sweeps), dominant_dir, sweep_confidence, has_strong, sweep_summary,
    )

    return result
