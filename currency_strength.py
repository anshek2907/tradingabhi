"""
currency_strength.py
────────────────────
Currency Strength Engine for EUR/USD trading.

Computes the relative strength of EUR vs USD using a two-tier approach:

Tier 1 — Always Available (df-only, zero API calls)
  Derives EUR/USD relative strength entirely from the existing EURUSD
  OHLCV dataframe using:
    • Rate-of-Change (5-period and 20-period)
    • RSI position (>50 = EUR trending stronger vs USD)
    • EMA50/EMA200 alignment score

Tier 2 — Enhanced Basket (optional, TwelveData API)
  Fetches 3 additional pairs (GBPUSD, USDJPY, USDCHF) with 50-bar
  history to triangulate EUR and USD strength against GBP, JPY, CHF.
  Cached for 30 minutes. Falls back to Tier 1 silently if API
  is unavailable, returns an error, or times out.

Bias Rules
──────────
  EUR strong + USD weak → bias = "CALL"  (buy EUR/USD)
  EUR weak + USD strong → bias = "PUT"   (sell EUR/USD)
  Near-equal            → bias = "NEUTRAL"

  threshold: |EUR_strength - USD_strength| > 8 points to declare bias

Confirmation-Only Principle
────────────────────────────
  Currency strength is a CONFIRMATION signal. It modifies the probability
  score (via regime_mult in ProbabilityEngine) but NEVER generates a
  standalone signal. The engine must be combined with pattern, structure,
  momentum, and regime analysis before any trade is taken.

Usage
─────
  from currency_strength import currency_strength_engine
  result = currency_strength_engine.compute(df_eurusd, api_key="YOUR_KEY")
  print(result.bias, result.eur_strength, result.usd_strength)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from logger import logger

# ── Tunable constants ──────────────────────────────────────────────────────
BIAS_THRESHOLD      = 8.0    # min |EUR - USD| to declare a non-neutral bias
STRONG_TIER_MIN     = 65.0   # bias_confidence >= 65 → STRONG
MODERATE_TIER_MIN   = 35.0   # bias_confidence >= 35 → MODERATE
CACHE_TTL_MINUTES   = 30     # Tier 2 basket cache lifetime
BASKET_FETCH_BARS   = 50     # bars to fetch per basket pair (lightweight)
BASKET_TIMEOUT_SEC  = 5      # per-pair request timeout (fast fail)

# Additional pairs for Tier 2 basket
_BASKET_PAIRS = ["GBPUSD", "USDJPY", "USDCHF"]

# Direction mapping for each basket pair relative to EUR and USD
# (sign: +1 = pair rises when currency is strong)
_EUR_SIGN = {
    "EURUSD": +1.0,   # EURUSD up → EUR strong
    "EURGBP": +1.0,   # computed: EURUSD / GBPUSD
    "EURJPY": +1.0,   # computed: EURUSD * USDJPY
    "EURCHF": +1.0,   # computed: EURUSD * USDCHF
}
_USD_SIGN = {
    "USDJPY": +1.0,   # USDJPY up → USD strong
    "USDCHF": +1.0,   # USDCHF up → USD strong
    "EURUSD": -1.0,   # EURUSD up → USD weak
    "GBPUSD": -1.0,   # GBPUSD up → USD weak
}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class CurrencyStrengthResult:
    """Result of currency strength computation."""
    eur_strength:      float = 50.0      # 0–100
    usd_strength:      float = 50.0      # 0–100
    relative_strength: float = 0.0       # EUR_strength − USD_strength
    bias:              str   = "NEUTRAL" # "CALL" | "PUT" | "NEUTRAL"
    bias_confidence:   float = 0.0       # 0–100 confidence in the bias
    tier:              str   = "WEAK"    # "STRONG" | "MODERATE" | "WEAK"
    method:            str   = "simple"  # "basket" | "simple"
    summary:           str   = ""

    def as_dict(self) -> dict:
        return {
            "eur_strength":      round(self.eur_strength, 1),
            "usd_strength":      round(self.usd_strength, 1),
            "relative_strength": round(self.relative_strength, 1),
            "bias":              self.bias,
            "bias_confidence":   round(self.bias_confidence, 1),
            "tier":              self.tier,
            "method":            self.method,
        }

    def format_log(self) -> str:
        arrow = "↑" if self.bias == "CALL" else "↓" if self.bias == "PUT" else "→"
        return (
            f"EUR={self.eur_strength:.1f} USD={self.usd_strength:.1f} "
            f"{arrow} bias={self.bias} conf={self.bias_confidence:.1f}% "
            f"[{self.tier}/{self.method}]"
        )


# ── Internal helpers ───────────────────────────────────────────────────────

def _roc(series: pd.Series, period: int) -> float:
    """Rate of change over `period` bars, returned as a percentage."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < period + 1:
        return 0.0
    latest = float(s.iloc[-1])
    past   = float(s.iloc[-period - 1])
    if past == 0:
        return 0.0
    return ((latest - past) / past) * 100.0


def _normalize_roc(roc_pct: float, clamp: float = 1.0) -> float:
    """
    Map ROC percentage to 0–100 score.
    ±clamp% → maps to 0/100; 0% → maps to 50.
    """
    clamped = max(-clamp, min(clamp, roc_pct))
    return 50.0 + (clamped / clamp) * 50.0


def _rsi_score(df: pd.DataFrame) -> float:
    """Return current RSI (or compute 14-period if not present). Range 0–100."""
    try:
        if "RSI" in df.columns:
            val = float(pd.to_numeric(df["RSI"], errors="coerce").iloc[-1])
            if not np.isnan(val):
                return val
        # Compute RSI manually
        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 15:
            return 50.0
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, 1e-10)
        rsi    = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return 50.0


def _ema_score(df: pd.DataFrame, direction: str = "CALL") -> float:
    """
    Return 0–100 score based on EMA50/EMA200 alignment and close position.
    100 = fully bullish, 0 = fully bearish.
    """
    try:
        last  = df.iloc[-1]
        close = float(pd.to_numeric(last["Close"], errors="coerce"))

        ema50_col  = "EMA50"  if "EMA50"  in df.columns else None
        ema200_col = "EMA200" if "EMA200" in df.columns else None

        if ema50_col and ema200_col:
            ema50  = float(pd.to_numeric(last[ema50_col],  errors="coerce"))
            ema200 = float(pd.to_numeric(last[ema200_col], errors="coerce"))
        else:
            # Compute on the fly
            close_s = pd.to_numeric(df["Close"], errors="coerce")
            ema50   = float(close_s.ewm(span=50,  adjust=False).mean().iloc[-1])
            ema200  = float(close_s.ewm(span=200, adjust=False).mean().iloc[-1])

        score = 50.0
        if ema50 > ema200:
            score += 30.0   # bullish EMA stack
        else:
            score -= 30.0   # bearish EMA stack
        if close > ema50:
            score += 20.0   # price above fast EMA
        else:
            score -= 20.0

        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


def _tier(confidence: float) -> str:
    if confidence >= STRONG_TIER_MIN:
        return "STRONG"
    if confidence >= MODERATE_TIER_MIN:
        return "MODERATE"
    return "WEAK"


def _bias_and_confidence(relative: float) -> tuple[str, float]:
    """Convert relative strength (EUR − USD) to bias and confidence."""
    abs_rel = abs(relative)
    if abs_rel < BIAS_THRESHOLD:
        return "NEUTRAL", 0.0
    confidence = min(100.0, abs_rel * 2.5)
    bias = "CALL" if relative > 0 else "PUT"
    return bias, confidence


# ── Tier 1: Simple (df-only) computation ──────────────────────────────────

def _compute_simple(df: pd.DataFrame) -> CurrencyStrengthResult:
    """
    Derive EUR vs USD relative strength from the EURUSD dataframe alone.

    EUR Strength proxy (EURUSD going up = EUR strong vs USD):
      • 50% from ROC (5-period + 20-period averaged, normalized)
      • 30% from RSI  (RSI > 50 = bullish EUR)
      • 20% from EMA alignment

    USD Strength = inverse of the same signals (since it's a single pair,
    USD strength is the mirror).  A small differentiation term uses
    20-period ROC vs 5-period ROC to capture divergence.
    """
    if df is None or len(df) < 25:
        return CurrencyStrengthResult(method="simple", summary="Insufficient data")

    # ROC scores (normalized 0-100; 50 = neutral)
    roc5  = _roc(df["Close"], 5)
    roc20 = _roc(df["Close"], 20)
    roc5_score  = _normalize_roc(roc5,  clamp=0.30)
    roc20_score = _normalize_roc(roc20, clamp=0.60)
    roc_score   = (roc5_score * 0.55 + roc20_score * 0.45)

    # RSI score: direct for EUR, inverted for USD
    rsi     = _rsi_score(df)
    rsi_eur = rsi
    rsi_usd = 100.0 - rsi

    # EMA score
    ema_eur = _ema_score(df, "CALL")
    ema_usd = 100.0 - ema_eur

    # Composite strength scores
    eur_strength = (roc_score * 0.50 + rsi_eur * 0.30 + ema_eur * 0.20)
    usd_strength = ((100.0 - roc_score) * 0.50 + rsi_usd * 0.30 + ema_usd * 0.20)

    eur_strength = round(max(0.0, min(100.0, eur_strength)), 2)
    usd_strength = round(max(0.0, min(100.0, usd_strength)), 2)
    relative     = round(eur_strength - usd_strength, 2)
    bias, conf   = _bias_and_confidence(relative)

    summary = (
        f"EUR={eur_strength:.1f} USD={usd_strength:.1f} rel={relative:+.1f} "
        f"bias={bias} conf={conf:.1f}%"
    )

    return CurrencyStrengthResult(
        eur_strength      = eur_strength,
        usd_strength      = usd_strength,
        relative_strength = relative,
        bias              = bias,
        bias_confidence   = round(conf, 2),
        tier              = _tier(conf),
        method            = "simple",
        summary           = summary,
    )


# ── Tier 2: Basket computation ─────────────────────────────────────────────

# Module-level cache: (result, fetched_at)
_basket_cache: tuple[CurrencyStrengthResult, datetime] | None = None


def _fetch_pair_df(symbol: str, api_key: str) -> pd.DataFrame | None:
    """
    Fetch the last BASKET_FETCH_BARS bars for `symbol` from TwelveData.
    Returns None on any error. Uses a tight timeout to fail fast.
    """
    try:
        import requests
        url    = "https://api.twelvedata.com/time_series"
        params = {
            "symbol":     symbol,
            "interval":   "5min",
            "outputsize": BASKET_FETCH_BARS,
            "apikey":     api_key,
        }
        res = requests.get(url, params=params, timeout=BASKET_TIMEOUT_SEC).json()
        if "values" not in res:
            logger.debug("[CurrStr] Basket fetch %s: API error %s", symbol, res.get("message",""))
            return None
        df = pd.DataFrame(res["values"])
        df["Close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        return df
    except Exception as e:
        logger.debug("[CurrStr] Basket fetch %s failed: %s", symbol, e)
        return None


def _pair_roc_score(df: pd.DataFrame | None, pair_sign: float, period: int = 5) -> float | None:
    """
    Compute normalized ROC score for a pair. Returns None if data unavailable.
    pair_sign: +1 if pair rising = currency strong, -1 if pair rising = currency weak.
    """
    if df is None or len(df) < period + 2:
        return None
    roc = _roc(df["Close"], period) * pair_sign
    return _normalize_roc(roc, clamp=0.25)


def _compute_basket(
    df_eurusd: pd.DataFrame,
    api_key:   str,
) -> CurrencyStrengthResult:
    """
    Tier 2: Fetch GBPUSD, USDJPY, USDCHF and build a mini-basket.

    EUR strength sources:
      EURUSD   (+): primary
      EURGBP   (+): EURUSD / GBPUSD (derived)
      EURJPY   (+): EURUSD * USDJPY (derived)
      EURCHF   (+): EURUSD * USDCHF (derived)

    USD strength sources:
      EURUSD   (−): inverse
      GBPUSD   (−): inverse
      USDJPY   (+): direct
      USDCHF   (+): direct

    Weights: EURUSD has 40% weight; others split remaining 60% equally.
    """
    global _basket_cache

    # Check cache freshness
    if _basket_cache is not None:
        cached_result, cached_at = _basket_cache
        age_min = (datetime.utcnow() - cached_at).total_seconds() / 60
        if age_min < CACHE_TTL_MINUTES:
            logger.debug("[CurrStr] Using basket cache (age=%.1fmin)", age_min)
            return cached_result

    logger.debug("[CurrStr] Fetching basket pairs: %s", _BASKET_PAIRS)

    # Fetch additional pairs
    df_gbpusd = _fetch_pair_df("GBP/USD", api_key)
    df_usdjpy = _fetch_pair_df("USD/JPY", api_key)
    df_usdchf = _fetch_pair_df("USD/CHF", api_key)

    # If all basket fetches failed, fall back to simple
    if df_gbpusd is None and df_usdjpy is None and df_usdchf is None:
        logger.warning("[CurrStr] All basket fetches failed — falling back to simple")
        return _compute_simple(df_eurusd)

    # ── EUR Strength basket ────────────────────────────────────────────────
    eur_scores: list[tuple[float, float]] = []  # (score, weight)

    # EURUSD (40% weight) — primary
    eur_roc_eu = _pair_roc_score(df_eurusd, +1.0, 5)
    if eur_roc_eu is not None:
        eur_scores.append((eur_roc_eu, 0.40))

    # EURGBP derived: EURUSD / GBPUSD — if EURUSD up more than GBPUSD → EURGBP up
    if df_gbpusd is not None and len(df_eurusd) >= 7 and len(df_gbpusd) >= 7:
        try:
            n = min(len(df_eurusd), len(df_gbpusd), 6)
            eu_close  = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-1])
            eu_past   = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-n])
            gb_close  = float(pd.to_numeric(df_gbpusd["Close"], errors="coerce").iloc[-1])
            gb_past   = float(pd.to_numeric(df_gbpusd["Close"], errors="coerce").iloc[-n])
            if eu_past > 0 and gb_past > 0 and gb_close > 0:
                eurgbp_now  = eu_close / gb_close
                eurgbp_past = eu_past  / gb_past
                roc = ((eurgbp_now - eurgbp_past) / eurgbp_past) * 100.0
                eur_scores.append((_normalize_roc(roc, 0.20), 0.20))
        except Exception:
            pass

    # EURJPY derived: EURUSD * USDJPY
    if df_usdjpy is not None and len(df_eurusd) >= 7 and len(df_usdjpy) >= 7:
        try:
            n = min(len(df_eurusd), len(df_usdjpy), 6)
            eu_close  = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-1])
            eu_past   = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-n])
            jy_close  = float(pd.to_numeric(df_usdjpy["Close"], errors="coerce").iloc[-1])
            jy_past   = float(pd.to_numeric(df_usdjpy["Close"], errors="coerce").iloc[-n])
            if eu_past > 0 and jy_past > 0:
                eurjpy_now  = eu_close * jy_close
                eurjpy_past = eu_past  * jy_past
                roc = ((eurjpy_now - eurjpy_past) / eurjpy_past) * 100.0
                eur_scores.append((_normalize_roc(roc, 0.30), 0.20))
        except Exception:
            pass

    # EURCHF derived: EURUSD * USDCHF
    if df_usdchf is not None and len(df_eurusd) >= 7 and len(df_usdchf) >= 7:
        try:
            n = min(len(df_eurusd), len(df_usdchf), 6)
            eu_close  = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-1])
            eu_past   = float(pd.to_numeric(df_eurusd["Close"], errors="coerce").iloc[-n])
            cf_close  = float(pd.to_numeric(df_usdchf["Close"], errors="coerce").iloc[-1])
            cf_past   = float(pd.to_numeric(df_usdchf["Close"], errors="coerce").iloc[-n])
            if eu_past > 0 and cf_past > 0:
                eurchf_now  = eu_close * cf_close
                eurchf_past = eu_past  * cf_past
                roc = ((eurchf_now - eurchf_past) / eurchf_past) * 100.0
                eur_scores.append((_normalize_roc(roc, 0.25), 0.20))
        except Exception:
            pass

    # ── USD Strength basket ────────────────────────────────────────────────
    usd_scores: list[tuple[float, float]] = []

    # EURUSD inverse (30% weight)
    if eur_roc_eu is not None:
        usd_scores.append((100.0 - eur_roc_eu, 0.30))

    # GBPUSD inverse (20% weight) — USD weak when GBP/USD rises
    gbp_roc = _pair_roc_score(df_gbpusd, -1.0, 5)
    if gbp_roc is not None:
        usd_scores.append((gbp_roc, 0.20))

    # USDJPY direct (25% weight)
    jpy_roc = _pair_roc_score(df_usdjpy, +1.0, 5)
    if jpy_roc is not None:
        usd_scores.append((jpy_roc, 0.25))

    # USDCHF direct (25% weight)
    chf_roc = _pair_roc_score(df_usdchf, +1.0, 5)
    if chf_roc is not None:
        usd_scores.append((chf_roc, 0.25))

    # ── Weighted averages ──────────────────────────────────────────────────
    def _weighted_avg(scores_weights: list) -> float:
        if not scores_weights:
            return 50.0
        total_w = sum(w for _, w in scores_weights)
        if total_w <= 0:
            return 50.0
        return sum(s * w for s, w in scores_weights) / total_w

    roc_eur = _weighted_avg(eur_scores)
    roc_usd = _weighted_avg(usd_scores)

    # Blend ROC with RSI and EMA from df_eurusd (still useful even in basket mode)
    rsi     = _rsi_score(df_eurusd)
    ema_eur = _ema_score(df_eurusd)
    ema_usd = 100.0 - ema_eur

    eur_strength = roc_eur * 0.60 + rsi * 0.25 + ema_eur * 0.15
    usd_strength = roc_usd * 0.60 + (100.0 - rsi) * 0.25 + ema_usd * 0.15

    eur_strength = round(max(0.0, min(100.0, eur_strength)), 2)
    usd_strength = round(max(0.0, min(100.0, usd_strength)), 2)
    relative     = round(eur_strength - usd_strength, 2)
    bias, conf   = _bias_and_confidence(relative)

    pairs_used = sum([
        1,
        1 if df_gbpusd is not None else 0,
        1 if df_usdjpy is not None else 0,
        1 if df_usdchf is not None else 0,
    ])
    summary = (
        f"EUR={eur_strength:.1f} USD={usd_strength:.1f} rel={relative:+.1f} "
        f"bias={bias} conf={conf:.1f}% [basket/{pairs_used} pairs]"
    )

    result = CurrencyStrengthResult(
        eur_strength      = eur_strength,
        usd_strength      = usd_strength,
        relative_strength = relative,
        bias              = bias,
        bias_confidence   = round(conf, 2),
        tier              = _tier(conf),
        method            = "basket",
        summary           = summary,
    )

    _basket_cache = (result, datetime.utcnow())
    return result


# ── Main Engine class ──────────────────────────────────────────────────────

class CurrencyStrengthEngine:
    """
    Singleton currency strength engine.

    Call compute() once per signal generation pass; it automatically
    selects Tier 1 (simple) or Tier 2 (basket) based on api_key availability.
    Results are cached for CACHE_TTL_MINUTES to avoid redundant API calls.
    """

    def __init__(self):
        self._simple_cache: tuple[CurrencyStrengthResult, datetime] | None = None

    def compute(
        self,
        df:      pd.DataFrame,
        api_key: Optional[str] = None,
    ) -> CurrencyStrengthResult:
        """
        Compute EUR and USD strength and return a CurrencyStrengthResult.

        Args:
            df:      Enriched EURUSD OHLCV DataFrame (any timeframe, ≥25 bars).
            api_key: TwelveData API key. If None or empty, falls back to Tier 1.

        Returns:
            CurrencyStrengthResult — always succeeds (worst case: NEUTRAL/WEAK).
        """
        if df is None or len(df) < 25:
            logger.debug("[CurrStr] Insufficient df — returning NEUTRAL")
            return CurrencyStrengthResult(
                method="simple", summary="Insufficient data", tier="WEAK"
            )

        # ── Tier 2: basket (if api_key provided) ──────────────────────────
        if api_key:
            try:
                result = _compute_basket(df, api_key)
                logger.info(
                    "[CurrStr] %s | %s",
                    result.format_log(), result.summary,
                )
                return result
            except Exception as e:
                logger.warning("[CurrStr] Basket failed (%s) — using simple", e)

        # ── Tier 1: simple (df only) ───────────────────────────────────────
        # Check simple cache (30 min TTL)
        if self._simple_cache is not None:
            cached, cached_at = self._simple_cache
            age_min = (datetime.utcnow() - cached_at).total_seconds() / 60
            if age_min < CACHE_TTL_MINUTES:
                logger.debug("[CurrStr] Simple cache hit (age=%.1fmin)", age_min)
                return cached

        result = _compute_simple(df)
        self._simple_cache = (result, datetime.utcnow())
        logger.info("[CurrStr] %s", result.format_log())
        return result

    def invalidate_cache(self) -> None:
        """Force cache expiry (e.g., call at the start of a new generation run)."""
        global _basket_cache
        self._simple_cache = None
        _basket_cache = None
        logger.debug("[CurrStr] Cache invalidated")


# ── Global singleton ───────────────────────────────────────────────────────
currency_strength_engine = CurrencyStrengthEngine()
