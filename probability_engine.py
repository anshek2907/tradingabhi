"""
probability_engine.py
─────────────────────
Centralized Weighted Probability Scoring Engine.

This is the single intelligence authority that assigns ONE final score
(0–100) to every candidate signal before it is accepted, ranked, or
published.  All filtering, ranking, and selection in signal_generator.py
routes through this module.

Score formula (v2 — balanced safe-profit)
──────────────────────────────────────────
probability_score =
    (win_rate                  * 0.32)   ← increased: historical track record is king
  + (direction_consistency     * 0.25)   ← unchanged: directional bias strength
  + (momentum_continuation     * 0.15)   ← increased: continuation quality
  + (atr_quality               * 0.12)   ← reduced: less aggressive ATR gating
  + (session_strength          * 0.08)   ← reduced: session weighting relaxed
  + (volatility_quality        * 0.08)   ← reduced: less aggressive vol filtering
  - (reversal_risk             * 0.06)   ← reduced penalty: avoid over-rejection

  × regime_multiplier           ← applied after linear sum

Then clamped to [0, 100].

Dynamic minimum score thresholds (regime-adaptive)
───────────────────────────────────────────────────
  TRENDING        : min_score = 70  (allow quality continuation signals)
  MODERATE/MIXED  : min_score = 72
  SIDEWAYS        : min_score = 78  (stricter — avoid choppy noise)
  HIGH_VOLATILITY : min_score = 80  (strong signals only)
  REVERSAL_HEAVY  : min_score = 80  (maximum caution)

Signal tiers
────────────
  STRONG_SIGNAL   : score >= 80
  MODERATE_SIGNAL : score >= 70  (regime-specific floor applied at gate)
  SKIP            : score <  70

Regime multipliers
──────────────────
  TRENDING        : 1.05  (boost continuation signals)
  SIDEWAYS        : 0.93  (slight reduction — not over-punished)
  HIGH_VOLATILITY : 0.90  (reduce unstable setups)
  REVERSAL_HEAVY  : 0.87  (require stronger confirmation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from logger import logger

# ── Signal tier labels ─────────────────────────────────────
TIER_STRONG   = "STRONG_SIGNAL"
TIER_MODERATE = "MODERATE_SIGNAL"
TIER_SKIP     = "SKIP"

TIER_STRONG_MIN   = 80.0
TIER_MODERATE_MIN = 70.0

# ── Dynamic minimum score thresholds per regime ────────────
# These are applied as the SKIP floor in is_acceptable_for_regime().
# STRONG/MODERATE tiers are determined by TIER_STRONG_MIN/TIER_MODERATE_MIN;
# however, signals in MODERATE range below the regime floor are rejected.
_REGIME_MIN_SCORE: dict[str, float] = {
    "TRENDING":        70.0,   # good quality continuation allowed
    "MODERATE":        72.0,   # default moderate market
    "SIDEWAYS":        78.0,   # stricter — avoid choppy noise
    "HIGH_VOLATILITY": 80.0,   # strong signals only
    "REVERSAL_HEAVY":  80.0,   # maximum caution
}
_DEFAULT_MIN_SCORE = 72.0

# ── Regime multipliers ─────────────────────────────────────
_REGIME_MULTIPLIER: dict[str, float] = {
    "TRENDING":        1.05,   # continuation bias → boost
    "SIDEWAYS":        0.93,   # slight reduction (was 0.92, softened)
    "HIGH_VOLATILITY": 0.90,   # unstable → reduce (was 0.88, softened)
    "REVERSAL_HEAVY":  0.87,   # strong confirmation required (was 0.85, softened)
}
_DEFAULT_MULTIPLIER = 1.00

# ── Weight table (v2 — balanced safe-profit) ───────────────
# Positive weights sum to 1.00; reversal_risk is subtracted (reduced penalty).
# Key changes from v1:
#   win_rate: 0.30 → 0.32  (historical track record prioritized)
#   momentum_strength: 0.10 → 0.15  (momentum continuation matters more)
#   atr_quality: 0.15 → 0.12  (less aggressive ATR gating)
#   session_strength: 0.10 → 0.08  (relaxed session weight)
#   volatility_quality: 0.10 → 0.08  (less aggressive vol filtering)
#   reversal_risk: 0.10 → 0.06  (reduced penalty to avoid over-rejection)
_WEIGHTS = {
    "win_rate":              0.32,   # ↑ historical win rate is king
    "direction_consistency": 0.25,   # unchanged — directional bias strength
    "atr_quality":           0.12,   # ↓ reduced ATR strictness
    "momentum_strength":     0.15,   # ↑ momentum continuation priority
    "session_strength":      0.08,   # ↓ relaxed session weighting
    "volatility_quality":    0.08,   # ↓ less aggressive vol filtering
    "reversal_risk":         0.06,   # ↓ reduced penalty (avoid over-rejection)
}


@dataclass
class ProbabilityInputs:
    """
    All raw sub-metrics needed to compute the probability score.
    All values are expected in the range [0, 100] unless noted.
    """
    # Core metrics
    win_rate:              float = 50.0   # historical win rate %
    direction_consistency: float = 50.0   # bullish/bearish consistency %
    atr_quality:           float = 50.0   # ATR activity × stability score
    momentum_strength:     float = 50.0   # momentum continuation composite
    session_strength:      float = 78.0   # trading session quality
    volatility_quality:    float = 50.0   # healthy volatility cluster score
    reversal_risk:         float = 50.0   # reversal frequency (higher = worse)

    # Regime context
    regime:             str   = "SIDEWAYS"
    regime_confidence:  float = 50.0      # 0-100 confidence in detected regime

    # Extra context (not used in formula, but stored for logging)
    volatility_zone:    str   = "healthy"
    time_str:           str   = ""
    direction:          str   = ""


@dataclass
class ProbabilityResult:
    """
    Immutable result returned by ProbabilityEngine.compute().
    """
    # Sub-score breakdown (each is the weighted contribution)
    raw_linear_score:  float = 0.0   # pre-regime, pre-clamp
    probability_score: float = 0.0   # final clamped score [0-100]
    signal_tier:       str   = TIER_SKIP
    regime_multiplier: float = 1.00

    # Individual weighted contributions (for logging/debugging)
    contrib_win_rate:              float = 0.0
    contrib_direction_consistency: float = 0.0
    contrib_atr_quality:           float = 0.0
    contrib_momentum:              float = 0.0
    contrib_session:               float = 0.0
    contrib_volatility:            float = 0.0
    contrib_reversal_penalty:      float = 0.0

    def as_dict(self) -> dict:
        return {
            "probability_score":            round(self.probability_score, 2),
            "signal_tier":                  self.signal_tier,
            "regime_multiplier":            round(self.regime_multiplier, 3),
            "raw_linear_score":             round(self.raw_linear_score, 2),
            "contrib_win_rate":             round(self.contrib_win_rate, 2),
            "contrib_direction_consistency": round(self.contrib_direction_consistency, 2),
            "contrib_atr_quality":          round(self.contrib_atr_quality, 2),
            "contrib_momentum":             round(self.contrib_momentum, 2),
            "contrib_session":              round(self.contrib_session, 2),
            "contrib_volatility":           round(self.contrib_volatility, 2),
            "contrib_reversal_penalty":     round(self.contrib_reversal_penalty, 2),
        }


def _classify_tier(score: float) -> str:
    if score >= TIER_STRONG_MIN:
        return TIER_STRONG
    if score >= TIER_MODERATE_MIN:
        return TIER_MODERATE
    return TIER_SKIP


def _regime_min_score(regime: str) -> float:
    """Return the dynamic minimum acceptable score for the given regime."""
    return _REGIME_MIN_SCORE.get(regime, _DEFAULT_MIN_SCORE)


class ProbabilityEngine:
    """
    Singleton-style scoring engine.  Instantiate once and call compute()
    for each candidate slot.

    Usage:
        from probability_engine import probability_engine, TIER_STRONG, TIER_MODERATE
        result = probability_engine.compute(inputs)
        if result.signal_tier == TIER_SKIP:
            continue
        signal["probability_score"] = result.probability_score
        signal["signal_tier"]       = result.signal_tier
    """

    # ── Public API ─────────────────────────────────────────

    @staticmethod
    def compute(inputs: ProbabilityInputs) -> ProbabilityResult:
        """
        Compute the centralized probability score for one candidate slot.

        Steps:
          1. Compute weighted linear sum (all positive terms + reversal penalty)
          2. Apply quality micro-penalties for structural weaknesses
          3. Apply regime multiplier (boost or reduce based on market type)
          4. Clamp to [0, 100]
          5. Classify into STRONG / MODERATE / SKIP tier
        """
        # ── Step 1: Weighted linear components ────────────
        # v2 weights: win_rate↑, momentum↑, reversal penalty↓, ATR/session/vol↓
        contrib_win_rate  = inputs.win_rate              * _WEIGHTS["win_rate"]
        contrib_dir_cons  = inputs.direction_consistency * _WEIGHTS["direction_consistency"]
        contrib_atr       = inputs.atr_quality           * _WEIGHTS["atr_quality"]
        contrib_momentum  = inputs.momentum_strength     * _WEIGHTS["momentum_strength"]
        contrib_session   = inputs.session_strength      * _WEIGHTS["session_strength"]
        contrib_vol       = inputs.volatility_quality    * _WEIGHTS["volatility_quality"]
        contrib_reversal  = inputs.reversal_risk         * _WEIGHTS["reversal_risk"]   # penalty

        linear = (
            contrib_win_rate
            + contrib_dir_cons
            + contrib_atr
            + contrib_momentum
            + contrib_session
            + contrib_vol
            - contrib_reversal   # subtracted (reduced weight vs v1)
        )

        # ── Step 2: Structural quality micro-penalties (v2 — softened) ────
        # Penalties are intentionally lighter to avoid over-rejection of valid signals.
        penalties = 0.0

        # Weak historical win rate (softened thresholds)
        if inputs.win_rate < 55.0:      # was 58.0 — only penalize genuinely weak
            penalties += 10.0           # was 14.0 — reduced penalty
        elif inputs.win_rate < 60.0:    # was 62.0
            penalties += 4.0            # was 6.0

        # High reversal risk (reduced penalty — avoid over-filtering valid signals)
        if inputs.reversal_risk > 55.0:   # was 48.0 — only penalize extreme reversals
            penalties += 5.0              # was 8.0
        elif inputs.reversal_risk > 47.0: # was 40.0
            penalties += 2.0              # was 3.0

        # Dead/noisy volatility zone — these are pre-gated before scoring
        # but we apply a small residual penalty for boundary cases
        if inputs.volatility_zone == "dead":
            penalties += 8.0            # was 12.0
        elif inputs.volatility_zone == "noisy":
            penalties += 5.0            # was 8.0
        elif inputs.volatility_zone == "unstable_spike":
            penalties += 10.0           # was 15.0

        # Low volatility quality (softened threshold)
        if inputs.volatility_quality < 48.0:   # was 55.0 — raised bar for penalty
            penalties += 6.0                   # was 10.0

        # ── Regime-confidence scaling on penalties ────────
        # Scale: 0.65 – 1.15 (was 0.75 – 1.25 → narrowed to soften impact)
        regime_conf_factor = 0.65 + (inputs.regime_confidence / 100.0) * 0.50
        penalties *= regime_conf_factor

        raw_pre_regime = max(0.0, linear - penalties)

        # ── Step 3: Regime multiplier ─────────────────────
        regime_mult = _REGIME_MULTIPLIER.get(inputs.regime, _DEFAULT_MULTIPLIER)

        # Regime-specific extra logic (softer adjustments vs v1)
        if inputs.regime == "TRENDING":
            # Boost signals with strong momentum + good win rate in trending market
            if inputs.momentum_strength >= 68.0 and inputs.win_rate >= 63.0:
                regime_mult = min(1.12, regime_mult + 0.05)  # was +0.04, now +0.05
            # Bonus for strong direction consistency (recurring timing)
            if inputs.direction_consistency >= 72.0:
                regime_mult = min(1.12, regime_mult + 0.02)
        elif inputs.regime == "SIDEWAYS":
            # Extra penalty only for very low direction consistency in sideways
            if inputs.direction_consistency < 60.0:   # was 65.0 → more tolerant
                regime_mult = max(0.88, regime_mult - 0.03)  # was -0.04
        elif inputs.regime == "HIGH_VOLATILITY":
            # Extra penalty only for unstable ATR in high-vol regime
            if inputs.atr_quality < 55.0:   # was 60.0 → more tolerant
                regime_mult = max(0.83, regime_mult - 0.04)  # was -0.05
        elif inputs.regime == "REVERSAL_HEAVY":
            # Penalise only extreme reversal risk (relaxed from >35 to >42)
            if inputs.reversal_risk > 42.0:
                regime_mult = max(0.82, regime_mult - 0.03)  # was -0.05

        final_score = raw_pre_regime * regime_mult
        final_score = max(0.0, min(100.0, final_score))

        tier = _classify_tier(final_score)

        result = ProbabilityResult(
            raw_linear_score              = round(raw_pre_regime, 2),
            probability_score             = round(final_score, 2),
            signal_tier                   = tier,
            regime_multiplier             = round(regime_mult, 3),
            contrib_win_rate              = round(contrib_win_rate, 2),
            contrib_direction_consistency  = round(contrib_dir_cons, 2),
            contrib_atr_quality           = round(contrib_atr, 2),
            contrib_momentum              = round(contrib_momentum, 2),
            contrib_session               = round(contrib_session, 2),
            contrib_volatility            = round(contrib_vol, 2),
            contrib_reversal_penalty      = round(contrib_reversal, 2),
        )

        logger.debug(
            "[ProbEng] %s %s | score=%.1f tier=%s | mult=%.3f | "
            "win=%.1f dir=%.1f atr=%.1f mom=%.1f sess=%.1f vol=%.1f rev=-%.1f",
            inputs.time_str, inputs.direction,
            final_score, tier, regime_mult,
            inputs.win_rate, inputs.direction_consistency,
            inputs.atr_quality, inputs.momentum_strength,
            inputs.session_strength, inputs.volatility_quality,
            inputs.reversal_risk,
        )

        return result

    @staticmethod
    def is_acceptable(result: ProbabilityResult) -> bool:
        """Return True if the signal should be kept (not SKIPped)."""
        return result.signal_tier != TIER_SKIP

    @staticmethod
    def is_acceptable_for_regime(result: ProbabilityResult, regime: str) -> bool:
        """
        Regime-aware acceptability gate.

        Applies the dynamic minimum score floor per market regime:
          TRENDING=70, MODERATE=72, SIDEWAYS=78, HIGH_VOL/REVERSAL=80

        A signal must:
          1. Not be SKIP tier (universal gate), AND
          2. Meet or exceed the regime-specific minimum score.

        This prevents over-rejection while still protecting against
        weak signals in difficult market conditions.
        """
        if result.signal_tier == TIER_SKIP:
            return False
        min_score = _regime_min_score(regime)
        return result.probability_score >= min_score

    @staticmethod
    def score_to_confidence(probability_score: float, base_confidence: float) -> float:
        """
        Blend the probability score with the existing base_confidence so the
        final published `confidence` field reflects both the historical pattern
        strength and the live market confidence.

          blended = base_confidence * 0.55  +  probability_score * 0.45
        """
        blended = base_confidence * 0.55 + probability_score * 0.45
        return max(0.0, min(99.0, blended))

    @staticmethod
    def rank_key(signal: dict) -> tuple:
        """
        Sorting key for final signal list.
        Primary: probability_score descending
        Secondary: pattern_strength descending
        """
        return (
            signal.get("probability_score", 0.0),
            signal.get("pattern_strength", 0),
        )


# ── Global singleton ───────────────────────────────────────
probability_engine = ProbabilityEngine()
