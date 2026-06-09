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

# ── Lazy import of weight tracker (avoids circular imports) ─────────────
# strategy_weight_tracker is imported lazily inside methods so that
# probability_engine can be imported standalone without requiring the full
# tracker module to be initialised first.
_weight_tracker_ref = None

def _get_weight_tracker():
    global _weight_tracker_ref
    if _weight_tracker_ref is None:
        try:
            from strategy_weight_tracker import strategy_weight_tracker as _swt
            _weight_tracker_ref = _swt
        except Exception:
            pass
    return _weight_tracker_ref

# ── Signal tier labels ─────────────────────────────────────
TIER_STRONG   = "STRONG_SIGNAL"
TIER_MODERATE = "MODERATE_SIGNAL"
TIER_SKIP     = "SKIP"

TIER_STRONG_MIN   = 78.0
TIER_MODERATE_MIN = 68.0

# ── Dynamic minimum score thresholds per regime ────────────
# These are applied as the SKIP floor in is_acceptable_for_regime().
# STRONG/MODERATE tiers are determined by TIER_STRONG_MIN/TIER_MODERATE_MIN;
# however, signals in MODERATE range below the regime floor are rejected.
_REGIME_MIN_SCORE: dict[str, float] = {
    "TRENDING":        68.0,   # allow moderate signals
    "MODERATE":        68.0,   # allow moderate signals
    "SIDEWAYS":        68.0,   # allow moderate signals
    "HIGH_VOLATILITY": 68.0,   # allow moderate signals
    "REVERSAL_HEAVY":  78.0,   # requires strong
}
_DEFAULT_MIN_SCORE = 68.0

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
    market_structure:   Optional[any] = None  # MarketStructureResult
    # Currency Strength confirmation (confirmation-only — never standalone)
    currency_bias:            str   = "NEUTRAL"  # "CALL" | "PUT" | "NEUTRAL"
    currency_strength_score:  float = 50.0       # bias_confidence 0–100


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

    def get_breakdown_items(self) -> list:
        """
        Return a list of (label, signed_points) tuples for display.

        Points are the actual weighted contribution to the final score
        (before regime multiplier), rounded to nearest integer.
        Positive = boosts score; negative = reduces score.

        Example output:
            [("Win Rate History", 26), ("Direction Bias", 18), ...,
             ("Reversal Risk",  -4)]
        """
        items = [
            ("Win Rate History",   int(round(self.contrib_win_rate))),
            ("Direction Bias",     int(round(self.contrib_direction_consistency))),
            ("Momentum Strength",  int(round(self.contrib_momentum))),
            ("ATR Quality",        int(round(self.contrib_atr_quality))),
            ("Session Strength",   int(round(self.contrib_session))),
            ("Volatility Quality", int(round(self.contrib_volatility))),
            ("Reversal Risk",      -int(round(self.contrib_reversal_penalty))),
        ]
        return items

    def format_breakdown(self, final_score: float | None = None) -> str:
        """
        Return a Telegram-formatted confidence breakdown block.

        Example:
            ━━━━━━━━━━━━━━━━━━
            📊 Confidence Breakdown
            Win Rate History:   +26
            Direction Bias:     +18
            Momentum Strength:  +12
            ATR Quality:        +8
            Session Strength:   +6
            Volatility Quality: +5
            Reversal Risk:      -4
            ─────────────────────
            Final Score:        71
            ━━━━━━━━━━━━━━━━━━
        """
        score = final_score if final_score is not None else self.probability_score
        items = self.get_breakdown_items()
        max_label = max(len(lbl) for lbl, _ in items)
        lines = ["📊 Confidence Breakdown"]
        for label, pts in items:
            sign  = "+" if pts >= 0 else ""
            pad   = " " * (max_label - len(label) + 2)
            lines.append(f"  {label}:{pad}{sign}{pts}")
        lines.append(f"  {'─' * (max_label + 6)}")
        lines.append(f"  Final Score:{' ' * (max_label - 11 + 2)}{int(round(score))}")
        return "\n".join(lines)


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
    def compute(
        inputs: ProbabilityInputs,
        voter_weight_adjustments: Optional[dict] = None,
    ) -> ProbabilityResult:
        """
        Compute the centralized probability score for one candidate slot.

        Args:
            inputs:                  All raw sub-metrics (ProbabilityInputs).
            voter_weight_adjustments: Optional dict of metric_key → multiplier
                                     from StrategyWeightTracker.get_formula_weight_adjustments().
                                     When supplied, each formula weight is scaled
                                     by its voter-performance multiplier before scoring.
                                     This implements the Dynamic Strategy Weighting system.

        Steps:
          1. Build effective weights (baseline * voter multiplier, re-normalised)
          2. Compute weighted linear sum (all positive terms + reversal penalty)
          3. Apply quality micro-penalties for structural weaknesses
          4. Apply regime multiplier (boost or reduce based on market type)
          5. Clamp to [0, 100]
          6. Classify into STRONG / MODERATE / SKIP tier
        """
        # ── Step 0: Build effective weights (voter-adaptive) ──────────
        # Apply voter multipliers to the baseline formula weights.
        # Each metric multiplier is clamped to [0.60, 1.40] (enforced in tracker)
        # so no single voter can dominate or zero-out a component.
        # After scaling, weights are re-normalised so positive contributions
        # still sum to 1.0, preserving the overall score range.
        adj = voter_weight_adjustments or {}

        def _w(key: str) -> float:
            return _WEIGHTS[key] * float(adj.get(key, 1.0))

        raw_win_rate  = _w("win_rate")
        raw_dir_cons  = _w("direction_consistency")
        raw_atr       = _w("atr_quality")
        raw_momentum  = _w("momentum_strength")
        raw_session   = _w("session_strength")
        raw_vol       = _w("volatility_quality")
        raw_reversal  = _w("reversal_risk")

        # Positive weights (excluding reversal penalty which is subtracted)
        _pos_sum = raw_win_rate + raw_dir_cons + raw_atr + raw_momentum + raw_session + raw_vol
        _norm = (_pos_sum / (sum(_WEIGHTS[k] for k in (
            "win_rate", "direction_consistency", "atr_quality",
            "momentum_strength", "session_strength", "volatility_quality"
        )))) if _pos_sum > 0 else 1.0

        # Effective weights (normalised so formula output scale stays [0,100])
        eff_win_rate  = raw_win_rate  / _norm
        eff_dir_cons  = raw_dir_cons  / _norm
        eff_atr       = raw_atr       / _norm
        eff_momentum  = raw_momentum  / _norm
        eff_session   = raw_session   / _norm
        eff_vol       = raw_vol       / _norm
        eff_reversal  = raw_reversal  # penalty — not normalised (kept consistent)

        if adj:
            logger.debug(
                "[ProbEng] Voter adjustments applied: %s",
                {k: round(v, 3) for k, v in adj.items()},
            )

        # ── Step 1: Weighted linear components ────────────
        contrib_win_rate  = inputs.win_rate              * eff_win_rate
        contrib_dir_cons  = inputs.direction_consistency * eff_dir_cons
        contrib_atr       = inputs.atr_quality           * eff_atr
        contrib_momentum  = inputs.momentum_strength     * eff_momentum
        contrib_session   = inputs.session_strength      * eff_session
        contrib_vol       = inputs.volatility_quality    * eff_vol
        contrib_reversal  = inputs.reversal_risk         * eff_reversal   # penalty

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

        # ── Step 4: Market Structure & Liquidity Modifiers ──
        if inputs.market_structure is not None:
            ms = inputs.market_structure
            # Reduce confidence near strong opposing liquidity
            if ms.near_opposing_liquidity:
                regime_mult = max(0.80, regime_mult - 0.05)
            # Boost continuation after BOS
            if inputs.direction == ms.recent_bos:
                regime_mult = min(1.15, regime_mult + 0.05)
            # Boost reversal after CHOCH
            if inputs.direction == ms.recent_choch:
                regime_mult = min(1.15, regime_mult + 0.05)
            # Increase confidence when trend, structure, and liquidity agree
            if inputs.direction == "CALL" and ms.trend == "BULLISH" and not ms.near_opposing_liquidity:
                regime_mult = min(1.15, regime_mult + 0.03)
            elif inputs.direction == "PUT" and ms.trend == "BEARISH" and not ms.near_opposing_liquidity:
                regime_mult = min(1.15, regime_mult + 0.03)

            # ── Liquidity Sweep Weight ────────────────────────────────────
            # A confirmed sweep in the signal direction increases conviction
            # for the reversal; an opposing strong sweep reduces it.
            # Uses getattr for backward compatibility with older MarketStructureResult.
            if getattr(ms, "has_strong_sweep", False):
                if getattr(ms, "sweep_direction", None) == inputs.direction:
                    # Strong sweep confirms our reversal direction → meaningful boost
                    regime_mult = min(1.18, regime_mult + 0.06)
                    logger.debug(
                        "[ProbEng] Sweep boost +0.06 (strong %s sweep confirms %s)",
                        ms.sweep_direction, inputs.direction,
                    )
                elif getattr(ms, "sweep_direction", None) and ms.sweep_direction != inputs.direction:
                    # Strong opposing sweep → reduce confidence
                    regime_mult = max(0.78, regime_mult - 0.04)
                    logger.debug(
                        "[ProbEng] Sweep penalty -0.04 (strong opposing %s sweep vs %s)",
                        ms.sweep_direction, inputs.direction,
                    )
            elif getattr(ms, "sweep_result", None) is not None and getattr(ms.sweep_result, "detected", False):
                sweep_conf = getattr(ms, "sweep_confidence", 0.0)
                sweep_dir  = getattr(ms, "sweep_direction", None)
                if sweep_dir == inputs.direction and sweep_conf >= 50.0:
                    # Moderate sweep in our direction → small boost
                    regime_mult = min(1.15, regime_mult + 0.03)
                    logger.debug(
                        "[ProbEng] Sweep boost +0.03 (moderate %s sweep conf=%.1f)",
                        sweep_dir, sweep_conf,
                    )

        # ── Currency Strength Confirmation (confirmation-only) ───────────────────
        # Max ±0.05 on regime_mult. Aligns as extra evidence but never overrides
        # structure, momentum, or sweep signals.
        cb   = inputs.currency_bias
        cs   = inputs.currency_strength_score / 100.0  # normalised 0–1
        if cb == inputs.direction:
            # Strength confirms direction → boost proportional to confidence
            cs_boost = 0.02 + cs * 0.03          # +0.02 (weak) to +0.05 (strong)
            regime_mult = min(1.20, regime_mult + cs_boost)
            logger.debug(
                "[ProbEng] CurrStr boost +%.3f (bias=%s conf=%.1f confirms %s)",
                cs_boost, cb, inputs.currency_strength_score, inputs.direction,
            )
        elif cb != "NEUTRAL" and cb != inputs.direction:
            # Opposing currency strength → proportional small penalty
            cs_penalty = cs * 0.03               # 0 (weak) to -0.03 (strong)
            regime_mult = max(0.78, regime_mult - cs_penalty)
            logger.debug(
                "[ProbEng] CurrStr penalty -%.3f (bias=%s conf=%.1f opposes %s)",
                cs_penalty, cb, inputs.currency_strength_score, inputs.direction,
            )

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
    def compute_with_voter_weights(
        inputs: ProbabilityInputs,
    ) -> ProbabilityResult:
        """
        Convenience wrapper that automatically pulls current voter-performance
        weight adjustments from StrategyWeightTracker and feeds them into compute().

        Use this in the main signal pipeline instead of bare compute() to
        enable fully automatic Dynamic Strategy Weighting.
        """
        tracker = _get_weight_tracker()
        if tracker is not None:
            try:
                adj = tracker.get_formula_weight_adjustments()
            except Exception:
                adj = {}
        else:
            adj = {}
        return ProbabilityEngine.compute(inputs, voter_weight_adjustments=adj)

    @staticmethod
    def is_acceptable(result: ProbabilityResult) -> bool:
        """Return True if the signal should be kept (not SKIPped)."""
        return result.signal_tier != TIER_SKIP

    @staticmethod
    def is_acceptable_for_regime(result: ProbabilityResult, regime: str, min_score_override: float = None) -> bool:
        """
        Regime-aware acceptability gate.

        Applies the dynamic minimum score floor per market regime:
          TRENDING=70, MODERATE=72, SIDEWAYS=78, HIGH_VOL/REVERSAL=80

        A signal must:
          1. Not be SKIP tier (universal gate), AND
          2. Meet or exceed the regime-specific minimum score.

        If min_score_override is provided, the strict TIER_SKIP gate is bypassed
        and we solely evaluate against the overridden minimum.
        """
        min_score = min_score_override if min_score_override is not None else _regime_min_score(regime)

        if min_score_override is not None:
            return result.probability_score >= min_score

        if result.signal_tier == TIER_SKIP:
            return False
            
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


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Decay Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceDecayEngine:
    """
    Non-destructive confidence decay engine.

    Computes how much a signal's confidence should decrease based on its age
    (time elapsed since the signal's scheduled entry time).  The decay is
    applied at READ TIME — stored values are never modified, so historical
    records remain intact.

    Decay schedule (tunable):
    ┌────────────────┬─────────────────────────────────────────────────┐
    │ Age            │ Effect                                           │
    ├────────────────┼─────────────────────────────────────────────────┤
    │ 0 – 2 min      │ No decay  (freshness window — entry just opened)│
    │ 2 – 5 min      │ Gradual linear decay  (0 → MAX_MILD pts lost)   │
    │ 5 – 10 min     │ Strong linear decay   (MAX_MILD → MAX_STRONG)   │
    │ 10+ min        │ Capped at MAX_STRONG  (no further reduction)     │
    └────────────────┴─────────────────────────────────────────────────┘

    Example with defaults (MAX_MILD=8, MAX_STRONG=25):
        0 min  →  0 pts lost  (100% original)
        2 min  →  0 pts lost
        3 min  →  ~2.7 pts lost
        5 min  →  8 pts lost
        7 min  →  ~14.7 pts lost
        10 min →  25 pts lost
        15 min →  25 pts lost  (capped)

    The minimum decayed confidence floor is 20 — a signal is never zeroed.

    Usage:
        from probability_engine import confidence_decay_engine
        decayed, detail = confidence_decay_engine.apply(original_conf, age_seconds)
    """

    # ── Tunable constants ──────────────────────────────────────────────────

    # Phase 1: no decay window (seconds)
    FRESH_WINDOW_SEC    = 2 * 60     # 0 – 2 minutes

    # Phase 2: gradual decay zone end (seconds)
    MILD_END_SEC        = 5 * 60     # 2 – 5 minutes

    # Phase 3: strong decay zone end (seconds) — capped after this
    STRONG_END_SEC      = 10 * 60   # 5 – 10 minutes

    # Maximum point loss at end of gradual (mild) zone
    MAX_MILD_DECAY      = 8.0       # pts lost at 5 min

    # Maximum point loss at end of strong zone (and beyond)
    MAX_STRONG_DECAY    = 25.0      # pts lost at 10+ min

    # Absolute floor — no signal confidence goes below this
    CONFIDENCE_FLOOR    = 20.0

    # ── Core API ──────────────────────────────────────────────────────────

    @classmethod
    def apply(
        cls,
        original_confidence: float,
        age_seconds: float,
    ) -> tuple[float, dict]:
        """
        Apply time-based decay to a signal confidence value.

        Args:
            original_confidence: original score [0, 100]
            age_seconds:         seconds elapsed since signal entry time
                                 (negative = signal is in the future → no decay)

        Returns:
            (decayed_confidence, detail_dict)

        detail_dict keys:
            age_seconds          : exact age passed in
            age_minutes          : age in minutes (rounded to 2 dp)
            decay_zone           : 'fresh' | 'mild' | 'strong' | 'expired'
            points_lost          : float pts subtracted from original score
            decayed_confidence   : final score after decay + floor clamp
            tier_before          : STRONG/MODERATE/SKIP tier from original
            tier_after           : STRONG/MODERATE/SKIP tier after decay
            tier_changed         : bool — True if tier degraded
        """
        age_seconds = max(0.0, float(age_seconds))
        original    = max(0.0, min(100.0, float(original_confidence)))

        # ── Compute points lost ──────────────────────────────────────────
        if age_seconds <= cls.FRESH_WINDOW_SEC:
            # Zone 1 — fresh: no decay
            points_lost = 0.0
            zone        = "fresh"

        elif age_seconds <= cls.MILD_END_SEC:
            # Zone 2 — mild: linear ramp from 0 to MAX_MILD_DECAY
            t           = (age_seconds - cls.FRESH_WINDOW_SEC) / (cls.MILD_END_SEC - cls.FRESH_WINDOW_SEC)
            points_lost = t * cls.MAX_MILD_DECAY
            zone        = "mild"

        elif age_seconds <= cls.STRONG_END_SEC:
            # Zone 3 — strong: linear ramp from MAX_MILD to MAX_STRONG
            t           = (age_seconds - cls.MILD_END_SEC) / (cls.STRONG_END_SEC - cls.MILD_END_SEC)
            points_lost = cls.MAX_MILD_DECAY + t * (cls.MAX_STRONG_DECAY - cls.MAX_MILD_DECAY)
            zone        = "strong"

        else:
            # Zone 4 — expired: capped at MAX_STRONG
            points_lost = cls.MAX_STRONG_DECAY
            zone        = "expired"

        decayed = max(cls.CONFIDENCE_FLOOR, original - points_lost)

        # ── Tier reclassification ────────────────────────────────────────
        tier_before = _classify_tier(original)
        tier_after  = _classify_tier(decayed)

        detail = {
            "age_seconds":        round(age_seconds, 1),
            "age_minutes":        round(age_seconds / 60, 2),
            "decay_zone":         zone,
            "points_lost":        round(points_lost, 2),
            "original_confidence": round(original, 2),
            "decayed_confidence": round(decayed, 2),
            "tier_before":        tier_before,
            "tier_after":         tier_after,
            "tier_changed":       tier_before != tier_after,
        }

        if points_lost > 0:
            logger.debug(
                "[Decay] age=%.0fs (%.1fmin) zone=%s lost=%.1f pts  "
                "%s → %.1f  tier=%s→%s",
                age_seconds, age_seconds / 60, zone, points_lost,
                original, decayed, tier_before, tier_after,
            )

        return round(decayed, 2), detail

    @classmethod
    def apply_to_signal(cls, signal: dict, now=None) -> tuple[float, dict]:
        """
        Convenience wrapper for a signal dict.

        Reads the signal's 'time' field (HH:MM string or datetime), computes
        the age vs `now`, then applies decay.

        Args:
            signal: dict with at minimum 'time' (str HH:MM) and
                    'confidence' or 'probability_score' key.
            now:    reference time (datetime-like). Defaults to datetime.now().

        Returns:
            (decayed_confidence, detail_dict)
        """
        from datetime import datetime as _dt
        import pandas as _pd

        if now is None:
            now = _dt.now()

        # Convert 'now' to naive datetime if needed
        if hasattr(now, 'tzinfo') and now.tzinfo is not None:
            try:
                now = now.replace(tzinfo=None)
            except Exception:
                pass

        # Extract original confidence
        original = float(
            signal.get("probability_score")
            or signal.get("confidence")
            or 70.0
        )

        # Parse signal time
        sig_time = signal.get("time")
        if sig_time is None:
            return original, {"decay_zone": "fresh", "points_lost": 0.0, "age_seconds": 0.0,
                              "decayed_confidence": original, "tier_changed": False}

        try:
            if isinstance(sig_time, str):
                h, m = map(int, sig_time.split(":"))
                today = now.date()
                from datetime import time as _time
                sig_dt = _dt.combine(today, _time(h, m))
            elif isinstance(sig_time, _dt):
                sig_dt = sig_time.replace(tzinfo=None) if getattr(sig_time, 'tzinfo', None) else sig_time
            else:
                sig_dt = _dt.combine(now.date(), sig_time)

            age_seconds = (now - sig_dt).total_seconds()
        except Exception:
            age_seconds = 0.0

        return cls.apply(original, age_seconds)

    # ── Batch API ─────────────────────────────────────────────────────────

    @classmethod
    def apply_to_signal_list(cls, signals: list, now=None) -> list:
        """
        Apply decay to a list of signal dicts.

        For each signal adds/updates:
            - confidence_decayed      : decayed confidence value
            - confidence_decay_zone   : 'fresh'|'mild'|'strong'|'expired'
            - confidence_decay_pts    : points removed
            - confidence_decay_age_min: age in minutes

        Original 'confidence' and 'probability_score' are NOT modified.

        Args:
            signals: list of signal dicts
            now:     reference time (default: datetime.now())

        Returns:
            Same list with decay metadata added in-place.
        """
        for sig in signals:
            try:
                decayed, detail = cls.apply_to_signal(sig, now=now)
                sig["confidence_decayed"]       = decayed
                sig["confidence_decay_zone"]    = detail.get("decay_zone", "fresh")
                sig["confidence_decay_pts"]     = detail.get("points_lost", 0.0)
                sig["confidence_decay_age_min"] = detail.get("age_minutes", 0.0)
                sig["confidence_tier_after"]    = detail.get("tier_after", TIER_SKIP)
            except Exception:
                pass
        return signals

    # ── Utility ───────────────────────────────────────────────────────────

    @classmethod
    def is_signal_expired(cls, signal: dict, now=None) -> bool:
        """
        Return True when a signal has passed its strong-decay cap boundary
        (10+ minutes old) — suitable for stale-signal cleanup decisions.
        Signals past STRONG_END_SEC are considered effectively expired.
        """
        _, detail = cls.apply_to_signal(signal, now=now)
        return detail.get("decay_zone") == "expired"

    @classmethod
    def get_decay_summary_text(cls, signal: dict, now=None) -> str:
        """
        Return a compact one-line decay summary for logging/Telegram.

        Example: "⏱ 3.2 min old | mild decay | -2.7 pts | conf 84→81"
        """
        orig   = float(signal.get("probability_score") or signal.get("confidence") or 70.0)
        decayed, detail = cls.apply_to_signal(signal, now=now)
        zone   = detail.get("decay_zone", "fresh")
        age    = detail.get("age_minutes", 0.0)
        lost   = detail.get("points_lost", 0.0)
        emoji  = {"fresh": "🟢", "mild": "🟡", "strong": "🔴", "expired": "⛔"}.get(zone, "")
        return (
            f"{emoji} {age:.1f}min old | {zone} decay "
            f"| -{lost:.1f}pts | conf {orig:.0f}→{decayed:.0f}"
        )


# ── Convenience module-level singleton ──────────────────────────────────────
confidence_decay_engine = ConfidenceDecayEngine()


def apply_decay_to_signal(signal: dict, now=None) -> tuple[float, dict]:
    """
    Module-level shortcut.  Equivalent to confidence_decay_engine.apply_to_signal().
    Import and call directly from any module without importing the class.

    Returns: (decayed_confidence, detail_dict)
    """
    return ConfidenceDecayEngine.apply_to_signal(signal, now=now)
