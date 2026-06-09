"""
agreement_engine.py
───────────────────
Multi-Strategy Agreement Engine (11-voter system).

Collects directional votes from 11 independent strategy sources and returns
an AgreementResult describing how strongly all strategies agree.

11 Strategy Voters:
  1. EMA_Trend              – EMA50 vs EMA200 directional cross
  2. RSI_Momentum           – RSI direction + momentum trend
  3. Pattern_Engine         – Recurring historical win-rate for this slot+direction
  4. Probability_Score      – Weighted probability score margin above regime floor
  5. Market_Regime          – Regime directional bias (TRENDING favors live direction)
  6. Volatility_Clustering  – Healthy volatility zone + quality score alignment
  7. Live_Confirmation      – EMA + RSI + ATR live gate (same as confirmation engine)
  8. Momentum_Continuation  – Candle momentum continuation score
  9. Sequence_Pattern       – Multi-candle sequence pattern engine (6 pattern types)
  10. Market_Structure      – Structural trend, recent BOS/CHOCH and liquidity zones
  11. Liquidity_Sweep       – Buy/Sell-side liquidity sweep detection and rejection

Agreement Tiers (11-voter system):
  9/11+ votes matching direction → STRONG_SIGNAL
  7/11-8/11 votes matching direction → MODERATE_SIGNAL
  <7/11                          → SKIP

Usage:
    from agreement_engine import agreement_engine, AgreementResult, AGREEMENT_SKIP
    result = agreement_engine.compute(df, direction, metrics, regime, prob_result)
    if result.tier == AGREEMENT_SKIP:
        continue  # reject this signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from logger import logger
from sequence_engine import sequence_engine, get_sequence_vote, SequenceResult

# ── Voter names (canonical) ────────────────────────────────────────────────
VOTER_EMA_TREND         = "EMA_Trend"
VOTER_RSI_MOMENTUM      = "RSI_Momentum"
VOTER_PATTERN_ENGINE    = "Pattern_Engine"
VOTER_PROBABILITY_SCORE = "Probability_Score"
VOTER_MARKET_REGIME     = "Market_Regime"
VOTER_VOLATILITY        = "Volatility_Clustering"
VOTER_LIVE_CONFIRMATION = "Live_Confirmation"
VOTER_MOMENTUM_CONT     = "Momentum_Continuation"
VOTER_SEQUENCE_PATTERN  = "Sequence_Pattern"
VOTER_MARKET_STRUCTURE  = "Market_Structure"
VOTER_LIQUIDITY_SWEEP   = "Liquidity_Sweep"

ALL_VOTERS = [
    VOTER_EMA_TREND,
    VOTER_RSI_MOMENTUM,
    VOTER_PATTERN_ENGINE,
    VOTER_PROBABILITY_SCORE,
    VOTER_MARKET_REGIME,
    VOTER_VOLATILITY,
    VOTER_LIVE_CONFIRMATION,
    VOTER_MOMENTUM_CONT,
    VOTER_SEQUENCE_PATTERN,
    VOTER_MARKET_STRUCTURE,
    VOTER_LIQUIDITY_SWEEP,
]

TOTAL_VOTERS = len(ALL_VOTERS)  # 11

# ── Agreement tiers ────────────────────────────────────────────────────────
AGREEMENT_STRONG   = "STRONG_SIGNAL"
AGREEMENT_MODERATE = "MODERATE_SIGNAL"
AGREEMENT_SKIP     = "SKIP"

STRONG_THRESHOLD   = 9  # 9/11+ → STRONG  (~82% agreement)
MODERATE_THRESHOLD = 7  # 7/11-8/11 → MODERATE  (~64% agreement)

# ── Vote values ────────────────────────────────────────────────────────────
VOTE_CALL    = "CALL"
VOTE_PUT     = "PUT"
VOTE_NEUTRAL = "NEUTRAL"


@dataclass
class AgreementResult:
    """Immutable result returned by AgreementEngine.compute()."""

    direction:           str  = ""            # Candidate direction being evaluated
    agreement_score:     int  = 0             # Votes agreeing with candidate direction
    total_voters:        int  = TOTAL_VOTERS  # Always 11
    tier:                str  = AGREEMENT_SKIP
    agreement_direction: str  = "MIXED"       # "CALL" | "PUT" | "MIXED"
    bullish_votes:       int  = 0
    bearish_votes:       int  = 0
    neutral_votes:       int  = 0
    votes: dict = field(default_factory=dict)  # voter_name → CALL/PUT/NEUTRAL

    def as_dict(self) -> dict:
        return {
            "agreement_score":     self.agreement_score,
            "total_voters":        self.total_voters,
            "tier":                self.tier,
            "agreement_direction": self.agreement_direction,
            "bullish_votes":       self.bullish_votes,
            "bearish_votes":       self.bearish_votes,
            "neutral_votes":       self.neutral_votes,
            "votes":               dict(self.votes),
        }

    def format_for_telegram(self) -> str:
        """
        Return a Telegram-formatted agreement breakdown block.

        Example:
            Agreement: 7/11 🟢
              ✅ EMA Trend: CALL
              ✅ RSI Momentum: CALL
              ✅ Pattern Engine: CALL
              ✅ Probability Score: CALL
              ✅ Market Regime: CALL
              ⬜ Volatility Clustering: NEUTRAL
              ✅ Live Confirmation: CALL
              ✅ Momentum Continuation: CALL
        """
        if self.tier == AGREEMENT_STRONG:
            score_emoji = "🟢"
        elif self.tier == AGREEMENT_MODERATE:
            score_emoji = "🟡"
        else:
            score_emoji = "🔴"

        lines = [f"Agreement: {self.agreement_score}/{self.total_voters} {score_emoji}"]
        for voter, vote in self.votes.items():
            label = voter.replace("_", " ")
            if vote == self.direction:
                icon = "✅"
            elif vote == VOTE_NEUTRAL:
                icon = "⬜"
            else:
                icon = "❌"  # opposing direction
            lines.append(f"  {icon} {label}: {vote}")
        return "\n".join(lines)

    def short_summary(self) -> str:
        """One-line summary: 'Agreement: 7/11 🟢 STRONG_SIGNAL'"""
        if self.tier == AGREEMENT_STRONG:
            emoji = "🟢"
        elif self.tier == AGREEMENT_MODERATE:
            emoji = "🟡"
        else:
            emoji = "🔴"
        return f"{self.agreement_score}/{self.total_voters} {emoji} {self.tier}"


class AgreementEngine:
    """
    Multi-strategy agreement engine.

    Singleton — instantiate once and call compute() for each candidate signal.
    All voters are computed from the same data that was already fetched/enriched
    for the signal_generator pipeline; no extra API calls needed.
    """

    def compute(
        self,
        df: pd.DataFrame,
        direction: str,
        metrics: dict,
        regime: str,
        prob_result,                    # ProbabilityResult (from probability_engine)
        live_direction: Optional[str] = None,
        market_structure: Optional[any] = None,
    ) -> AgreementResult:
        """
        Compute multi-strategy agreement for a single candidate signal.

        Args:
            df:             Full enriched DataFrame (latest candle = iloc[-1])
            direction:      Candidate direction to evaluate ("CALL" or "PUT")
            metrics:        Sub-metric dict from _analyse_probability_slot()
                            Keys: win_rate, direction_consistency, momentum_strength,
                                  volatility_quality, volatility_zone, reversal_risk, etc.
            regime:         Current market regime string
            prob_result:    ProbabilityResult from probability_engine.compute()
            live_direction: Pre-computed live direction (avoids recomputation)

        Returns:
            AgreementResult with vote breakdown and tier classification.
        """
        votes: dict[str, str] = {}

        if df is None or len(df) < 5:
            logger.debug("[Agreement] Insufficient df data — returning SKIP")
            return AgreementResult(direction=direction, tier=AGREEMENT_SKIP)

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        # ── Voter 1: EMA Trend ─────────────────────────────────────────────
        # EMA50 > EMA200 → bullish trend → CALL; vice versa → PUT.
        try:
            ema50  = float(last.get("EMA50",  0) or 0)
            ema200 = float(last.get("EMA200", 0) or 0)
            if ema50 > ema200 * 1.0001:         # tiny buffer to avoid noise at cross
                votes[VOTER_EMA_TREND] = VOTE_CALL
            elif ema50 < ema200 * 0.9999:
                votes[VOTER_EMA_TREND] = VOTE_PUT
            else:
                votes[VOTER_EMA_TREND] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_EMA_TREND] = VOTE_NEUTRAL

        # ── Voter 2: RSI Momentum ──────────────────────────────────────────
        # RSI > 52 and rising → CALL; RSI < 48 and falling → PUT.
        try:
            rsi_now  = float(last.get("RSI", 50) or 50)
            rsi_prev = float(prev.get("RSI", 50) or 50)
            if rsi_now > 52 and rsi_now >= rsi_prev:
                votes[VOTER_RSI_MOMENTUM] = VOTE_CALL
            elif rsi_now < 48 and rsi_now <= rsi_prev:
                votes[VOTER_RSI_MOMENTUM] = VOTE_PUT
            else:
                votes[VOTER_RSI_MOMENTUM] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_RSI_MOMENTUM] = VOTE_NEUTRAL

        # ── Voter 3: Pattern Engine ────────────────────────────────────────
        # Historical win rate for this slot + direction is strong (>= 62%).
        # Also checks direction_consistency to confirm the bias is real.
        try:
            win_rate = float(metrics.get("win_rate", 50))
            dir_cons = float(metrics.get("direction_consistency", 50))
            if win_rate >= 62 and dir_cons >= 60:
                votes[VOTER_PATTERN_ENGINE] = direction   # confirms our candidate
            elif win_rate >= 55 and dir_cons >= 55:
                votes[VOTER_PATTERN_ENGINE] = VOTE_NEUTRAL  # borderline
            else:
                # Opposite direction has stronger pattern
                opp = VOTE_PUT if direction == VOTE_CALL else VOTE_CALL
                votes[VOTER_PATTERN_ENGINE] = opp
        except Exception as e:
            votes[VOTER_PATTERN_ENGINE] = VOTE_NEUTRAL

        # ── Voter 4: Weighted Probability Score ────────────────────────────
        # Checks if prob_score is comfortably above the regime floor (margin ≥ 8 pts).
        # Borderline signals (prob_score close to floor) get NEUTRAL.
        try:
            from probability_engine import _regime_min_score
            prob_score   = float(getattr(prob_result, "probability_score", 0))
            regime_floor = _regime_min_score(regime)
            margin       = prob_score - regime_floor
            if margin >= 8.0:
                votes[VOTER_PROBABILITY_SCORE] = direction   # strong above floor
            elif margin >= 0.0:
                votes[VOTER_PROBABILITY_SCORE] = VOTE_NEUTRAL  # at floor — borderline
            else:
                votes[VOTER_PROBABILITY_SCORE] = VOTE_NEUTRAL  # should not reach here
        except Exception as e:
            votes[VOTER_PROBABILITY_SCORE] = VOTE_NEUTRAL

        # ── Voter 5: Market Regime ─────────────────────────────────────────
        # TRENDING: vote follows the live direction (strongest directional conviction).
        # REVERSAL_HEAVY: skeptical; only votes if direction_consistency is strong.
        # HIGH_VOLATILITY: only votes if prob_score is very strong (≥82).
        # SIDEWAYS: neutral (no directional bias expected).
        try:
            if regime == "TRENDING":
                if live_direction and live_direction == direction:
                    votes[VOTER_MARKET_REGIME] = direction
                elif live_direction and live_direction != direction:
                    votes[VOTER_MARKET_REGIME] = live_direction  # votes against candidate
                else:
                    votes[VOTER_MARKET_REGIME] = direction       # no live dir — benefit of doubt
            elif regime == "REVERSAL_HEAVY":
                dir_cons = float(metrics.get("direction_consistency", 50))
                if dir_cons >= 70:
                    votes[VOTER_MARKET_REGIME] = direction
                else:
                    votes[VOTER_MARKET_REGIME] = VOTE_NEUTRAL
            elif regime == "HIGH_VOLATILITY":
                prob_score = float(getattr(prob_result, "probability_score", 0))
                if prob_score >= 82:
                    votes[VOTER_MARKET_REGIME] = direction
                else:
                    votes[VOTER_MARKET_REGIME] = VOTE_NEUTRAL
            else:
                # SIDEWAYS / MODERATE → neutral
                votes[VOTER_MARKET_REGIME] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_MARKET_REGIME] = VOTE_NEUTRAL

        # ── Voter 6: Volatility Clustering ────────────────────────────────
        # Healthy volatility zone + quality score ≥ 58 → supports any direction.
        # Dead / noisy / spike zones are neutral (signal would already be gated by now).
        try:
            vol_zone    = metrics.get("volatility_zone", "noisy")
            vol_quality = float(metrics.get("volatility_quality", 50))
            if vol_zone == "healthy" and vol_quality >= 58:
                votes[VOTER_VOLATILITY] = direction   # healthy vol supports candidate
            else:
                votes[VOTER_VOLATILITY] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_VOLATILITY] = VOTE_NEUTRAL

        # ── Voter 7: Live Confirmation Engine ─────────────────────────────
        # Replicates the core live gate: EMA alignment + RSI direction + ATR health.
        # This mirrors _live_confirmation_ok() without the confidence threshold gate.
        try:
            ema50_v  = float(last.get("EMA50",  0) or 0)
            ema200_v = float(last.get("EMA200", 0) or 0)
            rsi_v    = float(last.get("RSI",   50) or 50)
            rsi_pv   = float(prev.get("RSI",   50) or 50)
            atr_now  = float(last.get("ATR",    0) or 0)
            atr_tail = df["ATR"].tail(80)
            atr_mean = float(pd.to_numeric(atr_tail, errors="coerce").mean() or 0)

            atr_ok = (atr_mean > 0
                      and atr_now >= atr_mean * 0.60
                      and atr_now <= atr_mean * 3.00)

            if atr_ok:
                if direction == VOTE_CALL:
                    live_ok = (ema50_v > ema200_v and rsi_v > 50 and rsi_v >= rsi_pv)
                else:
                    live_ok = (ema50_v < ema200_v and rsi_v < 50 and rsi_v <= rsi_pv)
            else:
                live_ok = False

            votes[VOTER_LIVE_CONFIRMATION] = direction if live_ok else VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_LIVE_CONFIRMATION] = VOTE_NEUTRAL

        # ── Voter 8: Momentum Continuation ────────────────────────────────
        # Composite momentum_strength from the slot metrics ≥ 58 → confident continuation.
        try:
            mom = float(metrics.get("momentum_strength", 50))
            if direction == VOTE_CALL and mom >= 58:
                votes[VOTER_MOMENTUM_CONT] = VOTE_CALL
            elif direction == VOTE_PUT and mom >= 58:
                votes[VOTER_MOMENTUM_CONT] = VOTE_PUT
            else:
                votes[VOTER_MOMENTUM_CONT] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_MOMENTUM_CONT] = VOTE_NEUTRAL

        # ── Voter 9: Sequence Pattern ──────────────────────────────────────
        # Multi-candle sequence engine: detects continuation, breakout, exhaustion,
        # momentum chain, and wick rejection patterns. Casts directional vote when
        # sequence_confidence >= 40. Strong patterns reinforce; weak/conflicting reduce.
        try:
            # Accept pre-computed seq_result if passed; otherwise run analysis
            seq_result: SequenceResult = metrics.get("_sequence_result") or \
                sequence_engine.analyse(df, direction_hint=direction)
            votes[VOTER_SEQUENCE_PATTERN] = get_sequence_vote(seq_result, direction)
        except Exception as e:
            votes[VOTER_SEQUENCE_PATTERN] = VOTE_NEUTRAL

        # ── Voter 10: Market Structure ─────────────────────────────────────
        # Bullish structure votes CALL if not near opposing liquidity. Bearish votes PUT.
        try:
            if market_structure is not None:
                if market_structure.trend == "BULLISH" and not market_structure.near_opposing_liquidity:
                    votes[VOTER_MARKET_STRUCTURE] = VOTE_CALL
                elif market_structure.trend == "BEARISH" and not market_structure.near_opposing_liquidity:
                    votes[VOTER_MARKET_STRUCTURE] = VOTE_PUT
                else:
                    votes[VOTER_MARKET_STRUCTURE] = VOTE_NEUTRAL
            else:
                votes[VOTER_MARKET_STRUCTURE] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_MARKET_STRUCTURE] = VOTE_NEUTRAL

        # ── Voter 11: Liquidity Sweep ───────────────────────────────────
        # A sweep in the SAME direction as our candidate confirms a trap reversal.
        # e.g., SSL (sell-side sweep → CALL signal) + candidate direction CALL → votes CALL.
        # If no sweep detected, votes NEUTRAL (abstains without penalising).
        # If opposing strong sweep detected, votes against the candidate direction.
        try:
            if market_structure is not None and getattr(market_structure, "sweep_result", None) is not None:
                ms_sweep = market_structure
                if ms_sweep.has_strong_sweep:
                    if ms_sweep.sweep_direction == direction:
                        # Strong sweep confirms our candidate direction (trap reversal)
                        votes[VOTER_LIQUIDITY_SWEEP] = direction
                    elif ms_sweep.sweep_direction and ms_sweep.sweep_direction != direction:
                        # Strong opposing sweep — price swept toward our direction,
                        # but rejection was the other way: vote against candidate
                        opp = VOTE_PUT if direction == VOTE_CALL else VOTE_CALL
                        votes[VOTER_LIQUIDITY_SWEEP] = opp
                    else:
                        votes[VOTER_LIQUIDITY_SWEEP] = VOTE_NEUTRAL
                elif ms_sweep.sweep_result.detected and ms_sweep.sweep_confidence >= 50:
                    # Moderate sweep with enough confidence — follow its direction
                    if ms_sweep.sweep_direction == direction:
                        votes[VOTER_LIQUIDITY_SWEEP] = direction
                    else:
                        votes[VOTER_LIQUIDITY_SWEEP] = VOTE_NEUTRAL
                else:
                    # No sweep detected or weak sweep — abstain
                    votes[VOTER_LIQUIDITY_SWEEP] = VOTE_NEUTRAL
            else:
                votes[VOTER_LIQUIDITY_SWEEP] = VOTE_NEUTRAL
        except Exception as e:
            votes[VOTER_LIQUIDITY_SWEEP] = VOTE_NEUTRAL

        # ── Tally votes ────────────────────────────────────────────────────
        bullish_votes = sum(1 for v in votes.values() if v == VOTE_CALL)
        bearish_votes = sum(1 for v in votes.values() if v == VOTE_PUT)
        neutral_votes = sum(1 for v in votes.values() if v == VOTE_NEUTRAL)

        # Agreement score = votes matching our candidate direction
        agreement_score = bullish_votes if direction == VOTE_CALL else bearish_votes

        # Overall market direction from all votes
        if bullish_votes > bearish_votes:
            agreement_direction = VOTE_CALL
        elif bearish_votes > bullish_votes:
            agreement_direction = VOTE_PUT
        else:
            agreement_direction = "MIXED"

        # ── Classify tier ──────────────────────────────────────────────────
        if agreement_score >= STRONG_THRESHOLD:
            tier = AGREEMENT_STRONG
        elif agreement_score >= MODERATE_THRESHOLD:
            tier = AGREEMENT_MODERATE
        else:
            tier = AGREEMENT_SKIP

        result = AgreementResult(
            direction           = direction,
            agreement_score     = agreement_score,
            total_voters        = TOTAL_VOTERS,
            tier                = tier,
            agreement_direction = agreement_direction,
            votes               = votes,
            bullish_votes       = bullish_votes,
            bearish_votes       = bearish_votes,
            neutral_votes       = neutral_votes,
        )

        logger.debug(
            "[Agreement] %s %s | score=%d/%d | tier=%s | "
            "EMA=%s RSI=%s Pat=%s Prob=%s Reg=%s Vol=%s Live=%s Mom=%s Seq=%s Str=%s Swp=%s",
            direction, regime,
            agreement_score, TOTAL_VOTERS, tier,
            votes.get(VOTER_EMA_TREND,         "?"),
            votes.get(VOTER_RSI_MOMENTUM,       "?"),
            votes.get(VOTER_PATTERN_ENGINE,     "?"),
            votes.get(VOTER_PROBABILITY_SCORE,  "?"),
            votes.get(VOTER_MARKET_REGIME,      "?"),
            votes.get(VOTER_VOLATILITY,         "?"),
            votes.get(VOTER_LIVE_CONFIRMATION,  "?"),
            votes.get(VOTER_MOMENTUM_CONT,      "?"),
            votes.get(VOTER_SEQUENCE_PATTERN,   "?"),
            votes.get(VOTER_MARKET_STRUCTURE,   "?"),
            votes.get(VOTER_LIQUIDITY_SWEEP,    "?"),
        )

        return result


# ── Global singleton ───────────────────────────────────────────────────────
agreement_engine = AgreementEngine()
