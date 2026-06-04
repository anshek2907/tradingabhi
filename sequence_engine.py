"""
sequence_engine.py
──────────────────
Sequence Pattern Engine — detects and scores multi-candle price sequences.

Detects six sequence types:
  1. Bullish Continuation   — consecutive closes rising, above EMAs, RSI trending up
  2. Bearish Continuation   — consecutive closes falling, below EMAs, RSI trending down
  3. Compression Breakout   — shrinking range followed by decisive expansion candle
  4. Exhaustion Reversal    — extended run + large wick rejection at extreme RSI
  5. Momentum Candle Chain  — chain of strong-bodied candles with ≥50% body/range
  6. Wick Rejection Cluster — ≥2 consecutive candles with dominant wicks rejecting a level

For each sequence the engine returns:
  - sequence_direction  : "CALL" | "PUT" | "NEUTRAL"
  - sequence_confidence : 0-100
  - continuation_prob   : 0-100  (probability of continuation in sequence_direction)
  - patterns_detected   : list of pattern names found

Integration points:
  • probability_engine.py  — sequence_confidence fed as momentum_strength bonus
  • agreement_engine.py    — Sequence_Pattern voter (9th voter)
  • signal_generator.py    — sequence result stored on candidate signal dict

Usage:
    from sequence_engine import sequence_engine, SequenceResult
    seq = sequence_engine.analyse(df)
    # seq.sequence_direction, seq.sequence_confidence, seq.continuation_prob
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

from logger import logger

# ── Pattern names (canonical) ─────────────────────────────────────────────
PAT_BULLISH_CONTINUATION  = "Bullish_Continuation"
PAT_BEARISH_CONTINUATION  = "Bearish_Continuation"
PAT_COMPRESSION_BREAKOUT  = "Compression_Breakout"
PAT_EXHAUSTION_REVERSAL   = "Exhaustion_Reversal"
PAT_MOMENTUM_CHAIN        = "Momentum_Chain"
PAT_WICK_REJECTION        = "Wick_Rejection_Cluster"

ALL_PATTERNS = [
    PAT_BULLISH_CONTINUATION,
    PAT_BEARISH_CONTINUATION,
    PAT_COMPRESSION_BREAKOUT,
    PAT_EXHAUSTION_REVERSAL,
    PAT_MOMENTUM_CHAIN,
    PAT_WICK_REJECTION,
]

# ── Direction constants ────────────────────────────────────────────────────
SEQ_CALL    = "CALL"
SEQ_PUT     = "PUT"
SEQ_NEUTRAL = "NEUTRAL"

# ── Detection thresholds ───────────────────────────────────────────────────
# Number of recent candles to examine for each pattern
CONTINUATION_LOOKBACK   = 5    # candles for continuation check
MOMENTUM_CHAIN_MIN      = 3    # minimum chain length to qualify
WICK_CLUSTER_MIN        = 2    # minimum wick candles for cluster
COMPRESSION_LOOKBACK    = 8    # candles in compression phase
EXHAUSTION_LOOKBACK     = 10   # run length checked for exhaustion

# Body/range threshold for momentum candles
MOMENTUM_BODY_RATIO     = 0.50  # candle body must be ≥ 50% of range
WICK_REJECTION_RATIO    = 0.60  # wick must be ≥ 60% of range (opposite side)

# RSI thresholds
RSI_OVERBOUGHT   = 70.0
RSI_OVERSOLD     = 30.0
RSI_BULL_ZONE    = 52.0
RSI_BEAR_ZONE    = 48.0

# Sequence confidence weights (applied to detected patterns)
_PATTERN_WEIGHTS: dict[str, float] = {
    PAT_BULLISH_CONTINUATION: 1.0,
    PAT_BEARISH_CONTINUATION: 1.0,
    PAT_COMPRESSION_BREAKOUT: 0.9,
    PAT_EXHAUSTION_REVERSAL:  0.85,
    PAT_MOMENTUM_CHAIN:       0.80,
    PAT_WICK_REJECTION:       0.75,
}

# Minimum candles required before analysis
MIN_CANDLES = 20


@dataclass
class SequenceResult:
    """Immutable result returned by SequenceEngine.analyse()."""

    # Primary outputs
    sequence_direction:  str   = SEQ_NEUTRAL  # dominant sequence direction
    sequence_confidence: float = 0.0          # overall 0-100 confidence
    continuation_prob:   float = 50.0         # 0-100 probability of continuation

    # Detail
    patterns_detected: list = field(default_factory=list)
    bullish_score:     float = 0.0
    bearish_score:     float = 0.0
    conflict_penalty:  float = 0.0

    # Per-pattern detail (pattern_name → sub-score)
    pattern_scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sequence_direction":  self.sequence_direction,
            "sequence_confidence": round(self.sequence_confidence, 2),
            "continuation_prob":   round(self.continuation_prob, 2),
            "patterns_detected":   list(self.patterns_detected),
            "bullish_score":       round(self.bullish_score, 2),
            "bearish_score":       round(self.bearish_score, 2),
            "conflict_penalty":    round(self.conflict_penalty, 2),
            "pattern_scores":      {k: round(v, 2) for k, v in self.pattern_scores.items()},
        }

    def format_for_telegram(self) -> str:
        """Return a Telegram-formatted sequence breakdown block."""
        if self.sequence_direction == SEQ_CALL:
            arrow = "🟢↑"
        elif self.sequence_direction == SEQ_PUT:
            arrow = "🔴↓"
        else:
            arrow = "⬜↔"

        lines = [
            f"Sequence: {arrow} {self.sequence_direction} "
            f"({int(round(self.sequence_confidence))}%)"
        ]
        if self.patterns_detected:
            lines.append(f"  Patterns: {', '.join(p.replace('_', ' ') for p in self.patterns_detected)}")
        lines.append(f"  Continuation Prob: {int(round(self.continuation_prob))}%")
        return "\n".join(lines)

    def short_summary(self) -> str:
        """One-line summary: 'Seq: CALL 72% cont=68%'"""
        return (
            f"Seq: {self.sequence_direction} "
            f"{int(round(self.sequence_confidence))}% "
            f"cont={int(round(self.continuation_prob))}%"
        )


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return default if (v != v) else v   # NaN check
    except Exception:
        return default


def _body_ratio(row) -> float:
    """Candle body as fraction of total range. Returns 0 if range is zero."""
    rng = abs(_safe_float(row.get("High")) - _safe_float(row.get("Low")))
    if rng < 1e-10:
        return 0.0
    body = abs(_safe_float(row.get("Close")) - _safe_float(row.get("Open")))
    return body / rng


def _upper_wick_ratio(row) -> float:
    """Upper wick as fraction of range."""
    rng = abs(_safe_float(row.get("High")) - _safe_float(row.get("Low")))
    if rng < 1e-10:
        return 0.0
    body_top = max(_safe_float(row.get("Open")), _safe_float(row.get("Close")))
    wick = _safe_float(row.get("High")) - body_top
    return max(0.0, wick / rng)


def _lower_wick_ratio(row) -> float:
    """Lower wick as fraction of range."""
    rng = abs(_safe_float(row.get("High")) - _safe_float(row.get("Low")))
    if rng < 1e-10:
        return 0.0
    body_bot = min(_safe_float(row.get("Open")), _safe_float(row.get("Close")))
    wick = body_bot - _safe_float(row.get("Low"))
    return max(0.0, wick / rng)


def _is_bullish_candle(row) -> bool:
    return _safe_float(row.get("Close")) > _safe_float(row.get("Open"))


def _is_bearish_candle(row) -> bool:
    return _safe_float(row.get("Close")) < _safe_float(row.get("Open"))


class SequenceEngine:
    """
    Multi-pattern sequence detection engine.

    Analyses the most-recent candles in df for six structural sequence types
    and returns a SequenceResult with a blended direction + confidence score.

    Design philosophy:
      • Bullish and bearish scores are computed independently.
      • Conflicting signals (e.g. bullish continuation + bearish wick rejection)
        reduce confidence via a conflict_penalty.
      • continuation_prob is a bounded estimate of the probability that the
        sequence_direction signal will materialise in the next candle.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def analyse(
        self,
        df: pd.DataFrame,
        direction_hint: Optional[str] = None,
    ) -> SequenceResult:
        """
        Analyse recent candles for sequence patterns.

        Args:
            df:             Full enriched DataFrame (latest candle = iloc[-1]).
                            Expected columns: Open, High, Low, Close, EMA50,
                            EMA200, RSI, ATR.
            direction_hint: Optional live direction hint ("CALL"/"PUT") from
                            signal_generator to guide tie-breaking.

        Returns:
            SequenceResult with all pattern detection results.
        """
        if df is None or len(df) < MIN_CANDLES:
            logger.debug("[SeqEng] Insufficient data — returning neutral result")
            return SequenceResult()

        # Work with the last N candles (enough for all detectors)
        tail = df.tail(max(EXHAUSTION_LOOKBACK + COMPRESSION_LOOKBACK + 5, 30)).copy()
        tail = tail.reset_index(drop=True)

        bullish_score = 0.0
        bearish_score = 0.0
        patterns_detected: list[str] = []
        pattern_scores: dict[str, float] = {}

        # ── Run all detectors ─────────────────────────────────────────
        results: list[tuple[str, str, float]] = [   # (pattern_name, direction, sub_score)
            self._detect_bullish_continuation(tail),
            self._detect_bearish_continuation(tail),
            self._detect_compression_breakout(tail),
            self._detect_exhaustion_reversal(tail),
            self._detect_momentum_chain(tail),
            self._detect_wick_rejection_cluster(tail),
        ]

        for pat_name, pat_dir, pat_score in results:
            if pat_score <= 0:
                continue
            weighted = pat_score * _PATTERN_WEIGHTS.get(pat_name, 1.0)
            patterns_detected.append(pat_name)
            pattern_scores[pat_name] = round(weighted, 2)
            if pat_dir == SEQ_CALL:
                bullish_score += weighted
            elif pat_dir == SEQ_PUT:
                bearish_score += weighted

        # ── Conflict penalty ──────────────────────────────────────────
        # When both bullish and bearish patterns exist, reduce confidence.
        conflict_penalty = 0.0
        if bullish_score > 0 and bearish_score > 0:
            min_score = min(bullish_score, bearish_score)
            # Penalty = up to 25 points proportional to the conflict magnitude
            conflict_penalty = min(25.0, min_score * 0.5)

        # ── Direction determination ───────────────────────────────────
        adj_bull = max(0.0, bullish_score - conflict_penalty * 0.5)
        adj_bear = max(0.0, bearish_score - conflict_penalty * 0.5)

        if adj_bull > adj_bear and adj_bull > 5.0:
            seq_direction = SEQ_CALL
            dominant_score = adj_bull
        elif adj_bear > adj_bull and adj_bear > 5.0:
            seq_direction = SEQ_PUT
            dominant_score = adj_bear
        else:
            # Tie — use direction_hint if available
            if direction_hint in (SEQ_CALL, SEQ_PUT):
                seq_direction = direction_hint
                dominant_score = max(adj_bull, adj_bear)
            else:
                seq_direction = SEQ_NEUTRAL
                dominant_score = 0.0

        # ── Sequence confidence (0-100) ───────────────────────────────
        # Scale dominant_score: max plausible raw sum ≈ 200 (all patterns firing)
        # Clamp to 0-100 after scaling.
        raw_max = sum(_PATTERN_WEIGHTS.values()) * 100.0  # theoretical max
        sequence_confidence = min(100.0, max(0.0, (dominant_score / max(raw_max, 1.0)) * 100.0 * 3.5))
        sequence_confidence = round(sequence_confidence, 2)

        # ── Continuation probability (0-100) ─────────────────────────
        # Blend: base 50 + directional edge from confidence, capped conservatively.
        if seq_direction == SEQ_NEUTRAL:
            continuation_prob = 50.0
        else:
            # Edge = (dominant - minority) / (dominant + minority + 1e-6)
            minority_score = adj_bear if seq_direction == SEQ_CALL else adj_bull
            edge = (dominant_score - minority_score) / max(dominant_score + minority_score, 1e-6)
            continuation_prob = 50.0 + edge * 35.0   # max edge = 35 above 50 = 85
            continuation_prob = min(85.0, max(20.0, continuation_prob))
            continuation_prob = round(continuation_prob, 2)

        result = SequenceResult(
            sequence_direction  = seq_direction,
            sequence_confidence = sequence_confidence,
            continuation_prob   = continuation_prob,
            patterns_detected   = patterns_detected,
            bullish_score       = round(bullish_score, 2),
            bearish_score       = round(bearish_score, 2),
            conflict_penalty    = round(conflict_penalty, 2),
            pattern_scores      = pattern_scores,
        )

        logger.debug(
            "[SeqEng] dir=%s conf=%.1f cont=%.1f | patterns=%s | "
            "bull=%.1f bear=%.1f penalty=%.1f",
            seq_direction, sequence_confidence, continuation_prob,
            patterns_detected, bullish_score, bearish_score, conflict_penalty,
        )

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Pattern detectors  (each returns: pattern_name, direction, sub_score 0-100)
    # ─────────────────────────────────────────────────────────────────────

    def _detect_bullish_continuation(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Bullish Continuation:
          - Last N closes each higher than previous (rising staircase)
          - Close above EMA50 and EMA200
          - RSI trending upward and > RSI_BULL_ZONE
        """
        try:
            n = CONTINUATION_LOOKBACK
            if len(tail) < n + 1:
                return PAT_BULLISH_CONTINUATION, SEQ_NEUTRAL, 0.0

            recent = tail.tail(n)
            closes = [_safe_float(r.get("Close")) for _, r in recent.iterrows()]

            # Count rising closes
            rising = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            rising_ratio = rising / max(n - 1, 1)

            last = tail.iloc[-1]
            ema50  = _safe_float(last.get("EMA50"))
            ema200 = _safe_float(last.get("EMA200"))
            close  = _safe_float(last.get("Close"))

            above_ema50  = 1.0 if (ema50 > 0 and close > ema50)  else 0.0
            above_ema200 = 1.0 if (ema200 > 0 and close > ema200) else 0.0

            # RSI trend
            rsi_vals = [_safe_float(tail.iloc[-i].get("RSI"), 50.0) for i in range(1, min(4, len(tail)) + 1)]
            rsi_vals.reverse()
            rsi_now   = rsi_vals[-1]
            rsi_trend = (rsi_vals[-1] - rsi_vals[0]) / max(len(rsi_vals) - 1, 1)

            rsi_ok = rsi_now > RSI_BULL_ZONE and rsi_trend >= 0

            # Sub-score: weight each component
            sub_score = (
                rising_ratio   * 50.0 +
                above_ema50    * 20.0 +
                above_ema200   * 15.0 +
                (15.0 if rsi_ok else 0.0)
            )

            if rising_ratio < 0.6:    # less than 60% rising — not a strong sequence
                sub_score *= 0.5

            return PAT_BULLISH_CONTINUATION, SEQ_CALL, round(sub_score, 2)

        except Exception as exc:
            logger.debug("[SeqEng] Bullish continuation error: %s", exc)
            return PAT_BULLISH_CONTINUATION, SEQ_NEUTRAL, 0.0

    def _detect_bearish_continuation(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Bearish Continuation:
          - Last N closes each lower than previous (falling staircase)
          - Close below EMA50 and EMA200
          - RSI trending downward and < RSI_BEAR_ZONE
        """
        try:
            n = CONTINUATION_LOOKBACK
            if len(tail) < n + 1:
                return PAT_BEARISH_CONTINUATION, SEQ_NEUTRAL, 0.0

            recent = tail.tail(n)
            closes = [_safe_float(r.get("Close")) for _, r in recent.iterrows()]

            falling = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i - 1])
            falling_ratio = falling / max(n - 1, 1)

            last = tail.iloc[-1]
            ema50  = _safe_float(last.get("EMA50"))
            ema200 = _safe_float(last.get("EMA200"))
            close  = _safe_float(last.get("Close"))

            below_ema50  = 1.0 if (ema50 > 0 and close < ema50)  else 0.0
            below_ema200 = 1.0 if (ema200 > 0 and close < ema200) else 0.0

            rsi_vals = [_safe_float(tail.iloc[-i].get("RSI"), 50.0) for i in range(1, min(4, len(tail)) + 1)]
            rsi_vals.reverse()
            rsi_now   = rsi_vals[-1]
            rsi_trend = (rsi_vals[-1] - rsi_vals[0]) / max(len(rsi_vals) - 1, 1)

            rsi_ok = rsi_now < RSI_BEAR_ZONE and rsi_trend <= 0

            sub_score = (
                falling_ratio  * 50.0 +
                below_ema50    * 20.0 +
                below_ema200   * 15.0 +
                (15.0 if rsi_ok else 0.0)
            )

            if falling_ratio < 0.6:
                sub_score *= 0.5

            return PAT_BEARISH_CONTINUATION, SEQ_PUT, round(sub_score, 2)

        except Exception as exc:
            logger.debug("[SeqEng] Bearish continuation error: %s", exc)
            return PAT_BEARISH_CONTINUATION, SEQ_NEUTRAL, 0.0

    def _detect_compression_breakout(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Compression Breakout:
          - Previous K candles have narrowing range (ATR or High-Low compression)
          - The final 1-2 candles show a decisive range expansion (≥1.5× compressed avg)
          - Direction = direction of the expansion candle
        """
        try:
            n = COMPRESSION_LOOKBACK
            if len(tail) < n + 2:
                return PAT_COMPRESSION_BREAKOUT, SEQ_NEUTRAL, 0.0

            # Compression phase: n candles before the last 2
            comp_phase = tail.iloc[-(n + 2):-2]
            last_two   = tail.iloc[-2:]

            comp_ranges = [
                abs(_safe_float(r.get("High")) - _safe_float(r.get("Low")))
                for _, r in comp_phase.iterrows()
            ]
            comp_ranges = [r for r in comp_ranges if r > 1e-10]
            if not comp_ranges:
                return PAT_COMPRESSION_BREAKOUT, SEQ_NEUTRAL, 0.0

            avg_comp_range = float(np.mean(comp_ranges))
            range_trend = 0.0
            if len(comp_ranges) >= 3:
                # Positive trend = ranges shrinking over time = compression
                first_half = np.mean(comp_ranges[:len(comp_ranges) // 2])
                second_half = np.mean(comp_ranges[len(comp_ranges) // 2:])
                range_trend = (first_half - second_half) / max(first_half, 1e-10)
                # range_trend > 0 means ranges got smaller (compression)

            # Expansion candle: last candle
            last = tail.iloc[-1]
            exp_range = abs(_safe_float(last.get("High")) - _safe_float(last.get("Low")))
            expansion_ratio = exp_range / max(avg_comp_range, 1e-10)

            # Need expansion ≥ 1.5× the compressed average
            if expansion_ratio < 1.5:
                return PAT_COMPRESSION_BREAKOUT, SEQ_NEUTRAL, 0.0

            # Direction = direction of expansion candle
            if _is_bullish_candle(last):
                direction = SEQ_CALL
            elif _is_bearish_candle(last):
                direction = SEQ_PUT
            else:
                return PAT_COMPRESSION_BREAKOUT, SEQ_NEUTRAL, 0.0

            # Sub-score: compression quality + expansion strength
            compression_quality = max(0.0, min(1.0, range_trend)) * 40.0
            expansion_strength  = min(60.0, (expansion_ratio - 1.5) * 30.0 + 20.0)
            sub_score = compression_quality + expansion_strength

            return PAT_COMPRESSION_BREAKOUT, direction, round(sub_score, 2)

        except Exception as exc:
            logger.debug("[SeqEng] Compression breakout error: %s", exc)
            return PAT_COMPRESSION_BREAKOUT, SEQ_NEUTRAL, 0.0

    def _detect_exhaustion_reversal(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Exhaustion Reversal:
          - Extended run of N candles in one direction
          - RSI at extreme (overbought ≥70 or oversold ≤30)
          - Final candle shows a long wick rejecting the extreme (opposite direction)
          - Direction = OPPOSITE of the extended run (reversal signal)
        """
        try:
            n = EXHAUSTION_LOOKBACK
            if len(tail) < n + 1:
                return PAT_EXHAUSTION_REVERSAL, SEQ_NEUTRAL, 0.0

            run_phase = tail.iloc[-n:-1]
            last      = tail.iloc[-1]

            bullish_in_run = sum(1 for _, r in run_phase.iterrows() if _is_bullish_candle(r))
            bearish_in_run = n - bullish_in_run
            bullish_ratio  = bullish_in_run / n

            rsi_last = _safe_float(last.get("RSI"), 50.0)

            # Check for bullish run → bearish exhaustion
            if bullish_ratio >= 0.70 and rsi_last >= RSI_OVERBOUGHT:
                # Long upper wick on last candle = rejection of highs
                upper_wick = _upper_wick_ratio(last)
                if upper_wick >= WICK_REJECTION_RATIO:
                    run_strength = (bullish_ratio - 0.7) / 0.3   # 0-1
                    rsi_extreme  = (rsi_last - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
                    wick_strength = upper_wick - WICK_REJECTION_RATIO
                    sub_score = (
                        run_strength  * 40.0 +
                        rsi_extreme   * 30.0 +
                        wick_strength * 30.0
                    )
                    return PAT_EXHAUSTION_REVERSAL, SEQ_PUT, round(sub_score, 2)

            # Check for bearish run → bullish exhaustion
            if (1 - bullish_ratio) >= 0.70 and rsi_last <= RSI_OVERSOLD:
                lower_wick = _lower_wick_ratio(last)
                if lower_wick >= WICK_REJECTION_RATIO:
                    run_strength = ((1 - bullish_ratio) - 0.7) / 0.3
                    rsi_extreme  = (RSI_OVERSOLD - rsi_last) / RSI_OVERSOLD
                    wick_strength = lower_wick - WICK_REJECTION_RATIO
                    sub_score = (
                        run_strength  * 40.0 +
                        rsi_extreme   * 30.0 +
                        wick_strength * 30.0
                    )
                    return PAT_EXHAUSTION_REVERSAL, SEQ_CALL, round(sub_score, 2)

            return PAT_EXHAUSTION_REVERSAL, SEQ_NEUTRAL, 0.0

        except Exception as exc:
            logger.debug("[SeqEng] Exhaustion reversal error: %s", exc)
            return PAT_EXHAUSTION_REVERSAL, SEQ_NEUTRAL, 0.0

    def _detect_momentum_chain(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Momentum Candle Chain:
          - Chain of MOMENTUM_CHAIN_MIN+ consecutive candles with body/range ≥ MOMENTUM_BODY_RATIO
          - All candles in chain point the same direction (all bullish or all bearish)
          - Each successive close stronger than previous
        """
        try:
            if len(tail) < MOMENTUM_CHAIN_MIN + 1:
                return PAT_MOMENTUM_CHAIN, SEQ_NEUTRAL, 0.0

            # Scan from most-recent backward to find longest qualifying chain
            best_chain_len  = 0
            best_chain_dir  = SEQ_NEUTRAL
            best_chain_avg  = 0.0

            for start in range(len(tail) - 1, MOMENTUM_CHAIN_MIN - 2, -1):
                row = tail.iloc[start]
                if _body_ratio(row) < MOMENTUM_BODY_RATIO:
                    continue

                # Start of a potential chain
                chain_dir = SEQ_CALL if _is_bullish_candle(row) else SEQ_PUT
                chain_body_ratios = [_body_ratio(row)]
                chain_len = 1

                for prev_i in range(start - 1, max(start - 8, -1), -1):
                    prev_row = tail.iloc[prev_i]
                    prev_dir = SEQ_CALL if _is_bullish_candle(prev_row) else SEQ_PUT
                    if prev_dir != chain_dir:
                        break
                    if _body_ratio(prev_row) < MOMENTUM_BODY_RATIO:
                        break
                    chain_len += 1
                    chain_body_ratios.append(_body_ratio(prev_row))

                if chain_len >= MOMENTUM_CHAIN_MIN and chain_len > best_chain_len:
                    best_chain_len = chain_len
                    best_chain_dir = chain_dir
                    best_chain_avg = float(np.mean(chain_body_ratios))

            if best_chain_len < MOMENTUM_CHAIN_MIN:
                return PAT_MOMENTUM_CHAIN, SEQ_NEUTRAL, 0.0

            # Sub-score: chain length + avg body quality
            length_score = min(60.0, (best_chain_len - MOMENTUM_CHAIN_MIN + 1) * 15.0)
            body_score   = (best_chain_avg - MOMENTUM_BODY_RATIO) / (1.0 - MOMENTUM_BODY_RATIO) * 40.0
            sub_score = length_score + max(0.0, body_score)

            return PAT_MOMENTUM_CHAIN, best_chain_dir, round(sub_score, 2)

        except Exception as exc:
            logger.debug("[SeqEng] Momentum chain error: %s", exc)
            return PAT_MOMENTUM_CHAIN, SEQ_NEUTRAL, 0.0

    def _detect_wick_rejection_cluster(
        self, tail: pd.DataFrame
    ) -> tuple[str, str, float]:
        """
        Wick Rejection Cluster:
          - WICK_CLUSTER_MIN+ consecutive candles with dominant wick on the same side
          - Upper wick cluster → bearish rejection (PUT signal — price rejecting highs)
          - Lower wick cluster → bullish rejection (CALL signal — price rejecting lows)
          - Wick must be ≥ WICK_REJECTION_RATIO of total candle range
        """
        try:
            if len(tail) < WICK_CLUSTER_MIN + 1:
                return PAT_WICK_REJECTION, SEQ_NEUTRAL, 0.0

            recent = tail.tail(WICK_CLUSTER_MIN + 3)

            # Check each sub-sequence of WICK_CLUSTER_MIN candles
            best_upper_wick = 0.0
            best_lower_wick = 0.0
            upper_count     = 0
            lower_count     = 0

            for i in range(len(recent) - WICK_CLUSTER_MIN, len(recent)):
                row = recent.iloc[i]
                uw  = _upper_wick_ratio(row)
                lw  = _lower_wick_ratio(row)
                if uw >= WICK_REJECTION_RATIO:
                    upper_count  += 1
                    best_upper_wick = max(best_upper_wick, uw)
                if lw >= WICK_REJECTION_RATIO:
                    lower_count  += 1
                    best_lower_wick = max(best_lower_wick, lw)

            if upper_count >= WICK_CLUSTER_MIN and upper_count >= lower_count:
                # Upper wick cluster = rejection of highs → PUT
                wick_quality = (best_upper_wick - WICK_REJECTION_RATIO) / (1.0 - WICK_REJECTION_RATIO)
                sub_score = wick_quality * 60.0 + (upper_count - WICK_CLUSTER_MIN) * 10.0
                return PAT_WICK_REJECTION, SEQ_PUT, round(min(100.0, sub_score), 2)

            if lower_count >= WICK_CLUSTER_MIN:
                # Lower wick cluster = rejection of lows → CALL
                wick_quality = (best_lower_wick - WICK_REJECTION_RATIO) / (1.0 - WICK_REJECTION_RATIO)
                sub_score = wick_quality * 60.0 + (lower_count - WICK_CLUSTER_MIN) * 10.0
                return PAT_WICK_REJECTION, SEQ_CALL, round(min(100.0, sub_score), 2)

            return PAT_WICK_REJECTION, SEQ_NEUTRAL, 0.0

        except Exception as exc:
            logger.debug("[SeqEng] Wick rejection error: %s", exc)
            return PAT_WICK_REJECTION, SEQ_NEUTRAL, 0.0


# ── Global singleton ─────────────────────────────────────────────────────
sequence_engine = SequenceEngine()


# ─────────────────────────────────────────────────────────────────────────
# Integration helpers (called from probability_engine and agreement_engine)
# ─────────────────────────────────────────────────────────────────────────

def get_sequence_momentum_bonus(seq_result: SequenceResult, direction: str) -> float:
    """
    Return a momentum bonus/penalty (signed, ±0–20) for the probability engine.

    Logic:
      • If seq_result.sequence_direction matches direction AND confidence ≥ 50:
        → Positive bonus proportional to confidence (max +20)
      • If seq_result.sequence_direction is OPPOSITE to direction:
        → Negative penalty proportional to confidence (max -15)
      • NEUTRAL or low confidence → 0
    """
    conf = seq_result.sequence_confidence
    seq_dir = seq_result.sequence_direction

    if seq_dir == SEQ_NEUTRAL or conf < 30.0:
        return 0.0

    if seq_dir == direction:
        bonus = (conf / 100.0) * 20.0          # max +20
        return round(bonus, 2)
    else:
        penalty = (conf / 100.0) * -15.0       # max -15
        return round(penalty, 2)


def get_sequence_vote(seq_result: SequenceResult, direction: str) -> str:
    """
    Return a voter-style vote string ("CALL" / "PUT" / "NEUTRAL") for
    use in the agreement engine's voter system.

    Requires sequence_confidence ≥ 40 to cast a directional vote.
    """
    if seq_result.sequence_confidence < 40.0:
        return "NEUTRAL"
    if seq_result.sequence_direction == SEQ_NEUTRAL:
        return "NEUTRAL"
    return seq_result.sequence_direction
