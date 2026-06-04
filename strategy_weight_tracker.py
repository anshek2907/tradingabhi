"""
strategy_weight_tracker.py
──────────────────────────
Dynamic Strategy Weighting System.

Tracks the real-world prediction accuracy of each strategy voter and
automatically adjusts their influence in the probability and agreement
engines on a rolling-window basis.

Tracked voters (9 total, matching agreement_engine.py):
  1. EMA_Trend              – EMA50 vs EMA200 cross direction
  2. RSI_Momentum           – RSI directional momentum
  3. Pattern_Engine         – Historical slot win-rate
  4. Probability_Score      – Centralized prob score margin
  5. Market_Regime          – Regime directional bias
  6. Volatility_Clustering  – Volatility zone quality
  7. Live_Confirmation      – EMA + RSI + ATR live gate
  8. Momentum_Continuation  – Candle momentum continuation
  9. Sequence_Pattern       – Multi-candle sequence engine

How it works:
  1. After every resolved trade (WIN/LOSS), record which voters agreed
     with the final signal direction.
  2. Maintain a rolling window (last 50 completed trades per voter)
     of vote outcomes (correct=1, incorrect=0).
  3. Daily: compute rolling win rate per voter; use it to scale that
     voter's weight in the ProbabilityEngine formula and AgreementEngine.
  4. Persist weights to strategy_voter_weights.json (atomic, with backups).

Weight adjustment rules:
  • Rolling win rate ≥ 70% → increase voter weight by +STEP (max +MAX_DRIFT)
  • Rolling win rate 55–69% → small +HALF_STEP boost (marginal outperformers)
  • Rolling win rate 45–54% → no change (neutral / not enough edge)
  • Rolling win rate 35–44% → small -HALF_STEP reduction
  • Rolling win rate < 35%  → decrease voter weight by -STEP (min -MAX_DRIFT)

Voter weights are used in two places:
  A) AgreementEngine: scale each voter's vote contribution (0.5–2.0 multiplier)
  B) ProbabilityEngine: map voter performance onto formula metric weights

File: strategy_voter_weights.json
Usage:
    from strategy_weight_tracker import strategy_weight_tracker
    # Record a completed trade
    strategy_weight_tracker.record_vote_outcomes(votes_dict, trade_result, direction)
    # Get current voter weights
    weights = strategy_weight_tracker.get_voter_weights()
    # Daily update (called from generate_daily_signals)
    strategy_weight_tracker.run_daily_update()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from logger import logger
from persistence import safe_load_json, safe_save_json

# ── Constants ─────────────────────────────────────────────────────────────────

VOTER_WEIGHTS_FILE = "strategy_voter_weights.json"

# All 9 voter names (must match agreement_engine.ALL_VOTERS)
ALL_VOTERS = [
    "EMA_Trend",
    "RSI_Momentum",
    "Pattern_Engine",
    "Probability_Score",
    "Market_Regime",
    "Volatility_Clustering",
    "Live_Confirmation",
    "Momentum_Continuation",
    "Sequence_Pattern",
]

# Rolling window: use last N resolved trades per voter
ROLLING_WINDOW = 50

# Per-update step sizes (fraction of baseline weight)
_STEP      = 0.04   # ±4% per update for strong performers/underperformers
_HALF_STEP = 0.02   # ±2% per update for marginal performers

# Max drift of the multiplier from baseline (1.0)
# Voter weight multiplier is clamped to [1.0 - MAX_DRIFT, 1.0 + MAX_DRIFT]
_MAX_DRIFT = 0.40   # ±40% from baseline → multiplier range [0.60, 1.40]

# Minimum trades before adjusting a voter's weight
_MIN_TRADES_FOR_ADJUSTMENT = 10

# Thresholds for weight changes
_WR_HIGH       = 70.0   # ≥70% → strong outperformer → +STEP
_WR_ABOVE_AVG  = 55.0   # 55–69% → marginal outperformer → +HALF_STEP
_WR_BELOW_AVG  = 45.0   # 35–44% → marginal underperformer → -HALF_STEP
_WR_LOW        = 35.0   # <35% → strong underperformer → -STEP

# Mapping: voter name → ProbabilityEngine formula metric key
# Used when translating voter performance into formula weight adjustments
VOTER_TO_METRIC: dict[str, str] = {
    "EMA_Trend":            "direction_consistency",
    "RSI_Momentum":         "momentum_strength",
    "Pattern_Engine":       "win_rate",
    "Probability_Score":    "win_rate",          # reinforces win_rate trust
    "Market_Regime":        "direction_consistency",
    "Volatility_Clustering": "volatility_quality",
    "Live_Confirmation":    "atr_quality",
    "Momentum_Continuation": "momentum_strength",
    "Sequence_Pattern":     "momentum_strength",
}


@dataclass
class VoterStats:
    """Running statistics for a single strategy voter."""
    voter_name:   str
    # Rolling history: list of dicts {timestamp, correct (bool)}
    history:      list = field(default_factory=list)
    # Current multiplier (applied to voter's contribution, baseline = 1.0)
    multiplier:   float = 1.0
    last_updated: str = ""

    @property
    def rolling_win_rate(self) -> float:
        """Win rate over the last ROLLING_WINDOW entries (0–100)."""
        window = self.history[-ROLLING_WINDOW:]
        if not window:
            return 50.0
        correct = sum(1 for e in window if e.get("correct", False))
        return round(correct / len(window) * 100.0, 2)

    @property
    def total_predictions(self) -> int:
        return len(self.history)

    def as_dict(self) -> dict:
        window = self.history[-ROLLING_WINDOW:]
        return {
            "voter_name":        self.voter_name,
            "multiplier":        round(self.multiplier, 6),
            "rolling_win_rate":  self.rolling_win_rate,
            "window_size":       len(window),
            "total_predictions": self.total_predictions,
            "last_updated":      self.last_updated,
        }


class StrategyWeightTracker:
    """
    Tracks per-voter prediction accuracy and updates voter weights daily.

    Persists to VOTER_WEIGHTS_FILE.  Thread-safe for single-process use
    (Python GIL — no explicit locks needed for the bot's async event loop).
    """

    def __init__(self, weights_file: str = VOTER_WEIGHTS_FILE):
        self.weights_file = weights_file
        self._stats: dict[str, VoterStats] = {}
        self._load()

    # ─────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = safe_load_json(self.weights_file, default={})
        stored_voters = data.get("voters", {})

        for voter in ALL_VOTERS:
            sv = stored_voters.get(voter, {})
            self._stats[voter] = VoterStats(
                voter_name   = voter,
                history      = sv.get("history", []),
                multiplier   = float(sv.get("multiplier", 1.0)),
                last_updated = sv.get("last_updated", ""),
            )
            # Clamp multiplier on load (guard against corrupted data)
            self._stats[voter].multiplier = self._clamp_multiplier(
                self._stats[voter].multiplier
            )

        # Trim old history entries (keep last 90 days worth)
        self._trim_history()
        logger.debug("[StrategyWeights] Loaded %d voter stats", len(self._stats))

    def _save(self) -> None:
        voters_payload = {}
        for voter, stats in self._stats.items():
            # Only persist the last ROLLING_WINDOW * 2 entries to keep file small
            trimmed_history = stats.history[-ROLLING_WINDOW * 2:]
            voters_payload[voter] = {
                "multiplier":        round(stats.multiplier, 6),
                "rolling_win_rate":  stats.rolling_win_rate,
                "window_size":       len(stats.history[-ROLLING_WINDOW:]),
                "total_predictions": stats.total_predictions,
                "last_updated":      stats.last_updated,
                "history":           trimmed_history,
            }

        payload = {
            "voters":       voters_payload,
            "last_saved":   datetime.now().isoformat(),
            "all_voters":   ALL_VOTERS,
        }
        safe_save_json(self.weights_file, payload, indent=2)

    def _trim_history(self) -> None:
        """Remove history entries older than 30 days."""
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        for stats in self._stats.values():
            stats.history = [
                e for e in stats.history
                if e.get("timestamp", "") >= cutoff
            ]

    @staticmethod
    def _clamp_multiplier(val: float) -> float:
        return max(1.0 - _MAX_DRIFT, min(1.0 + _MAX_DRIFT, float(val)))

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def record_vote_outcomes(
        self,
        votes: dict,
        trade_result: str,
        direction: str,
    ) -> None:
        """
        Record whether each voter's vote was correct for a completed trade.

        Args:
            votes:         Dict of voter_name → "CALL" | "PUT" | "NEUTRAL"
                           (comes from AgreementResult.votes)
            trade_result:  "WIN" or "LOSS"
            direction:     Signal direction that was traded ("CALL" or "PUT")
        """
        if trade_result not in ("WIN", "LOSS"):
            return

        ts = datetime.now().isoformat()
        trade_won = trade_result == "WIN"

        for voter in ALL_VOTERS:
            vote = votes.get(voter, "NEUTRAL")
            if vote == "NEUTRAL":
                # Neutral votes don't contribute to win/loss tracking
                continue

            # A voter is correct if it agreed with direction AND trade won,
            # OR if it voted AGAINST direction AND trade lost.
            voter_agreed = (vote == direction)
            correct = (voter_agreed and trade_won) or (not voter_agreed and not trade_won)

            if voter not in self._stats:
                self._stats[voter] = VoterStats(voter_name=voter)

            self._stats[voter].history.append({
                "timestamp": ts,
                "vote":      vote,
                "direction": direction,
                "result":    trade_result,
                "correct":   correct,
            })

        logger.debug(
            "[StrategyWeights] Recorded %s trade for direction=%s | voters=%s",
            trade_result, direction, list(votes.keys()),
        )

    def run_daily_update(self) -> dict[str, float]:
        """
        Recompute voter multipliers based on rolling win rates.
        Should be called once per day (from generate_daily_signals).

        Returns:
            Dict of voter_name → new_multiplier
        """
        self._trim_history()
        updates: list[str] = []
        now_str = datetime.now().strftime("%Y-%m-%d")

        for voter, stats in self._stats.items():
            window = stats.history[-ROLLING_WINDOW:]
            n = len(window)

            if n < _MIN_TRADES_FOR_ADJUSTMENT:
                logger.debug(
                    "[StrategyWeights] %s — only %d trades (need %d) — skipping",
                    voter, n, _MIN_TRADES_FOR_ADJUSTMENT,
                )
                continue

            wr = stats.rolling_win_rate
            old_mult = stats.multiplier

            # Determine adjustment direction and size
            if wr >= _WR_HIGH:
                delta = +_STEP
            elif wr >= _WR_ABOVE_AVG:
                delta = +_HALF_STEP
            elif wr < _WR_LOW:
                delta = -_STEP
            elif wr < _WR_BELOW_AVG:
                delta = -_HALF_STEP
            else:
                delta = 0.0  # neutral zone — no change

            new_mult = self._clamp_multiplier(old_mult + delta)
            if abs(new_mult - old_mult) > 1e-8:
                stats.multiplier   = round(new_mult, 6)
                stats.last_updated = now_str
                updates.append(
                    f"{voter}: {old_mult:.3f}→{new_mult:.3f} (wr={wr:.1f}%,n={n})"
                )

        if updates:
            logger.info(
                "[StrategyWeights] Daily update — %d voters adjusted: %s",
                len(updates), " | ".join(updates),
            )
        else:
            logger.info("[StrategyWeights] Daily update — no weight changes needed")

        self._save()
        return {v: s.multiplier for v, s in self._stats.items()}

    def get_voter_weights(self) -> dict[str, float]:
        """
        Return current voter multipliers (voter_name → multiplier 0.60–1.40).
        Baseline is 1.0. Higher = voter trusted more; lower = trusted less.
        """
        return {v: s.multiplier for v, s in self._stats.items()}

    def get_voter_multiplier(self, voter_name: str) -> float:
        """Return multiplier for a single voter (defaults to 1.0 if unseen)."""
        return self._stats.get(voter_name, VoterStats(voter_name=voter_name)).multiplier

    def get_formula_weight_adjustments(self) -> dict[str, float]:
        """
        Translate voter multipliers into ProbabilityEngine formula weight deltas.

        Each voter maps to a formula metric (VOTER_TO_METRIC).  When multiple
        voters share the same metric, their multipliers are averaged.
        Returns a dict of metric_key → blended_multiplier.

        This is consumed by probability_engine to dynamically scale formula weights.
        """
        metric_contributions: dict[str, list[float]] = {}
        for voter, metric in VOTER_TO_METRIC.items():
            mult = self.get_voter_multiplier(voter)
            metric_contributions.setdefault(metric, []).append(mult)

        return {
            metric: round(sum(mults) / len(mults), 6)
            for metric, mults in metric_contributions.items()
        }

    def get_stats_report(self) -> str:
        """Return a formatted text report of all voter stats."""
        lines = ["Strategy Voter Performance Report"]
        lines.append(f"  Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        for voter in ALL_VOTERS:
            stats = self._stats.get(voter)
            if not stats:
                continue
            wr = stats.rolling_win_rate
            n  = len(stats.history[-ROLLING_WINDOW:])
            mult = stats.multiplier
            trend = "UP" if mult > 1.02 else ("DOWN" if mult < 0.98 else "FLAT")
            lines.append(
                f"  {voter:<24} wr={wr:5.1f}%  n={n:3d}  mult={mult:.3f}  [{trend}]"
            )
        return "\n".join(lines)

    def get_telegram_summary(self) -> str:
        """Return a short Telegram-friendly summary block."""
        lines = ["Strategy Voter Weights"]
        for voter in ALL_VOTERS:
            stats = self._stats.get(voter)
            if not stats:
                continue
            wr   = stats.rolling_win_rate
            mult = stats.multiplier
            n    = len(stats.history[-ROLLING_WINDOW:])
            if mult >= 1.10:
                bar = "++++"
            elif mult >= 1.02:
                bar = "++"
            elif mult <= 0.90:
                bar = "----"
            elif mult <= 0.98:
                bar = "--"
            else:
                bar = "  =="
            label = voter.replace("_", " ")
            lines.append(f"  {bar} {label}: {wr:.0f}% (x{mult:.2f}, n={n})")
        return "\n".join(lines)

    def reset_voter(self, voter_name: str) -> None:
        """Reset a single voter to baseline multiplier (1.0)."""
        if voter_name in self._stats:
            self._stats[voter_name].multiplier   = 1.0
            self._stats[voter_name].last_updated = datetime.now().strftime("%Y-%m-%d")
            self._save()
            logger.info("[StrategyWeights] Reset voter %s to baseline", voter_name)

    def reset_all(self) -> None:
        """Reset all voters to baseline multipliers (1.0)."""
        for voter in ALL_VOTERS:
            if voter in self._stats:
                self._stats[voter].multiplier = 1.0
        self._save()
        logger.info("[StrategyWeights] All voters reset to baseline")


# ── Global singleton ──────────────────────────────────────────────────────────
strategy_weight_tracker = StrategyWeightTracker()
