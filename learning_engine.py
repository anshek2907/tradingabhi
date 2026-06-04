from logger import logger
import os
from datetime import datetime, timedelta
from persistence import safe_load_json, safe_save_json

class LearningEngine:
    def __init__(self, memory_file="trade_memory.json"):
        self.memory_file = memory_file
        self.memory = self._load_memory()
        self.trades_since_cleanup = 0
        self._cleanup_old_memory()
        
    def _load_memory(self):
        return safe_load_json(self.memory_file, default=[])

    def _save_memory(self):
        safe_save_json(self.memory_file, self.memory, indent=4)

    def _cleanup_old_memory(self):
        """Keep ONLY last 14 days memory and cap at 500 entries. Auto remove older data."""
        cutoff_date = datetime.now() - timedelta(days=14)
        original_length = len(self.memory)
        
        # 1. Filter by date
        valid_memory = []
        for trade in self.memory:
            try:
                ts_str = trade.get("timestamp", datetime.now().isoformat())
                trade_date = datetime.fromisoformat(ts_str)
                if trade_date >= cutoff_date:
                    valid_memory.append(trade)
            except ValueError:
                pass
                
        # 2. Enforce hard cap of 500 (keep latest)
        if len(valid_memory) > 500:
            valid_memory = valid_memory[-500:]
            
        self.memory = valid_memory
        
        if len(self.memory) < original_length:
            self._save_memory()

    def record_trade(
        self,
        time_of_day: str,
        direction: str,
        confidence: int,
        atr: float,
        rsi: float,
        result: str,
        source: str = "telegram",
        regime: str = "UNKNOWN",
        probability_score: float = 0.0,
        trade_type: str = "short_term",
    ):
        """
        Record winning/losing timings and conditions.
        result should be 'WIN' or 'LOSS'
        regime: market regime label at time of trade
        probability_score: centralized probability score at time of trade (0-100)
        trade_type: 'short_term' or 'swing'
        """
        trade_data = {
            "timestamp":         datetime.now().isoformat(),
            "time_of_day":       time_of_day,
            "direction":         direction,
            "confidence":        confidence,
            "atr":               atr,
            "rsi":               rsi,
            "result":            result.upper(),
            "source":            source,
            "regime":            regime,
            "probability_score": round(probability_score, 2),
            "trade_type":        trade_type,
        }
        self.memory.append(trade_data)
        self._save_memory()

        self.trades_since_cleanup += 1
        if self.trades_since_cleanup >= 20:
            self._cleanup_old_memory()
            self.trades_since_cleanup = 0

    def get_adaptive_adjustment(
        self,
        time_of_day: str,
        direction: str,
        confidence: int,
        atr: float,
        rsi: float,
        source: str = "telegram",
        regime: str = "UNKNOWN",
    ) -> int:
        """
        Analyze past trades and return a confidence adjustment.
        Boosts strong timings slightly. Reduces weak timings gradually.
        Source-aware: timings filtered by origin when history is sufficient.
        Regime-aware: applies bonus/penalty based on current regime's historical success.
        """
        if not self.memory:
            return 0

        adjustment = 0

        # 1. Analyze Timings (winning vs losing timings)
        # Prefer source-filtered trades if there are enough; fall back to all
        source_trades = [t for t in self.memory if t.get("source", "telegram") == source]
        timing_pool   = source_trades if len(source_trades) >= 5 else self.memory
        timing_trades = [
            t for t in timing_pool
            if t["time_of_day"] == time_of_day and t["direction"] == direction
        ]
        if len(timing_trades) >= 2:
            wins     = sum(1 for t in timing_trades if t["result"] == "WIN")
            win_rate = wins / len(timing_trades)

            if win_rate >= 0.70:
                adjustment += 3  # Boost strong timings
            elif win_rate >= 0.51:
                adjustment += 1
            elif win_rate <= 0.30:
                adjustment -= 3  # Reduce weak timings
            elif win_rate <= 0.50:
                adjustment -= 1

            # ── Win/loss streak amplification ──────────────────
            # Sort by timestamp to get chronological order
            sorted_trades = sorted(timing_trades, key=lambda t: t.get("timestamp", ""))
            if len(sorted_trades) >= 2:
                streak = 1
                last_result = sorted_trades[-1]["result"]
                for t in reversed(sorted_trades[:-1]):
                    if t["result"] == last_result:
                        streak += 1
                    else:
                        break
                if last_result == "WIN" and streak >= 3:
                    # Win streak: +1 extra per additional win beyond 2
                    adjustment += min(2, streak - 2)
                elif last_result == "LOSS" and streak >= 2:
                    # Loss streak: -2 per additional loss beyond 1
                    adjustment -= min(3, streak - 1)

        # 2. Analyze RSI / ATR behavior patterns
        atr_threshold = max(atr * 0.2, 0.0001)
        similar_conditions = [
            t for t in self.memory
            if t["direction"] == direction
            and abs(t["rsi"] - rsi) <= 5
            and abs(t["atr"] - atr) <= atr_threshold
        ]
        if len(similar_conditions) >= 3:
            cond_wins     = sum(1 for t in similar_conditions if t["result"] == "WIN")
            cond_win_rate = cond_wins / len(similar_conditions)

            if cond_win_rate >= 0.65:
                adjustment += 2
            elif cond_win_rate <= 0.40:
                adjustment -= 2

        # 3. Analyze best confidence ranges
        conf_range = [
            t for t in self.memory
            if abs(t["confidence"] - confidence) <= 3
        ]
        if len(conf_range) >= 3:
            conf_wins     = sum(1 for t in conf_range if t["result"] == "WIN")
            conf_win_rate = conf_wins / len(conf_range)

            if conf_win_rate >= 0.75:
                adjustment += 1
            elif conf_win_rate <= 0.35:
                adjustment -= 2

        # 4. Probability-score-aware adjustment
        # If historical trades with a HIGH probability_score for this timing
        # have a strong win rate, boost further; if they underperform, penalise.
        high_prob_trades = [
            t for t in self.memory
            if t.get("time_of_day") == time_of_day
            and t.get("direction") == direction
            and t.get("probability_score", 0.0) >= 75.0
        ]
        if len(high_prob_trades) >= 3:
            hp_wins     = sum(1 for t in high_prob_trades if t["result"] == "WIN")
            hp_win_rate = hp_wins / len(high_prob_trades)
            if hp_win_rate >= 0.70:
                adjustment += 1   # high-score signals are historically reliable
            elif hp_win_rate <= 0.40:
                adjustment -= 1   # high-score signals underperforming — be cautious

        # 5. Regime-aware adjustment
        adj_regime = self.get_regime_adjustment(time_of_day, direction, regime)
        adjustment += adj_regime

        # Cap the total adjustment to prevent excessive overriding of base logic
        return max(min(adjustment, 8), -8)

    def get_regime_adjustment(
        self,
        time_of_day: str,
        direction: str,
        regime: str,
    ) -> int:
        """
        Return an adjustment based on how well this timing performs in the current regime.

        +2 → timing historically wins in this regime
        -2 → timing historically loses in this regime
         0 → not enough regime-specific data
        """
        if not regime or regime == "UNKNOWN" or not self.memory:
            return 0

        regime_trades = [
            t for t in self.memory
            if t.get("time_of_day") == time_of_day
            and t.get("direction") == direction
            and t.get("regime") == regime
        ]
        if len(regime_trades) < 2:
            return 0

        wins     = sum(1 for t in regime_trades if t["result"] == "WIN")
        win_rate = wins / len(regime_trades)

        if win_rate >= 0.70:
            return 2
        if win_rate <= 0.35:
            return -2
        return 0

# Global singleton instance for easy access
learning_engine = LearningEngine()


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic Weight Optimizer
# ══════════════════════════════════════════════════════════════════════════════

DYNAMIC_WEIGHTS_FILE = "dynamic_weights.json"

# ── Baseline weights (v2 — must match probability_engine._WEIGHTS exactly) ──
_BASELINE_WEIGHTS = {
    "win_rate":              0.32,
    "direction_consistency": 0.25,
    "atr_quality":           0.12,
    "momentum_strength":     0.15,
    "session_strength":      0.08,
    "volatility_quality":    0.08,
    "reversal_risk":         0.06,
}

# Max drift from baseline: ±30% of each baseline value
_MAX_DRIFT_FACTOR = 0.30


class DynamicWeightOptimizer:
    """
    Self-improving scoring weight manager.

    Tracks which probability formula components correlate best with actual wins
    and makes small bounded adjustments to their weights.  Adjustments are
    persisted to dynamic_weights.json so they survive restarts.

    Weight bounds:
        Each weight can shift ±30% from its baseline value.
        e.g. win_rate baseline 0.32  →  allowed range [0.224, 0.416]

    The adjusted weights are soft overrides — deleting dynamic_weights.json
    resets to the hardcoded v2 baseline, preventing runaway drift.

    Usage:
        from learning_engine import dynamic_weight_optimizer
        weights = dynamic_weight_optimizer.get_weights()
        # feed into ProbabilityEngine or use directly
        dynamic_weight_optimizer.update_from_backtest(backtest_results)
    """

    def __init__(self, weights_file: str = DYNAMIC_WEIGHTS_FILE):
        self.weights_file = weights_file
        self._weights     = self._load_weights()

    # ── I/O ───────────────────────────────────────────────────────────────

    def _load_weights(self) -> dict:
        data = safe_load_json(self.weights_file, default={})
        stored = data.get("weights", {})
        if not stored or not isinstance(stored, dict):
            return dict(_BASELINE_WEIGHTS)

        # Validate and clamp stored weights against baseline bounds
        weights = {}
        for key, baseline in _BASELINE_WEIGHTS.items():
            stored_val = stored.get(key, baseline)
            lo = baseline * (1 - _MAX_DRIFT_FACTOR)
            hi = baseline * (1 + _MAX_DRIFT_FACTOR)
            weights[key] = max(lo, min(hi, float(stored_val)))
        return weights

    def _save_weights(self, meta: dict | None = None) -> None:
        from datetime import datetime as _dt
        payload = {
            "weights":              {k: round(v, 6) for k, v in self._weights.items()},
            "baseline_weights":     {k: round(v, 6) for k, v in _BASELINE_WEIGHTS.items()},
            "last_updated":         _dt.now().strftime("%Y-%m-%d"),
        }
        if meta:
            payload.update(meta)
        safe_save_json(self.weights_file, payload)

    # ── Public API ─────────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """Return current effective probability formula weights."""
        return dict(self._weights)

    def reset_to_baseline(self) -> None:
        """Reset all weights to hardcoded v2 baseline."""
        self._weights = dict(_BASELINE_WEIGHTS)
        self._save_weights({"note": "Manual reset to baseline"})
        logger.info("[WeightOpt] Weights reset to v2 baseline")

    def update_from_backtest(self, backtest_results) -> None:
        """
        Adjust weights based on backtesting performance metrics.

        Strategy:
          • High overall win rate (>70%) → slightly boost win_rate weight
            (historical win rates are proven reliable predictors)
          • High continuation success (>65%) → boost momentum_strength weight
          • High reversal failure rate (>52%) → boost reversal_risk penalty weight
          • Low volatility quality (<50%) → slightly reduce volatility_quality weight
          • Very low overall win rate (<50%) → broad reset toward baseline

        Each adjustment is capped at ±5% of the current weight value per run.
        Cumulative drift is bounded at ±30% of baseline (enforced at load time).

        Args:
            backtest_results: BacktestResults dataclass or dict with keys:
                              overall_win_rate, continuation_success,
                              reversal_failure_rate, volatility_quality_score
        """
        # Accept both BacktestResults dataclass and plain dict
        if hasattr(backtest_results, "as_dict"):
            data = backtest_results.as_dict()
        elif isinstance(backtest_results, dict):
            data = backtest_results
        else:
            logger.warning("[WeightOpt] Unknown backtest_results type — skipping")
            return

        overall_wr   = float(data.get("overall_win_rate",      50.0))
        cont_success = float(data.get("continuation_success",  50.0))
        rev_fail     = float(data.get("reversal_failure_rate", 50.0))
        vol_quality  = float(data.get("volatility_quality_score", 50.0))
        total_sims   = int(  data.get("total_simulated",         0))

        # Need at least 50 simulated trades to trust adjustments
        if total_sims < 50:
            logger.info(
                "[WeightOpt] Too few simulated trades (%d) — skipping weight update",
                total_sims,
            )
            return

        adjustments = {}
        _STEP = 0.005   # 0.5% nudge per update (small, bounded)

        # ── Rule 1: Win rate quality ───────────────────────────────────
        if overall_wr > 70:
            adjustments["win_rate"] = +_STEP          # winning → trust win_rate more
        elif overall_wr < 50:
            adjustments["win_rate"] = -_STEP          # losing  → reduce over-reliance

        # ── Rule 2: Continuation quality ──────────────────────────────
        if cont_success > 65:
            adjustments["momentum_strength"] = +_STEP
        elif cont_success < 45:
            adjustments["momentum_strength"] = -_STEP

        # ── Rule 3: Reversal failure ───────────────────────────────────
        if rev_fail > 55:
            adjustments["reversal_risk"] = +_STEP    # reversals hurting → penalize more
        elif rev_fail < 35:
            adjustments["reversal_risk"] = -_STEP    # reversals rare → reduce penalty

        # ── Rule 4: Volatility quality ─────────────────────────────────
        if vol_quality > 70:
            adjustments["volatility_quality"] = +_STEP
        elif vol_quality < 45:
            adjustments["volatility_quality"] = -_STEP

        # ── Apply adjustments (bounded) ────────────────────────────────
        applied = []
        for key, delta in adjustments.items():
            if key not in self._weights:
                continue
            baseline = _BASELINE_WEIGHTS[key]
            lo       = baseline * (1 - _MAX_DRIFT_FACTOR)
            hi       = baseline * (1 + _MAX_DRIFT_FACTOR)
            old_val  = self._weights[key]
            new_val  = max(lo, min(hi, old_val + delta))
            if abs(new_val - old_val) > 1e-8:
                self._weights[key] = round(new_val, 6)
                applied.append(f"{key}: {old_val:.4f}→{new_val:.4f}")

        if applied:
            self._save_weights({
                "last_backtest_wr":         round(overall_wr, 2),
                "last_backtest_cont":       round(cont_success, 2),
                "last_backtest_rev_fail":   round(rev_fail, 2),
                "last_backtest_vol":        round(vol_quality, 2),
                "adjustments_applied":      len(applied),
            })
            logger.info(
                "[WeightOpt] Weights updated from backtest (wr=%.1f%% cont=%.1f%% "
                "rev=%.1f%% vol=%.1f): %s",
                overall_wr, cont_success, rev_fail, vol_quality,
                " | ".join(applied),
            )
        else:
            logger.debug("[WeightOpt] No weight adjustments needed from backtest")

    def format_weights_report(self) -> str:
        """Format current weights vs baseline for logging / Telegram."""
        lines = ["⚖️ Dynamic Probability Weights"]
        for key, val in self._weights.items():
            base  = _BASELINE_WEIGHTS.get(key, val)
            drift = val - base
            sign  = "+" if drift >= 0 else ""
            label = key.replace("_", " ").title()
            lines.append(f"  {label}: {val:.4f} (base {base:.4f} {sign}{drift:.4f})")
        return "\n".join(lines)


# ── Global singleton ───────────────────────────────────────────────────────
dynamic_weight_optimizer = DynamicWeightOptimizer()


def get_dynamic_weights() -> dict:
    """
    Convenience function — returns the current effective probability formula
    weights from the DynamicWeightOptimizer.

    If dynamic_weights.json is missing or empty, returns the v2 baseline.
    """
    return dynamic_weight_optimizer.get_weights()


# ── Voter-weight forwarding helpers ────────────────────────────────────────
# These proxies let bot.py call voter recording via the already-imported
# learning_engine module, without needing a separate import.

def record_voter_outcome(votes: dict, trade_result: str, direction: str) -> None:
    """
    Forward a completed trade's voter votes to the StrategyWeightTracker.

    Call this from bot.py whenever a trade result (WIN/LOSS) is confirmed:
        from learning_engine import record_voter_outcome
        record_voter_outcome(signal.get("agreement_votes", {}), "WIN", signal["direction"])

    Args:
        votes:        Dict of voter_name → "CALL" | "PUT" | "NEUTRAL"
                      Stored as signal["agreement_votes"] on each generated signal.
        trade_result: "WIN" or "LOSS"
        direction:    The signal direction that was traded ("CALL" or "PUT")
    """
    try:
        from strategy_weight_tracker import strategy_weight_tracker as _swt
        _swt.record_vote_outcomes(votes, trade_result, direction)
    except Exception as exc:
        logger.warning("[VoterRecord] Failed to record voter outcome: %s", exc)


def get_voter_weight_report() -> str:
    """
    Return a formatted text report of current strategy voter weights.
    Suitable for admin Telegram commands or log output.
    """
    try:
        from strategy_weight_tracker import strategy_weight_tracker as _swt
        return _swt.get_stats_report()
    except Exception as exc:
        return f"[VoterWeights] Report unavailable: {exc}"


def get_voter_telegram_summary() -> str:
    """
    Return a short Telegram-friendly summary of current voter multipliers.
    """
    try:
        from strategy_weight_tracker import strategy_weight_tracker as _swt
        return _swt.get_telegram_summary()
    except Exception as exc:
        return f"[VoterWeights] Summary unavailable: {exc}"


# ── Session Intelligence forwarding helpers ────────────────────────────────
# These proxies let bot.py record session outcomes via the already-imported
# learning_engine module without requiring a separate import chain.

def record_session_outcome(
    ist_minutes: int,
    direction: str,
    result: str,
    df=None,
) -> None:
    """
    Forward a completed trade result to the SessionIntelligence engine
    for adaptive session win-rate tracking.

    Call this from bot.py after a trade resolves:
        from learning_engine import record_session_outcome
        record_session_outcome(ist_minutes, signal["direction"], "WIN")

    Args:
        ist_minutes: signal time in IST minutes from midnight (h*60+m)
        direction:   "CALL" or "PUT"
        result:      "WIN" or "LOSS"
        df:          optional enriched DataFrame (for live vol/cont metrics)
    """
    try:
        from session_intelligence import session_intel as _si
        _si.record_session_outcome(ist_minutes, direction, result, df)
    except Exception as exc:
        logger.warning("[SessionRecord] Failed to record session outcome: %s", exc)


def get_session_performance_report() -> str:
    """
    Return a formatted text report of session-level performance.
    Suitable for admin Telegram commands.
    """
    try:
        from session_intelligence import session_intel as _si
        return _si.get_performance_report()
    except Exception as exc:
        return f"[SessionIntel] Report unavailable: {exc}"


def get_session_telegram_summary() -> str:
    """
    Return a short Telegram-friendly summary of session win rates.
    """
    try:
        from session_intelligence import session_intel as _si
        return _si.get_telegram_summary()
    except Exception as exc:
        return f"[SessionIntel] Summary unavailable: {exc}"
