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
    ):
        """
        Record winning/losing timings and conditions.
        result should be 'WIN' or 'LOSS'
        regime: market regime label at time of trade
        probability_score: centralized probability score at time of trade (0-100)
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
