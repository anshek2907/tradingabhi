"""
signal_manager.py
─────────────────
Timing Performance Database for the Recurring Pattern Engine.

Responsibilities:
  • Store per-timing win/loss history (keyed by IST time + direction)
  • Calculate Pattern Strength Score  (0-100)
  • Strengthen successful timings / weaken failing ones after each trading day
  • Persist everything to timing_stats.json
  • Provide adaptive multipliers back to signal_generator.py

Schema of timing_stats.json:
{
  "HH:MM|CALL": {
    "time": "HH:MM",
    "direction": "CALL",
    "total_trades": 12,
    "wins": 9,
    "losses": 3,
    "pattern_strength": 82,          # 0-100 score
    "historical_success_rate": 75.0, # %
    "last_updated": "2026-05-17",
    "daily_history": [               # list of daily results (last 30 days)
      {"date": "2026-05-16", "result": "WIN"},
      ...
    ]
  },
  ...
}
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from persistence import safe_load_json, safe_save_json
from logger import logger

TIMING_STATS_FILE = "timing_stats.json"
_MAX_DAILY_HISTORY_DAYS = 30


class TimingPerformanceDB:
    """Thread-safe (single-process) timing performance store."""

    def __init__(self, stats_file: str = TIMING_STATS_FILE):
        self.stats_file = stats_file
        self._db: dict = self._load()

    # ──────────────────────────────────────────────────────
    # I/O
    # ──────────────────────────────────────────────────────

    def _load(self) -> dict:
        data = safe_load_json(self.stats_file, default={})
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self) -> None:
        safe_save_json(self.stats_file, self._db)

    # ──────────────────────────────────────────────────────
    # Internal key helpers
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _key(time_str: str, direction: str) -> str:
        return f"{time_str}|{direction.upper()}"

    def _get_record(self, time_str: str, direction: str) -> dict:
        key = self._key(time_str, direction)
        if key not in self._db:
            self._db[key] = {
                "time": time_str,
                "direction": direction.upper(),
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "pattern_strength": 50,
                "historical_success_rate": 50.0,
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "daily_history": [],
                # Per-regime performance breakdown
                "regime_history": {
                    "TRENDING":        {"trades": 0, "wins": 0},
                    "SIDEWAYS":        {"trades": 0, "wins": 0},
                    "HIGH_VOLATILITY": {"trades": 0, "wins": 0},
                    "REVERSAL_HEAVY":  {"trades": 0, "wins": 0},
                },
            }
        # Back-fill regime_history for records created before this feature
        rec = self._db[key]
        if "regime_history" not in rec:
            rec["regime_history"] = {
                "TRENDING":        {"trades": 0, "wins": 0},
                "SIDEWAYS":        {"trades": 0, "wins": 0},
                "HIGH_VOLATILITY": {"trades": 0, "wins": 0},
                "REVERSAL_HEAVY":  {"trades": 0, "wins": 0},
            }
        return rec

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    def record_result(
        self,
        time_str: str,
        direction: str,
        result: str,
        regime: str = "UNKNOWN",
    ) -> None:
        """
        Record WIN or LOSS for a timing slot after a trading day.

        Args:
            time_str:  IST time e.g. "15:05"
            direction: "CALL" or "PUT"
            result:    "WIN" or "LOSS"
            regime:    market regime at trade time (optional, for regime history tracking)
        """
        result = result.upper()
        if result not in ("WIN", "LOSS"):
            logger.warning(f"[TimingDB] Invalid result '{result}' for {time_str}|{direction}")
            return

        rec   = self._get_record(time_str, direction)
        today = datetime.now().strftime("%Y-%m-%d")

        # Add to daily history (prevent duplicate same-day entries)
        history: list = rec["daily_history"]
        existing_dates = {e["date"] for e in history}
        if today not in existing_dates:
            history.append({"date": today, "result": result})

        # Trim to last N days
        cutoff = (datetime.now() - timedelta(days=_MAX_DAILY_HISTORY_DAYS)).strftime("%Y-%m-%d")
        rec["daily_history"] = [e for e in history if e["date"] >= cutoff]

        # Recompute aggregate stats from full history
        all_wins  = sum(1 for e in rec["daily_history"] if e["result"] == "WIN")
        all_total = len(rec["daily_history"])
        rec["total_trades"]           = all_total
        rec["wins"]                   = all_wins
        rec["losses"]                 = all_total - all_wins
        rec["historical_success_rate"] = round((all_wins / all_total) * 100, 1) if all_total > 0 else 50.0
        rec["last_updated"]           = today

        # Update regime-specific performance
        if regime and regime != "UNKNOWN":
            self._update_regime_history(rec, regime, result)

        # Recalculate pattern strength
        rec["pattern_strength"] = self._compute_pattern_strength(rec)

        self._save()
        logger.info(
            f"[TimingDB] Recorded {result} for {time_str}|{direction} → "
            f"strength={rec['pattern_strength']} success={rec['historical_success_rate']}%"
        )

    def get_pattern_strength(self, time_str: str, direction: str) -> int:
        """Return pattern strength 0-100 for a timing slot."""
        key = self._key(time_str, direction)
        if key not in self._db:
            return 50  # neutral default for unseen timings
        return self._db[key].get("pattern_strength", 50)

    def get_historical_success_rate(self, time_str: str, direction: str) -> float:
        """Return historical success rate % (0-100)."""
        key = self._key(time_str, direction)
        if key not in self._db:
            return 50.0
        return self._db[key].get("historical_success_rate", 50.0)

    def get_confidence_multiplier(self, time_str: str, direction: str) -> float:
        """
        Return a multiplier (0.80 – 1.20) to scale signal confidence
        based on pattern strength. Neutral at strength=50.
        """
        strength = self.get_pattern_strength(time_str, direction)
        # Linear map: strength 0→0.80, 50→1.00, 100→1.20
        return round(0.80 + (strength / 100) * 0.40, 3)

    def get_adaptive_adjustment(
        self,
        time_str: str,
        direction: str,
        regime: str = "UNKNOWN",
    ) -> int:
        """
        Return an integer adjustment to add to base confidence.
        Range: -12 … +12
        Includes regime-specific bonus/penalty when regime history exists.
        """
        strength = self.get_pattern_strength(time_str, direction)
        # Map strength (0-100) to base adjustment (-10 … +10)
        base_adj = int(round((strength - 50) / 5))
        base_adj = max(-10, min(10, base_adj))

        # Add regime-specific delta
        regime_adj = self._compute_regime_adjustment(time_str, direction, regime)
        total = base_adj + regime_adj
        return max(-12, min(12, total))

    def get_all_stats(self) -> dict:
        """Return the full timing stats dict (read-only copy)."""
        return dict(self._db)

    def get_timing_report(self, time_str: str, direction: str) -> dict:
        """Return a summary dict for display / logging."""
        key = self._key(time_str, direction)
        rec = self._db.get(key, {})
        return {
            "time": time_str,
            "direction": direction,
            "pattern_strength": rec.get("pattern_strength", 50),
            "historical_success_rate": rec.get("historical_success_rate", 50.0),
            "total_trades": rec.get("total_trades", 0),
            "wins": rec.get("wins", 0),
            "losses": rec.get("losses", 0),
        }

    # ──────────────────────────────────────────────────────
    # End-of-day adaptive update
    # ──────────────────────────────────────────────────────

    def run_end_of_day_update(self, results: list[dict]) -> None:
        """
        Call once after each trading day to apply batch updates.

        Args:
            results: list of {
                "time": "HH:MM",
                "direction": "CALL|PUT",
                "result": "WIN|LOSS",
                "regime": "TRENDING|SIDEWAYS|HIGH_VOLATILITY|REVERSAL_HEAVY"  (optional)
            }
        """
        logger.info(f"[TimingDB] Running end-of-day update for {len(results)} trades")
        for r in results:
            self.record_result(
                r.get("time", ""),
                r.get("direction", ""),
                r.get("result", "LOSS"),
                regime=r.get("regime", "UNKNOWN"),
            )

    # ──────────────────────────────────────────────────────
    # Pattern strength calculation
    # ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_pattern_strength(rec: dict) -> int:
        """
        Calculate Pattern Strength Score (0-100) from a record.

        Factors:
          • Historical success rate     (weight 45%)
          • Recent consistency (7 days) (weight 30%)
          • Sample size confidence      (weight 20%)
          • Regime fit bonus/penalty    (±5 points)
        """
        total        = rec.get("total_trades", 0)
        success_rate = rec.get("historical_success_rate", 50.0)
        history: list = rec.get("daily_history", [])

        # 1. Base score from historical success rate
        base_score = success_rate * 0.45  # 0-45

        # 2. Recent consistency (last 7 calendar days)
        cutoff_7d = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        recent = [e for e in history if e["date"] >= cutoff_7d]
        if recent:
            recent_wins = sum(1 for e in recent if e["result"] == "WIN")
            recent_rate = (recent_wins / len(recent)) * 100
        else:
            recent_rate = success_rate  # fall back to overall if no recent data
        recent_score = recent_rate * 0.30  # 0-30

        # 3. Sample size confidence (more trades → more reliable)
        # Caps at 20 points when total_trades >= 14
        sample_score = min(20.0, (total / 14) * 20)

        raw = base_score + recent_score + sample_score
        return max(0, min(100, int(round(raw))))

    @staticmethod
    def _update_regime_history(rec: dict, regime: str, result: str) -> None:
        """Update per-regime win/loss counters in the record."""
        rh = rec.setdefault("regime_history", {})
        if regime not in rh:
            rh[regime] = {"trades": 0, "wins": 0}
        rh[regime]["trades"] += 1
        if result == "WIN":
            rh[regime]["wins"] += 1

    def _compute_regime_adjustment(self, time_str: str, direction: str, regime: str) -> int:
        """
        Return -5 / 0 / +5 based on how well this timing performs in `regime`.
        Only applied when there are at least 3 regime-specific trades.
        """
        if not regime or regime == "UNKNOWN":
            return 0
        key = self._key(time_str, direction)
        rec = self._db.get(key, {})
        rh  = rec.get("regime_history", {}).get(regime, {})
        trades = rh.get("trades", 0)
        wins   = rh.get("wins",   0)
        if trades < 3:
            return 0
        win_rate = wins / trades
        if win_rate >= 0.70:
            return 5
        if win_rate <= 0.35:
            return -5
        return 0

    def get_regime_pattern_strength(
        self,
        time_str: str,
        direction: str,
        regime: str = "UNKNOWN",
    ) -> int:
        """
        Return pattern strength adjusted for regime-specific performance.
        Used by signal_generator to get a more accurate strength estimate
        when regime history is available.
        """
        base     = self.get_pattern_strength(time_str, direction)
        regime_adj = self._compute_regime_adjustment(time_str, direction, regime)
        return max(0, min(100, base + regime_adj))


# ─────────────────────────────────────────────────────────
# Global singleton — importable by signal_generator.py
# ─────────────────────────────────────────────────────────
timing_db = TimingPerformanceDB()
