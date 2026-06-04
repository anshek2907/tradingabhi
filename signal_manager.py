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

    # ──────────────────────────────────────────────────────
    # Regime Performance Memory
    # ──────────────────────────────────────────────────────

    def get_regime_overall_stats(self) -> dict:
        """
        Aggregate regime win rates across ALL timing slots in the DB.

        Returns a dict:
        {
            "TRENDING":        {"win_rate": 68.2, "total_trades": 45},
            "SIDEWAYS":        {"win_rate": 51.0, "total_trades": 20},
            "HIGH_VOLATILITY": {"win_rate": 55.0, "total_trades": 12},
            "REVERSAL_HEAVY":  {"win_rate": 48.0, "total_trades": 8},
        }
        """
        REGIMES = ["TRENDING", "SIDEWAYS", "HIGH_VOLATILITY", "REVERSAL_HEAVY"]
        agg: dict[str, dict] = {r: {"wins": 0, "total": 0} for r in REGIMES}

        for rec in self._db.values():
            rh = rec.get("regime_history", {})
            for regime in REGIMES:
                rd = rh.get(regime, {})
                agg[regime]["wins"]  += rd.get("wins",   0)
                agg[regime]["total"] += rd.get("trades", 0)

        result: dict = {}
        for regime, data in agg.items():
            total = data["total"]
            wins  = data["wins"]
            result[regime] = {
                "win_rate":     round((wins / total * 100) if total > 0 else 50.0, 1),
                "total_trades": total,
            }
        return result

    def adapt_regime_thresholds(self, base_thresholds: dict) -> dict:
        """
        Dynamically adjust regime minimum score thresholds based on
        live regime performance from the DB.

        Rules:
          win_rate > 75% → lower threshold by 2 pts (allow more signals)
          win_rate > 65% → lower threshold by 1 pt
          win_rate < 45% → raise  threshold by 3 pts (be more cautious)
          win_rate < 55% → raise  threshold by 1 pt

        Only applies when a regime has at least 5 recorded trades.
        Adjustments are clamped: thresholds never go below 65 or above 90.

        Args:
            base_thresholds: dict of regime → min_score (e.g. from _REGIME_MIN_SCORE)

        Returns:
            Adjusted thresholds dict (same keys, modified values).
        """
        stats    = self.get_regime_overall_stats()
        adjusted = dict(base_thresholds)

        for regime, data in stats.items():
            total    = data["total_trades"]
            win_rate = data["win_rate"]

            if total < 5:
                continue   # not enough data to be confident

            current = adjusted.get(regime, 72.0)

            if win_rate > 75:
                delta = -2
            elif win_rate > 65:
                delta = -1
            elif win_rate < 45:
                delta = +3
            elif win_rate < 55:
                delta = +1
            else:
                delta = 0

            if delta != 0:
                new_threshold = max(65.0, min(90.0, current + delta))
                if new_threshold != current:
                    logger.info(
                        "[TimingDB] Regime threshold adapted: %s %.0f → %.0f "
                        "(win_rate=%.1f%% total=%d)",
                        regime, current, new_threshold, win_rate, total,
                    )
                    adjusted[regime] = new_threshold

        return adjusted

    def get_top_timings(self, n: int = 10) -> list:
        """
        Return the top N timing slots by pattern strength × win rate.

        Returns a list of dicts sorted by composite score (descending):
        [
            {
                "key":               "15:05|CALL",
                "time":              "15:05",
                "direction":         "CALL",
                "pattern_strength":  87,
                "win_rate":          81.0,
                "total_trades":      14,
                "composite_score":   70.5,
                "bullish_consistency": 81.0,
                "bearish_consistency": 19.0,
            },
            ...
        ]
        """
        rows: list[dict] = []
        for key, rec in self._db.items():
            total = rec.get("total_trades", 0)
            if total < 3:
                continue   # too few trades to trust

            ps  = rec.get("pattern_strength",      50)
            wr  = rec.get("historical_success_rate", 50.0)
            t   = rec.get("time",      "")
            d   = rec.get("direction", "")

            # Composite ranking score: pattern strength × win_rate / 100
            composite = (ps * wr) / 100.0

            # Bullish / bearish consistency from daily history
            history   = rec.get("daily_history", [])
            wins_     = sum(1 for e in history if e["result"] == "WIN")
            total_h   = len(history)
            bull_cons = round((wins_ / total_h * 100) if total_h > 0 else 50.0, 1)
            bear_cons = round(100.0 - bull_cons, 1)

            rows.append({
                "key":                key,
                "time":               t,
                "direction":          d,
                "pattern_strength":   ps,
                "win_rate":           wr,
                "total_trades":       total,
                "composite_score":    round(composite, 1),
                "bullish_consistency": bull_cons,
                "bearish_consistency": bear_cons,
            })

        rows.sort(key=lambda x: x["composite_score"], reverse=True)
        return rows[:n]

    def get_timing_full_report(self, time_str: str, direction: str) -> dict:
        """
        Return a detailed timing report for Telegram / debugging.

        Extends get_timing_report() with regime breakdown and consistency metrics.
        """
        key = self._key(time_str, direction)
        rec = self._db.get(key, {})

        history   = rec.get("daily_history", [])
        wins_h    = sum(1 for e in history if e["result"] == "WIN")
        total_h   = len(history)
        win_rate  = round((wins_h / total_h * 100) if total_h > 0 else 50.0, 1)

        # Recent 7-day consistency
        cutoff_7d = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        recent    = [e for e in history if e["date"] >= cutoff_7d]
        recent_wr = round((sum(1 for e in recent if e["result"] == "WIN") /
                           len(recent) * 100) if recent else win_rate, 1)

        # Regime breakdown
        regime_detail: dict = {}
        for regime, rd in rec.get("regime_history", {}).items():
            t_ = rd.get("trades", 0)
            w_ = rd.get("wins",   0)
            regime_detail[regime] = {
                "win_rate":     round((w_ / t_ * 100) if t_ > 0 else 50.0, 1),
                "total_trades": t_,
            }

        return {
            "time":               time_str,
            "direction":          direction,
            "pattern_strength":   rec.get("pattern_strength",      50),
            "win_rate":           win_rate,
            "recent_7d_win_rate": recent_wr,
            "total_trades":       total_h,
            "wins":               wins_h,
            "losses":             total_h - wins_h,
            "bullish_consistency": win_rate if direction == "CALL" else 100.0 - win_rate,
            "bearish_consistency": win_rate if direction == "PUT"  else 100.0 - win_rate,
            "regime_breakdown":   regime_detail,
            "last_updated":       rec.get("last_updated", ""),
        }

    def update_from_backtest(self, timing_win_rates: dict) -> None:
        """
        Seed timing_stats.json with historical win rates from the backtesting engine.

        Only updates records that have fewer than 3 trades recorded (i.e., new slots
        with no live trade history yet).  This gives the signal generator a sensible
        starting point for pattern strength on unseen timings.

        Args:
            timing_win_rates: dict["HH:MM|DIR", win_rate_pct] from BacktestResults
        """
        updated = 0
        for key, bt_wr in timing_win_rates.items():
            try:
                time_str, direction = key.split("|")
            except ValueError:
                continue

            rec = self._get_record(time_str, direction)
            if rec["total_trades"] < 3:
                # Seed with backtest win rate (don't overwrite live data)
                rec["historical_success_rate"] = round(float(bt_wr), 1)
                rec["pattern_strength"]        = self._compute_pattern_strength(rec)
                updated += 1

        if updated > 0:
            self._save()
            logger.info("[TimingDB] Seeded %d new timing slots from backtest", updated)

    # ──────────────────────────────────────────────────────
    # Confidence Decay Integration
    # ──────────────────────────────────────────────────────

    @staticmethod
    def apply_confidence_decay(signal: dict, now=None) -> tuple[float, dict]:
        """
        Apply time-based confidence decay to a single signal dict.

        This is a thin wrapper around ConfidenceDecayEngine.apply_to_signal()
        that can be called without importing probability_engine directly.

        The decay is NON-DESTRUCTIVE — stored values (confidence,
        probability_score) are never modified.  Instead the decayed value
        and decay metadata are returned.

        Decay schedule:
            0 – 2 min  → no decay
            2 – 5 min  → gradual decay  (up to –8 pts)
            5 – 10 min → strong decay   (up to –25 pts)
            10+ min    → capped at –25 pts (signal marked 'expired')

        Args:
            signal: dict with 'time' (HH:MM string or datetime) and
                    'confidence' or 'probability_score' key.
            now:    reference time (defaults to datetime.now())

        Returns:
            (decayed_confidence_float, detail_dict)
        """
        try:
            from probability_engine import ConfidenceDecayEngine
            return ConfidenceDecayEngine.apply_to_signal(signal, now=now)
        except Exception as exc:
            logger.warning("[TimingDB] Decay calculation failed (non-critical): %s", exc)
            original = float(signal.get("confidence") or signal.get("probability_score") or 70.0)
            return original, {"decay_zone": "fresh", "points_lost": 0.0, "tier_changed": False}

    @staticmethod
    def get_stale_signals(signals: list, now=None) -> list:
        """
        Return the subset of signals that are in the 'expired' decay zone
        (10+ minutes past their scheduled entry time).

        Args:
            signals: list of signal dicts
            now:     reference time (defaults to datetime.now())

        Returns:
            List of expired signal dicts (sub-list of input).
        """
        try:
            from probability_engine import ConfidenceDecayEngine
            return [s for s in signals if ConfidenceDecayEngine.is_signal_expired(s, now=now)]
        except Exception:
            return []

    @staticmethod
    def cleanup_stale_with_decay(signals: list, now=None) -> tuple[list, list]:
        """
        Filter an active signal list using confidence decay.

        - Signals in 'expired' zone (10+ min old) are removed.
        - Surviving signals are annotated with decay metadata in-place.
        - Forced signals (is_forced=True) bypass the expired check.

        Args:
            signals: list of signal dicts (manager.active_signals)
            now:     reference time (defaults to datetime.now())

        Returns:
            (fresh_signals, removed_signals)
            Both are lists of signal dicts.
        """
        try:
            from probability_engine import ConfidenceDecayEngine
        except Exception:
            return signals, []

        fresh:   list = []
        removed: list = []

        for sig in signals:
            try:
                # Forced signals are immune to stale cleanup
                if sig.get("is_forced"):
                    fresh.append(sig)
                    continue

                decayed, detail = ConfidenceDecayEngine.apply_to_signal(sig, now=now)
                zone = detail.get("decay_zone", "fresh")

                # Annotate with decay info (non-destructive)
                sig["confidence_decayed"]       = decayed
                sig["confidence_decay_zone"]    = zone
                sig["confidence_decay_pts"]     = detail.get("points_lost", 0.0)
                sig["confidence_decay_age_min"] = detail.get("age_minutes", 0.0)

                if zone == "expired":
                    removed.append(sig)
                    logger.info(
                        "[TimingDB] Stale signal removed: %s %s (%.1f min old, conf %s→%.0f)",
                        sig.get("time", "?"), sig.get("direction", "?"),
                        detail.get("age_minutes", 0.0),
                        sig.get("confidence") or sig.get("probability_score"),
                        decayed,
                    )
                else:
                    fresh.append(sig)

            except Exception:
                fresh.append(sig)  # keep on any error

        if removed:
            logger.info(
                "[TimingDB] Stale cleanup: removed %d expired signals, %d remain",
                len(removed), len(fresh),
            )

        return fresh, removed


# ─────────────────────────────────────────────────────────
# Global singleton — importable by signal_generator.py
# ─────────────────────────────────────────────────────────
timing_db = TimingPerformanceDB()
