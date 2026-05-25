"""
backtesting_engine.py
─────────────────────
Advanced 14-Day Historical Backtesting Engine.

Replays the last 14 days of enriched historical price data to compute:
  • Overall win rate across all time slots
  • Regime-segmented win rates (TRENDING / SIDEWAYS / HIGH_VOLATILITY / REVERSAL_HEAVY)
  • Per-timing win rates ("HH:MM|CALL" and "HH:MM|PUT")
  • Best performing timings (win rate ≥ 70%)
  • Weak/failing timings   (win rate < 45%)
  • Continuation success rate (momentum continuation quality)
  • Reversal failure rate   (how often reversals hurt signals)
  • Volatility quality correlation with trade outcomes

Results are persisted to backtest_results.json and used to:
  • Feed DynamicWeightOptimizer for probability weight adjustment
  • Seed TimingPerformanceDB with historically-grounded pattern strengths
  • Provide regime-specific performance data for adaptive threshold adjustment

Usage:
    from backtesting_engine import BacktestingEngine
    engine = BacktestingEngine()
    results = engine.run(df)     # df is the 14-day enriched DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from logger import logger
from persistence import safe_load_json, safe_save_json

BACKTEST_FILE   = "backtest_results.json"
MIN_SLOT_TRADES = 5   # minimum candles per slot to include in stats


@dataclass
class BacktestResults:
    """
    Comprehensive backtesting output.

    All win rates are percentages (0–100).
    timing_win_rates keys are "HH:MM|CALL" or "HH:MM|PUT".
    """
    overall_win_rate:         float = 50.0
    total_simulated:          int   = 0
    regime_win_rates:         Dict[str, float] = field(default_factory=dict)
    timing_win_rates:         Dict[str, float] = field(default_factory=dict)
    best_timings:             List[str]        = field(default_factory=list)
    weak_timings:             List[str]        = field(default_factory=list)
    continuation_success:     float = 50.0
    reversal_failure_rate:    float = 50.0
    volatility_quality_score: float = 50.0
    top_regime:               str   = "UNKNOWN"
    top_regime_win_rate:      float = 50.0
    timing_details:           List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "overall_win_rate":         round(self.overall_win_rate,         2),
            "total_simulated":          self.total_simulated,
            "regime_win_rates":         {k: round(v, 2) for k, v in self.regime_win_rates.items()},
            "timing_win_rates":         {k: round(v, 2) for k, v in self.timing_win_rates.items()},
            "best_timings":             self.best_timings,
            "weak_timings":             self.weak_timings,
            "continuation_success":     round(self.continuation_success,     2),
            "reversal_failure_rate":    round(self.reversal_failure_rate,    2),
            "volatility_quality_score": round(self.volatility_quality_score, 2),
            "top_regime":               self.top_regime,
            "top_regime_win_rate":      round(self.top_regime_win_rate,      2),
        }

    def format_summary(self) -> str:
        """One-paragraph summary suitable for logging or Telegram."""
        lines = [
            f"📈 BACKTEST SUMMARY (14-day)",
            f"Overall Win Rate:    {self.overall_win_rate:.1f}%",
            f"Best Regime:         {self.top_regime} ({self.top_regime_win_rate:.1f}%)",
            f"Continuation Rate:   {self.continuation_success:.1f}%",
            f"Reversal Fail Rate:  {self.reversal_failure_rate:.1f}%",
            f"Volatility Quality:  {self.volatility_quality_score:.1f}",
            f"Best Timings:        {len(self.best_timings)}",
            f"Weak Timings:        {len(self.weak_timings)}",
        ]
        return "\n".join(lines)


class BacktestingEngine:
    """
    14-day historical replay backtesting engine.

    Does NOT make any API calls — it operates entirely on the pre-fetched,
    enriched DataFrame from get_historical_data().  This keeps the daily
    API budget unchanged.
    """

    def __init__(self, results_file: str = BACKTEST_FILE):
        self.results_file = results_file

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> BacktestResults:
        """
        Run backtesting on the enriched 14-day DataFrame.

        Steps:
          1. Restrict to last 14 days and IST session (13:00–22:00)
          2. For each time slot × direction:
               a. Compute directional win rate from Result_CALL / Result_PUT
               b. Approximate regime using EMA + ATR + reversal heuristics
               c. Aggregate regime-specific win rates
          3. Compute continuation success / reversal failure / volatility quality
          4. Rank timings into best / weak lists
          5. Save to backtest_results.json

        Args:
            df: Full enriched DataFrame (must have TimeOfDay, datetime_ist,
                Result_CALL, Result_PUT, EMA50, EMA200, ATR, RSI, etc.)

        Returns:
            BacktestResults with all computed stats.
        """
        if df is None or len(df) < 200:
            logger.warning("[Backtest] Insufficient data (%d rows) — skipping",
                           0 if df is None else len(df))
            return BacktestResults()

        # Ensure IST columns exist
        if "TimeOfDay" not in df.columns or "datetime_ist" not in df.columns:
            logger.warning("[Backtest] Missing TimeOfDay/datetime_ist columns — skipping")
            return BacktestResults()

        # ── Restrict to 14 days ─────────────────────────────────────────
        cutoff = df["datetime_ist"].max() - pd.Timedelta(days=14)
        df14   = df[df["datetime_ist"] >= cutoff].copy()

        if len(df14) < 100:
            logger.warning("[Backtest] Insufficient 14-day data (%d rows) — skipping", len(df14))
            return BacktestResults()

        # ── IST session filter: 13:00 – 22:00 ──────────────────────────
        session_times: list[str] = []
        for t_str in sorted(df14["TimeOfDay"].unique()):
            try:
                h, m = map(int, t_str.split(":"))
                if 13 * 60 <= h * 60 + m <= 22 * 60:
                    session_times.append(t_str)
            except ValueError:
                continue

        # ── Per-regime buckets ───────────────────────────────────────────
        REGIMES = ["TRENDING", "SIDEWAYS", "HIGH_VOLATILITY", "REVERSAL_HEAVY"]
        regime_buckets: dict[str, dict] = {
            r: {"wins": 0, "total": 0} for r in REGIMES
        }

        timing_win_rates: dict[str, float] = {}
        timing_details:   list[dict]       = []

        all_wins  = 0
        all_total = 0

        # Continuation / reversal / vol quality aggregation
        cont_wins   = cont_total   = 0
        rev_fails   = rev_total    = 0
        vol_q_sum   = 0.0
        vol_q_count = 0

        for t_str in session_times:
            slot = df14[df14["TimeOfDay"] == t_str]
            if len(slot) < MIN_SLOT_TRADES:
                continue

            # Approximate regime for this slot
            slot_regime = self._classify_slot_regime(slot)

            for direction in ("CALL", "PUT"):
                col = "Result_CALL" if direction == "CALL" else "Result_PUT"
                if col not in slot.columns:
                    continue

                wins_arr = slot[col]
                wins_n   = int(wins_arr.sum())
                n        = len(slot)
                win_rate = wins_n / n * 100

                key = f"{t_str}|{direction}"
                timing_win_rates[key] = round(win_rate, 2)

                all_wins  += wins_n
                all_total += n

                regime_buckets[slot_regime]["wins"]  += wins_n
                regime_buckets[slot_regime]["total"] += n

                # ── Continuation stats ─────────────────────────────────
                mom_col = "Mom_Cont_CALL" if direction == "CALL" else "Mom_Cont_PUT"
                if mom_col in slot.columns:
                    mc = int(slot[mom_col].sum())
                    cont_wins  += mc
                    cont_total += n

                # ── Reversal failure ───────────────────────────────────
                if "Reversal" in slot.columns:
                    rev = int(slot["Reversal"].sum())
                    rev_fails += rev
                    rev_total += n

                # ── Volatility quality ─────────────────────────────────
                if "Healthy_Volatility" in slot.columns:
                    vq = float(slot["Healthy_Volatility"].mean() * 100)
                    vol_q_sum   += vq
                    vol_q_count += 1

                timing_details.append({
                    "key":       key,
                    "time":      t_str,
                    "direction": direction,
                    "win_rate":  round(win_rate, 2),
                    "n":         n,
                    "regime":    slot_regime,
                })

        # ── Aggregate stats ──────────────────────────────────────────────
        overall_wr  = (all_wins / all_total * 100) if all_total > 0 else 50.0
        cont_rate   = (cont_wins / cont_total * 100) if cont_total > 0 else 50.0
        rev_rate    = (rev_fails / rev_total  * 100) if rev_total  > 0 else 50.0
        vol_quality = (vol_q_sum / vol_q_count)      if vol_q_count > 0 else 50.0

        # ── Regime win rates ─────────────────────────────────────────────
        regime_wr: dict[str, float] = {}
        top_regime    = "UNKNOWN"
        top_regime_wr = 0.0
        for reg, bucket in regime_buckets.items():
            if bucket["total"] > 0:
                wr = bucket["wins"] / bucket["total"] * 100
                regime_wr[reg] = round(wr, 1)
                if wr > top_regime_wr:
                    top_regime_wr = wr
                    top_regime    = reg
            else:
                regime_wr[reg] = 50.0

        # ── Best / weak timings ──────────────────────────────────────────
        timing_details.sort(key=lambda x: x["win_rate"], reverse=True)
        best_timings = [
            d["key"] for d in timing_details
            if d["win_rate"] >= 70.0 and d["n"] >= MIN_SLOT_TRADES
        ][:15]
        weak_timings = [
            d["key"] for d in timing_details
            if d["win_rate"] < 45.0 and d["n"] >= MIN_SLOT_TRADES
        ][:15]

        results = BacktestResults(
            overall_win_rate         = round(overall_wr,  2),
            total_simulated          = all_total,
            regime_win_rates         = regime_wr,
            timing_win_rates         = timing_win_rates,
            best_timings             = best_timings,
            weak_timings             = weak_timings,
            continuation_success     = round(cont_rate,   2),
            reversal_failure_rate    = round(rev_rate,    2),
            volatility_quality_score = round(vol_quality, 2),
            top_regime               = top_regime,
            top_regime_win_rate      = round(top_regime_wr, 2),
            timing_details           = timing_details,
        )

        # ── Save results ─────────────────────────────────────────────────
        try:
            safe_save_json(self.results_file, results.as_dict())
            logger.info(
                "[Backtest] ✅ Complete | overall=%.1f%% | best_regime=%s (%.1f%%) | "
                "cont=%.1f%% | rev_fail=%.1f%% | vol_q=%.1f | "
                "best_timings=%d weak=%d total_candles=%d",
                overall_wr, top_regime, top_regime_wr,
                cont_rate, rev_rate, vol_quality,
                len(best_timings), len(weak_timings), all_total,
            )
        except Exception as e:
            logger.error("[Backtest] Failed to save results: %s", e)

        return results

    def load_last_results(self) -> dict:
        """Load the last saved backtest results from disk."""
        return safe_load_json(self.results_file, default={})

    def get_is_best_timing(self, time_str: str, direction: str) -> bool:
        """Return True if this timing was a best performer in the last backtest."""
        data = self.load_last_results()
        key  = f"{time_str}|{direction}"
        return key in data.get("best_timings", [])

    def get_is_weak_timing(self, time_str: str, direction: str) -> bool:
        """Return True if this timing was a weak performer in the last backtest."""
        data = self.load_last_results()
        key  = f"{time_str}|{direction}"
        return key in data.get("weak_timings", [])

    def get_timing_backtest_win_rate(
        self, time_str: str, direction: str
    ) -> Optional[float]:
        """Return the historical win rate for a timing from last backtest, or None."""
        data = self.load_last_results()
        key  = f"{time_str}|{direction}"
        rates = data.get("timing_win_rates", {})
        return rates.get(key)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_slot_regime(slot: pd.DataFrame) -> str:
        """
        Fast heuristic regime approximation for a time slot.

        Uses EMA relationship, ATR coefficient of variation, and reversal frequency.
        Does NOT call detect_market_regime() (too slow per slot).

        Returns one of: TRENDING | SIDEWAYS | HIGH_VOLATILITY | REVERSAL_HEAVY
        """
        try:
            # ATR coefficient of variation
            if "ATR" in slot.columns:
                atr_vals = pd.to_numeric(slot["ATR"], errors="coerce").dropna()
                atr_avg  = float(atr_vals.mean())  if len(atr_vals) > 0 else 0.0
                atr_std  = float(atr_vals.std())   if len(atr_vals) > 1 else 0.0
                atr_cv   = atr_std / max(atr_avg, 1e-8)
            else:
                atr_avg, atr_cv = 0.0, 0.0

            # Reversal frequency
            if "Reversal" in slot.columns:
                rev_freq = float(slot["Reversal"].mean())
            else:
                rev_freq = 0.4

            # EMA separation
            if "EMA50" in slot.columns and "EMA200" in slot.columns:
                ema50_mean  = float(slot["EMA50"].mean())
                ema200_mean = float(slot["EMA200"].mean())
                ema_diff    = abs(ema50_mean - ema200_mean)
            else:
                ema_diff = 0.0

            # ── Classification logic ───────────────────────────────────
            # High volatility: large ATR variance
            if atr_cv > 0.40:
                return "HIGH_VOLATILITY"

            # Reversal heavy: most candles reverse the prior direction
            if rev_freq > 0.55:
                return "REVERSAL_HEAVY"

            # Trending: clear EMA separation + low reversal frequency
            if ema_diff > atr_avg * 0.40 and rev_freq < 0.42:
                return "TRENDING"

            return "SIDEWAYS"

        except Exception:
            return "SIDEWAYS"


# ── Module-level singleton ─────────────────────────────────────────────────
backtesting_engine = BacktestingEngine()
