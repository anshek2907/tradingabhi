import pandas as pd
from typing import Callable, Optional
from datetime import datetime
import json

from logger import logger
from indicators import add_indicators
from market_regime import REGIME_SIDEWAYS, REGIME_REVERSAL_HEAVY
from signal_generator import _classify_live_market
from probability_engine import probability_engine, ProbabilityInputs, TIER_STRONG, TIER_MODERATE
from agreement_engine import agreement_engine, AGREEMENT_SKIP
from sequence_engine import sequence_engine, get_sequence_momentum_bonus
from session_intelligence import session_intel
from signal_generator import _analyse_probability_slot
from market_structure import market_structure_engine

# ==============================================================================
# SWING SIGNAL GENERATOR (15m - 1H trades)
# ==============================================================================

def generate_swing_signals(
    df_15m: pd.DataFrame, 
    df_30m: pd.DataFrame, 
    df_1h: pd.DataFrame
) -> list[dict]:
    """
    Evaluates multi-timeframe alignment for high-probability swing setups.
    
    Returns a list of swing signal dicts:
    [{
       "direction": "CALL"/"PUT",
       "entry": x.xxx,
       "stop_loss": x.xxx,
       "target_1": x.xxx,
       "target_2": x.xxx,
       "confidence": 85,
       "agreement": "7/9",
       "regime": "TRENDING",
       "time": "14:15",
       ...
    }]
    """
    logger.info("Evaluating Multi-Timeframe Swing Signals...")
    signals = []
    
    if df_15m is None or df_30m is None or df_1h is None:
        logger.warning("Missing dataframe(s) for swing generation.")
        return signals
        
    if len(df_15m) < 50 or len(df_30m) < 50 or len(df_1h) < 50:
        logger.warning("Insufficient data length for swing generation.")
        return signals

    # Get latest complete candles
    last_15m = df_15m.iloc[-1]
    last_30m = df_30m.iloc[-1]
    last_1h = df_1h.iloc[-1]
    
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    time_str = now_ist.strftime("%H:%M")
    
    # ── 1. Determine Direction from 1H ──
    dir_1h = "CALL" if last_1h["Close"] > last_1h["EMA50"] and last_1h["EMA50"] > last_1h["EMA200"] else \
             "PUT" if last_1h["Close"] < last_1h["EMA50"] and last_1h["EMA50"] < last_1h["EMA200"] else "NEUTRAL"
             
    if dir_1h == "NEUTRAL":
        logger.debug("Swing Engine: 1H timeframe is not clearly trending.")
        return signals
        
    # ── 2. Confirm with 30m ──
    dir_30m = "CALL" if last_30m["Close"] > last_30m["EMA50"] else \
              "PUT" if last_30m["Close"] < last_30m["EMA50"] else "NEUTRAL"
              
    if dir_1h != dir_30m:
        logger.debug(f"Swing Engine: 1H ({dir_1h}) and 30m ({dir_30m}) timeframes conflict.")
        return signals
        
    # ── 3. Confirm 15m Momentum ──
    mom_15m = "CALL" if last_15m["Close"] > last_15m["Open"] and last_15m["RSI"] > 50 else \
              "PUT" if last_15m["Close"] < last_15m["Open"] and last_15m["RSI"] < 50 else "NEUTRAL"
              
    if dir_1h != mom_15m:
        logger.debug(f"Swing Engine: 15m momentum ({mom_15m}) does not align with trend ({dir_1h}).")
        return signals
        
    direction = dir_1h
    
    # ── 4. Market Regime & Volatility Check (evaluated on 15m for entry precision) ──
    market_profile = _classify_live_market(df_15m)
    regime = market_profile["regime"]
    regime_confidence = market_profile["score"]
    
    if regime_confidence < 70:
        logger.debug(f"Swing Engine: Regime confidence ({regime_confidence:.1f}) < 70.")
        return signals
        
    if regime in (REGIME_SIDEWAYS, REGIME_REVERSAL_HEAVY):
        logger.debug(f"Swing Engine: Market is {regime}, rejecting swing trade.")
        return signals
        
    # Session intelligence
    ist_minutes = now_ist.hour * 60 + now_ist.minute
    session_str, session_detail = session_intel.compute_session_strength(ist_minutes, direction, df_15m)
    
    # Analyze 15m probability metrics
    cutoff = df_15m["datetime_ist"].max() - pd.Timedelta(days=14) if "datetime_ist" in df_15m.columns else None
    df14 = df_15m[df_15m["datetime_ist"] >= cutoff].copy() if cutoff else df_15m
    metrics = _analyse_probability_slot(df_15m.tail(50), df14, direction, session_str)
    
    vol_zone = metrics["volatility_zone"]
    if vol_zone in ("dead", "noisy", "unstable_spike"):
        logger.debug(f"Swing Engine: Rejecting due to weak/unstable volatility ({vol_zone}).")
        return signals
        
    # 4.5 Market Structure & Liquidity
    market_structure = market_structure_engine.analyse(df_1h)
    if direction == "CALL" and market_structure.trend == "BEARISH" and market_structure.recent_choch == "PUT":
        logger.debug(f"Swing Engine: Rejecting due to bearish 1H structure CHOCH.")
        return signals
    if direction == "PUT" and market_structure.trend == "BULLISH" and market_structure.recent_choch == "CALL":
        logger.debug(f"Swing Engine: Rejecting due to bullish 1H structure CHOCH.")
        return signals

    # 5. Sequence & Probability Engine
    seq_result = sequence_engine.analyse(df_15m, direction_hint=direction, market_structure=market_structure)
    seq_bonus = get_sequence_momentum_bonus(seq_result, direction)
    metrics["momentum_strength"] = max(0.0, min(100.0, metrics["momentum_strength"] + seq_bonus * 0.5))
    
    prob_inputs = ProbabilityInputs(
        win_rate              = metrics["win_rate"],
        direction_consistency = metrics["direction_consistency"],
        atr_quality           = metrics["atr_quality"],
        momentum_strength     = metrics["momentum_strength"],
        session_strength      = metrics["session_strength"],
        volatility_quality    = metrics["volatility_quality"],
        reversal_risk         = metrics["reversal_risk"],
        regime                = regime,
        regime_confidence     = float(regime_confidence),
        volatility_zone       = vol_zone,
        time_str              = time_str,
        direction             = direction,
        market_structure      = market_structure,
    )
    prob_result = probability_engine.compute_with_voter_weights(prob_inputs)
    
    if prob_result.probability_score < 75:
        logger.debug(f"Swing Engine: Probability {prob_result.probability_score:.1f} < 75.")
        return signals
        
    # 6. Agreement Engine (Requires >= 8/10)
    agreement = agreement_engine.compute(df_15m, direction, metrics, regime, prob_result, market_structure=market_structure)
    
    if agreement.agreement_score < 8:
        logger.debug(f"Swing Engine: Agreement {agreement.agreement_score}/{agreement.total_voters} < 8/10.")
        return signals
        
    # ── 7. Calculate Entry, Stop Loss, Targets using 1H ATR ──
    entry_price = float(last_15m["Close"])
    atr_1h = float(last_1h["ATR"]) if not pd.isna(last_1h["ATR"]) else entry_price * 0.001
    
    if direction == "CALL":
        stop_loss = entry_price - atr_1h
        target_1 = entry_price + atr_1h
        target_2 = entry_price + (atr_1h * 2)
    else:
        stop_loss = entry_price + atr_1h
        target_1 = entry_price - atr_1h
        target_2 = entry_price - (atr_1h * 2)
        
    signal = {
        "direction": direction,
        "entry": round(entry_price, 5),
        "stop_loss": round(stop_loss, 5),
        "target_1": round(target_1, 5),
        "target_2": round(target_2, 5),
        "confidence": int(prob_result.probability_score),
        "agreement": f"{agreement.agreement_score}/{agreement.total_voters}",
        "regime": regime,
        "time": time_str
    }
    
    signals.append(signal)
    logger.info(f"🟢 SWING GENERATED: {direction} | Conf: {signal['confidence']}% | Agr: {signal['agreement']} | Reg: {regime}")
    return signals
