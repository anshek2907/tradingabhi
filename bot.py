from logger import logger
import json
import os
import time
from datetime import time as clock_time

import pandas as pd
import requests

# ONLY detect Railway (not BOT_TOKEN)
if os.getenv("RAILWAY_ENVIRONMENT"):
    from config_prod import BOT_TOKEN, CHAT_ID, TD_API_KEY
else:
    from config_local import BOT_TOKEN, CHAT_ID, TD_API_KEY

from fixed_trade import get_fixed_signal
from forex_trade import get_forex_signal
from indicators import add_indicators, calculate_score
from signal_list import (
    apply_signal_text,
    get_adaptive_trade_threshold,
    manager as signal_manager,
    process_signal_list,
    should_force_fast_mode,
    store_tracked_signal,
    TRADE_LOCK_SECONDS,
    update_signal_list,
)
from market_safety import run_market_safety
from cache_manager import cache

PAIR = "EUR/USD"
SLEEP_TIME = 120   # 2 minutes
IDLE_SLEEP_TIME = 900   # 15 minutes during weekends/off-market hours
MARKET_OPEN_TIME = clock_time(13, 0)
MARKET_CLOSE_TIME = clock_time(22, 0)  # London/NY Prime ONLY (Aligned with generator)
NEWS_BLOCK_MINUTES = 15
TRADE_COOLDOWN_MINUTES = 20
HIGH_IMPACT_NEWS_EVENTS = [
    # Add high-impact event times in Asia/Kolkata timezone.
    # Example: "2026-05-03 18:00"
]
MAX_SIGNAL_MESSAGES_PER_CYCLE = 10

LAST_SIGNAL_INPUT_UPDATE_ID = None

# API rate limiting
API_CALL_TIMES = []
API_CALLS_PER_MINUTE_LIMIT = 5
_rate_limit_pause_until = 0.0  # epoch seconds; set on 429 for 90-second cooldown


def track_api_call():
    """Track API call timestamp for rate limiting."""
    now = time.time()
    API_CALL_TIMES.append(now)
    # Remove calls older than 60 seconds
    API_CALL_TIMES[:] = [t for t in API_CALL_TIMES if now - t < 60]


SKIPPED_API_CALLS_COUNT = 0

def is_api_call_allowed(interval_name: str = "API"):
    """Check if we can make another API call within rate limit."""
    global SKIPPED_API_CALLS_COUNT
    now = time.time()
    
    # Periodically clean up old timestamps to avoid long-term memory growth
    API_CALL_TIMES[:] = [t for t in API_CALL_TIMES if now - t < 60]
    
    if now < _rate_limit_pause_until:
        remaining = int(_rate_limit_pause_until - now)
        SKIPPED_API_CALLS_COUNT += 1
        logger.info(f"Emergency cooldown active — skipping {interval_name} call ({remaining}s remaining, total skipped: {SKIPPED_API_CALLS_COUNT})")
        return False
    
    allowed = len(API_CALL_TIMES) < API_CALLS_PER_MINUTE_LIMIT
    
    if not allowed:
        SKIPPED_API_CALLS_COUNT += 1
        logger.info(f"Rate limit reached — skipping {interval_name} call (total skipped: {SKIPPED_API_CALLS_COUNT})")
        
    return allowed


if not TD_API_KEY:
    raise ValueError("TD_API_KEY is missing or empty")


def is_market_open():
    market_open, _ = get_market_status()
    return market_open


def get_market_status(now=None):
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")

    if now.weekday() >= 5:
        return False, "weekend"

    current_time = now.time()

    if current_time < MARKET_OPEN_TIME or current_time > MARKET_CLOSE_TIME:
        return False, "closed"

    return True, "open"


def get_next_market_open(now=None):
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")

    candidate = now.normalize() + pd.Timedelta(
        hours=MARKET_OPEN_TIME.hour,
        minutes=MARKET_OPEN_TIME.minute
    )

    if now.weekday() < 5 and now < candidate:
        return candidate

    candidate += pd.Timedelta(days=1)

    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)

    return candidate


def get_idle_sleep_seconds(now=None):
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")

    seconds_until_open = (get_next_market_open(now) - now).total_seconds()
    return max(1, int(min(IDLE_SLEEP_TIME, seconds_until_open)))


def is_near_candle_close():
    now = pd.Timestamp.now(tz="Asia/Kolkata")
    return now.second >= 45


# get_current_candle_key moved to cache_manager.py


def is_high_impact_news_window(now=None):
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")

    for event_time in HIGH_IMPACT_NEWS_EVENTS:
        event = pd.Timestamp(event_time)

        if event.tzinfo is None:
            event = event.tz_localize("Asia/Kolkata")
        else:
            event = event.tz_convert("Asia/Kolkata")

        minutes_from_event = abs((now - event).total_seconds()) / 60

        if minutes_from_event <= NEWS_BLOCK_MINUTES:
            return True, event

    return False, None


# ==============================
# TELEGRAM
# ==============================

_TELEGRAM_MAX_RETRIES = 3
_TELEGRAM_RETRY_DELAY = 2       # seconds between retries
_TELEGRAM_TIMEOUT = 15          # seconds per request

# HTTP status codes that are worth retrying (transient server-side issues)
_TELEGRAM_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def send_telegram(msg):
    """
    Send a Telegram message with up to 3 automatic retries on transient failures.

    Retries on:
      - requests.Timeout / requests.ConnectionError  (network-level)
      - HTTP 429 (rate limit) and 5xx (server error) responses

    Permanent 4xx errors (except 429) fail immediately — no retry.
    Never raises; failures are logged and silently discarded to keep the bot alive.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
    }

    last_error = None
    for attempt in range(1, _TELEGRAM_MAX_RETRIES + 1):
        try:
            res = requests.post(url, data=payload, timeout=_TELEGRAM_TIMEOUT)

            if res.status_code == 200:
                logger.info("Telegram message sent successfully")
                return

            if res.status_code in _TELEGRAM_RETRYABLE_HTTP:
                last_error = f"HTTP {res.status_code}: {res.text[:120]}"
                if attempt < _TELEGRAM_MAX_RETRIES:
                    logger.warning(f"Telegram retry {attempt}/{_TELEGRAM_MAX_RETRIES} — {last_error}")
                    time.sleep(_TELEGRAM_RETRY_DELAY)
                continue  # retry

            # Non-retryable HTTP error (e.g. 400 Bad Request, 401 Unauthorized)
            logger.error(f"Telegram permanent error HTTP {res.status_code}: {res.text[:200]}")
            return

        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = str(e)
            if attempt < _TELEGRAM_MAX_RETRIES:
                logger.warning(f"Telegram retry {attempt}/{_TELEGRAM_MAX_RETRIES} — {type(e).__name__}: {e}")
                time.sleep(_TELEGRAM_RETRY_DELAY)

        except Exception as e:
            # Unexpected error — log and give up immediately
            logger.error(f"Telegram unexpected error: {e}")
            return

    logger.error("Telegram delivery failed after retries")


# ==============================
# DAILY SIGNAL LIST SENDER
# ==============================

_daily_signal_list_sent_date = None   # tracks which calendar date the list was sent

GENERATED_SIGNALS_PATH = os.path.join(os.path.dirname(__file__), "generated_signals.json")
DAILY_SEND_HOUR = 10    # 10:00 AM IST
DAILY_SEND_MINUTE = 0


def _filter_generated_signals_for_today(signals, now):
    """Keep only fresh, correctly dated IST generated signals."""
    if not isinstance(signals, list):
        return []

    today = now.date().isoformat()
    file_date = None
    try:
        file_date = pd.Timestamp(os.path.getmtime(GENERATED_SIGNALS_PATH), unit="s", tz="UTC").tz_convert("Asia/Kolkata").date().isoformat()
    except Exception:
        file_date = None
    fresh = []
    for sig in signals:
        try:
            direction = str(sig.get("direction", "")).upper()
            if direction not in ("CALL", "PUT"):
                continue

            generated_date = sig.get("generated_date")
            if generated_date and str(generated_date) != today:
                continue
            if not generated_date and file_date is not None and file_date != today:
                continue

            h, m = map(int, str(sig.get("time", "")).split(":"))
            signal_time = now.normalize() + pd.Timedelta(hours=h, minutes=m)
            if signal_time < now:
                continue

            sig["direction"] = direction
            sig["pair"] = str(sig.get("pair", "EURUSD")).replace("/", "")
            fresh.append(sig)
        except Exception:
            continue

    return sorted(fresh, key=lambda x: x.get("time", ""))


def maybe_send_daily_signal_list():
    """
    Send generated_signals.json to Telegram once per day at 10:00 AM IST.
    Sends TWO separate messages:
      Message 1 — CALL signals (skipped if empty)
      Message 2 — PUT signals  (skipped if empty)
    Prevents duplicates with a per-day lock that resets at midnight.
    """
    global _daily_signal_list_sent_date

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now.date()

    # Reset guard on new calendar day
    if _daily_signal_list_sent_date is not None and _daily_signal_list_sent_date != today:
        _daily_signal_list_sent_date = None

    # Already sent today — skip
    if _daily_signal_list_sent_date == today:
        return

    # Only send at / after 10:00 AM IST
    if now.hour < DAILY_SEND_HOUR or (now.hour == DAILY_SEND_HOUR and now.minute < DAILY_SEND_MINUTE):
        return

    # Load signal list
    try:
        with open(GENERATED_SIGNALS_PATH, "r") as f:
            signals = json.load(f)
    except FileNotFoundError:
        logger.warning("generated_signals.json not found — skipping daily send")
        _daily_signal_list_sent_date = today
        return
    except Exception as e:
        logger.error(f"Failed to read generated_signals.json: {e}")
        return

    if not signals:
        logger.info("Daily signal list is empty — skipping send")
        _daily_signal_list_sent_date = today
        return

    signals = _filter_generated_signals_for_today(signals, now)
    if not signals:
        logger.info("No fresh generated signals for today — skipping daily send")
        _daily_signal_list_sent_date = today
        return

    try:
        with open(GENERATED_SIGNALS_PATH, "w", encoding="utf-8") as f:
            json.dump(signals, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist generated signal cleanup: {e}")

    # Separate CALL and PUT (exclude forced martingale duplicates in display)
    call_sigs = [s for s in signals if s.get("direction") == "CALL"]
    put_sigs  = [s for s in signals if s.get("direction") == "PUT"]

    def _fmt_signal(s: dict, direction: str) -> str:
        t = s.get("time", "??:??")
        pair = str(s.get("pair", "EURUSD")).replace("/", "")
        return f"{t} {pair} {direction}"

    lines = []

    if call_sigs:
        lines.append("📊 TODAY GENERATED CALL SIGNALS")
        for s in sorted(call_sigs, key=lambda x: x.get("time", "")):
            lines.append(_fmt_signal(s, "CALL"))

    if put_sigs:
        if lines:
            lines.append("")
        lines.append("📊 TODAY GENERATED PUT SIGNALS")
        for s in sorted(put_sigs, key=lambda x: x.get("time", "")):
            lines.append(_fmt_signal(s, "PUT"))

    if not lines:
        logger.info("No CALL or PUT signals to send today.")
    else:
        logger.info(f"Sending generated daily list to Telegram ({len(signals)} signals)")
        send_telegram("\n".join(lines))

    _daily_signal_list_sent_date = today


_eod_results_sent_date = None
EOD_SEND_HOUR = 23
EOD_SEND_MINUTE = 30

def maybe_send_eod_results():
    global _eod_results_sent_date

    now = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now.date()

    if _eod_results_sent_date is not None and _eod_results_sent_date != today:
        _eod_results_sent_date = None

    if _eod_results_sent_date == today:
        return

    if now.hour < EOD_SEND_HOUR or (now.hour == EOD_SEND_HOUR and now.minute < EOD_SEND_MINUTE):
        return

    try:
        with open(GENERATED_SIGNALS_PATH, "r") as f:
            signals = json.load(f)
    except Exception as e:
        _eod_results_sent_date = today
        return

    if not signals:
        _eod_results_sent_date = today
        return

    signals = [
        s for s in signals
        if not s.get("generated_date") or str(s.get("generated_date")) == today.isoformat()
    ]
    if not signals:
        _eod_results_sent_date = today
        return

    from signal_manager import timing_db
    manager = signal_manager
    
    # We load if not loaded, though main loop likely loaded it
    if not manager._initialized:
        manager.load()
        manager._initialized = True

    resolved_today = [
        t for t in manager.tracked_trades 
        if t.get("resolved") and t.get("signal_time") is not None and t["signal_time"].date() == today
    ]

    lines = ["📊 GENERATED SIGNAL RESULTS"]
    sent_any = False
    timing_results = []

    for s in sorted(signals, key=lambda x: x.get("time", "")):
        t_str = s.get("time", "")
        direction = s.get("direction", "")
        
        found_result = None
        for rt in resolved_today:
            # Match time and direction
            if rt["signal_time"].strftime("%H:%M") == t_str and rt.get("direction") == direction and rt.get("source") in ("generated", "forced"):
                found_result = rt.get("result")
                break
                
        if found_result:
            timing_results.append({
                "time": t_str,
                "direction": direction,
                "result": found_result
            })

        if found_result == "WIN":
            pair = str(s.get("pair", "EURUSD")).replace("/", "")
            lines.append(f"{t_str} {pair} {direction} ✅ WIN")
            sent_any = True
        elif found_result == "LOSS":
            pair = str(s.get("pair", "EURUSD")).replace("/", "")
            lines.append(f"{t_str} {pair} {direction} ❌ LOSS")
            sent_any = True

    if timing_results:
        timing_db.run_end_of_day_update(timing_results)

    if sent_any:
        send_telegram("\n".join(lines))
        logger.info("Sent EOD generated signal results.")

    _eod_results_sent_date = today


def fetch_signal_text_from_telegram():
    global LAST_SIGNAL_INPUT_UPDATE_ID

    try:
        params = {"timeout": 5}
        if LAST_SIGNAL_INPUT_UPDATE_ID is not None:
            params["offset"] = LAST_SIGNAL_INPUT_UPDATE_ID + 1

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        res = requests.get(url, params=params, timeout=20).json()

        if not res.get("ok"):
            return None

        updates = res.get("result", [])
        if not updates:
            return None

        latest_text = None
        max_update_id = LAST_SIGNAL_INPUT_UPDATE_ID

        for update in updates:
            update_id = update.get("update_id")
            if not isinstance(update_id, int):
                continue

            if max_update_id is None or update_id > max_update_id:
                max_update_id = update_id

            message = update.get("message") or update.get("edited_message") or {}
            chat_id = str(message.get("chat", {}).get("id", ""))

            if chat_id and str(CHAT_ID) != chat_id:
                continue

            text = (message.get("text") or "").strip()
            if text:
                latest_text = text

        LAST_SIGNAL_INPUT_UPDATE_ID = max_update_id
        return latest_text

    except Exception as e:
        logger.error(f"Telegram input error: {e}")
        return None


# ==============================
# DATA FETCH
# ==============================
def get_data(interval):
    if not is_api_call_allowed(interval):
        return None

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": PAIR,
        "interval": interval,
        "apikey": TD_API_KEY,
        "outputsize": 200
    }

    res = requests.get(url, params=params).json()
    track_api_call()

    if "values" not in res:
        logger.error(f"API ERROR: {res}")

        if res.get("code") == 429:
            global _rate_limit_pause_until
            _rate_limit_pause_until = time.time() + 90
            logger.warning("429 received — emergency cooldown: all fetches paused for 90 seconds.")
            return None

        logger.error(f"*API Error* {res}")
        return None

    df = pd.DataFrame(res["values"])

    if "datetime" in df.columns:
        df["CandleTime"] = pd.to_datetime(df["datetime"])

    price_columns = ["open", "high", "low", "close"]
    df[price_columns] = df[price_columns].astype(float)
    df = df.iloc[::-1]

    df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)

    return df


def run_external_signal_engine(df, cached_minute_df=None):
    """Run external signal processing, optionally using cached minute data."""
    # Ensure we always pass a usable `df` into process_signal_list so external
    # signal processing runs even when the auto-bot skipped or df is small.
    # Prefer provided `df`, then cached_minute_df, then attempt a fresh 5min fetch.
    if df is None or (hasattr(df, "__len__") and len(df) < 200):
        fallback_df = None
        if cached_minute_df is not None and len(cached_minute_df) >= 200:
            fallback_df = cached_minute_df
        elif is_api_call_allowed("fallback 5min"):
            try:
                fallback_df = get_data("5min")
            except Exception:
                fallback_df = None

        if fallback_df is not None and len(fallback_df) >= 200:
            df = fallback_df

    if cached_minute_df is not None:
        minute_data_fetcher = lambda: cached_minute_df
    else:
        minute_data_fetcher = lambda: get_data("1min") if is_api_call_allowed("1min") else None

    signal_messages = process_signal_list(df, minute_data_fetcher=minute_data_fetcher)
    for signal_message in signal_messages[:MAX_SIGNAL_MESSAGES_PER_CYCLE]:
        send_telegram(signal_message)


# ==============================
# MAIN LOOP
# ==============================
def run():
    logger.info("BOT RUNNING")

    last_signal_time = None
    last_trade_time = None

    while True:
        try:
            # 0) Send daily signal list at 10:00 AM (once per day).
            maybe_send_daily_signal_list()
            maybe_send_eod_results()

            # 1) Update external signal list first.
            message_text = fetch_signal_text_from_telegram()
            if message_text:
                apply_signal_text(message_text)
            else:
                update_signal_list()

            # 2) Check market status.
            market_open, _ = get_market_status()

            if not market_open:
                logger.info("Market closed — idle mode")
                logger.info(f"Next market open: {get_next_market_open():%Y-%m-%d %H:%M %Z}")
                cache.cleanup_stale_cache()
                time.sleep(get_idle_sleep_seconds())
                continue

            logger.info("Market Open — Running")

            force_fast_mode = should_force_fast_mode()

            if force_fast_mode:
                interval = "1min"
                sleep_time = 20
            else:
                interval = "5min"
                sleep_time = SLEEP_TIME

            df = cache.get_processed_dataframe(interval, get_data, add_indicators)

            # 3) Fetch df is complete above.
            if df is None:
                time.sleep(sleep_time)
                continue

            # Fetch 1-minute data using cache manager
            cached_minute_df = cache.get_dataframe("1min", get_data)

            # 4) Run auto bot logic.
            # df is already processed with indicators
            confidence, grade = calculate_score(df)
            trade_threshold = get_adaptive_trade_threshold(75)

            # Debug: print final confidence used by bot decision path
            logger.debug(f"FINAL CONFIDENCE USED: {confidence}%")

            if confidence < trade_threshold:
                logger.info(f"Signal rejected - confidence too low: {confidence}% (threshold {trade_threshold}%)")
                # 5) Run external signal processing after auto bot.
                run_external_signal_engine(df, cached_minute_df)
                time.sleep(sleep_time)
                continue

            fixed = get_fixed_signal(df)
            current_candle_time = df.iloc[-1].get("CandleTime", df.index[-1])

            if fixed:
                # NEW: Market Safety Check for Auto-Bot
                safety_ok, safety_msg, penalty = run_market_safety(df, fixed['signal'])
                
                # Apply penalty to confidence
                confidence -= penalty
                
                if not safety_ok or confidence < trade_threshold:
                    reason = safety_msg if not safety_ok else f"Confidence dropped below threshold after safety penalty ({confidence}% < {trade_threshold}%)"
                    logger.info(f"Auto-trade rejected: {reason}")
                    run_external_signal_engine(df, cached_minute_df)
                    time.sleep(sleep_time)
                    continue
                now = pd.Timestamp.now(tz="Asia/Kolkata")

                if last_trade_time is not None:
                    cooldown_minutes = (now - last_trade_time).total_seconds() / 60

                    if cooldown_minutes < TRADE_COOLDOWN_MINUTES:
                        remaining = TRADE_COOLDOWN_MINUTES - cooldown_minutes
                        logger.info(f"Trade cooldown active - {remaining:.1f} min remaining")
                        run_external_signal_engine(df, cached_minute_df)
                        time.sleep(sleep_time)
                        continue

                # --- Global trade lock check (auto-trade) ---
                if signal_manager.last_confirmed_trade_time is not None:
                    elapsed = (now - signal_manager.last_confirmed_trade_time).total_seconds()
                    if elapsed < TRADE_LOCK_SECONDS:
                        remaining_lock = int(TRADE_LOCK_SECONDS - elapsed)
                        logger.warning(
                            f"Trade lock active — blocking auto-trade "
                            f"({remaining_lock}s remaining). Trade skipped."
                        )
                        run_external_signal_engine(df, cached_minute_df)
                        time.sleep(sleep_time)
                        continue

                news_blocked, news_time = is_high_impact_news_window()

                if news_blocked:
                    logger.info(f"Trade blocked due to high-impact news at {news_time:%H:%M}")
                    run_external_signal_engine(df, cached_minute_df)
                    time.sleep(sleep_time)
                    continue

                if current_candle_time == last_signal_time:
                    logger.debug("Duplicate signal skipped")
                    run_external_signal_engine(df, cached_minute_df)
                    time.sleep(sleep_time)
                    continue

                logger.info(f"Score: {confidence} {grade}")

                send_telegram(f"""
*PRE-SIGNAL*

*Market:* EUR/USD
*Signal:* {fixed['signal']}

*Score*
Grade: {grade}
Confidence: {confidence}%

*Fixed Trade*
Entry: {fixed['entry']}
Starts In: {fixed['seconds_left']//60} min

Waiting for confirmation.
""")

                forex = get_forex_signal(df, fixed["signal"], confidence)

                confirm = fixed

                msg = f"""
*CONFIRMED SIGNAL*

*Market:* EUR/USD
*Direction:* {forex['direction']}

*Score*
Grade: {grade}
Confidence: {confidence}%

*Fixed Trade*
Entry: {confirm['entry']}
Expiry: {confirm['expiry']}

*Forex Trade*
Entry: {forex['entry']}
Entry Window: {forex['entry_window']}
Hold: {forex['hold']}

*Risk Targets*
TP: {forex['tp']}
SL: {forex['sl']}
Multiplier: {forex['multiplier']}
Auto Close: {forex['auto_close']}
"""

                logger.info(msg)
                send_telegram(msg)
                try:
                    entry_price = float(df.iloc[-1]["Close"])
                    expiry_time = pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(minutes=5)

                    store_tracked_signal(
                        signal_time=pd.Timestamp.now(tz="Asia/Kolkata"),
                        direction=forex["direction"],
                        entry_price=entry_price,
                        expiry_time=expiry_time,
                        signal_type="auto_trade",
                        pair="EURUSD",
                        confidence=confidence,
                        df=df,
                        source="auto",
                    )
                except Exception as e:
                    logger.error(f"Tracking error: {e}")

                last_trade_time = pd.Timestamp.now(tz="Asia/Kolkata")
                signal_manager.last_confirmed_trade_time = last_trade_time  # Update global trade lock

            else:
                logger.debug(f"Candle {current_candle_time}: No automated signal found.")

            # 5) Process external signal list with the same df.
            run_external_signal_engine(df, cached_minute_df)

            # 6) Final update of last_signal_time to prevent re-processing same candle
            last_signal_time = current_candle_time
            logger.debug(f"Candle {current_candle_time} processing completed.")
            
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run()
