import pandas as pd

_IST = "Asia/Kolkata"


def get_fixed_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    ema50 = last["EMA50"]
    ema200 = last["EMA200"]
    atr = last["ATR"]
    atr_mean = df["ATR"].mean()
    candle_size = abs(last["Close"] - last["Open"])
    avg_candle = (df["Close"] - df["Open"]).abs().tail(10).mean()
    distance = abs(last["Close"] - ema50)
    trend_strength = last["TrendStrength"]

    required_values = [
        ema50,
        ema200,
        last["RSI"],
        prev["RSI"],
        atr,
        atr_mean,
        avg_candle,
        trend_strength,
    ]

    if any(pd.isna(value) for value in required_values):
        return None

    trend_threshold = max(0.0002, atr * 0.15)
    ema_distance_threshold = max(0.0003, atr * 0.50)

    if atr <= atr_mean:
        return None

    if candle_size <= avg_candle:
        return None

    if distance > ema_distance_threshold:
        return None

    if trend_strength <= trend_threshold:
        return None

    if (
        ema50 > ema200
        and last["RSI"] > 55
        and last["RSI"] > prev["RSI"]
    ):
        signal = "CALL"
    elif (
        ema50 < ema200
        and last["RSI"] < 45
        and last["RSI"] < prev["RSI"]
    ):
        signal = "PUT"
    else:
        return None

    now = pd.Timestamp.now(tz=_IST)

    # Round to next minute.
    next_minute = now.ceil("min")

    # Entry = 2 minutes ahead.
    entry = next_minute + pd.Timedelta(minutes=2)

    # Expiry = 5 minutes.
    expiry = entry + pd.Timedelta(minutes=5)

    # Prevent late signals.
    if (entry - now).total_seconds() < 90:
        return None

    return {
        "signal": signal,
        "entry": entry.strftime("%H:%M"),
        "expiry": expiry.strftime("%H:%M"),
        "seconds_left": int((entry - now).total_seconds())
    }
