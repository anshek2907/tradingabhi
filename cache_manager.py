from logger import logger
import pandas as pd

class CacheManager:
    def __init__(self):
        self.raw_cache = {}
        self.raw_cache_time = {}
        self.processed_cache = {}
        
    def cleanup_stale_cache(self):
        """Clear cache if older than 15 minutes to avoid indefinite RAM usage."""
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        
        stale_keys = []
        for interval, cached_time in self.raw_cache_time.items():
            if (now - cached_time).total_seconds() > 900:
                logger.debug(f"Clearing stale {interval} cache (>15m)")
                stale_keys.append(interval)
                
        for interval in stale_keys:
            self.raw_cache.pop(interval, None)
            self.raw_cache_time.pop(interval, None)
            self.processed_cache.pop(interval, None)

    def get_candle_key(self, interval):
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if interval == "1min": return now.floor("min")
        if interval == "5min": return now.floor("5min")
        if interval == "15min": return now.floor("15min")
        if interval == "30min": return now.floor("30min")
        if interval == "1h": return now.floor("1h")
        return now.floor("min")

    def get_dataframe(self, interval, fetch_func):
        self.cleanup_stale_cache()
        current_candle_key = self.get_candle_key(interval)
        
        cached_df = self.raw_cache.get(interval)
        cached_time = self.raw_cache_time.get(interval)
        
        if cached_df is not None and cached_time == current_candle_key:
            logger.debug(f"Cache HIT for {interval} | Key: {current_candle_key}")
            return cached_df.copy(deep=True)
        else:
            logger.debug(f"Cache MISS for {interval} | Key: {current_candle_key} | Refreshing from API")
            df = fetch_func(interval)
            if df is not None:
                self.raw_cache[interval] = df.copy(deep=True)
                self.raw_cache_time[interval] = current_candle_key
                self.processed_cache[interval] = None  # Reset processed
            return df.copy(deep=True) if df is not None else None

    def get_processed_dataframe(self, interval, fetch_func, process_func):
        df = self.get_dataframe(interval, fetch_func)
        if df is None:
            return None
            
        processed_df = self.processed_cache.get(interval)
        if processed_df is not None:
            logger.debug(f"Processed cache HIT for {interval}")
            return processed_df.copy(deep=True)
            
        self.processed_cache[interval] = process_func(df.copy(deep=True))
        return self.processed_cache[interval].copy(deep=True)

cache = CacheManager()

