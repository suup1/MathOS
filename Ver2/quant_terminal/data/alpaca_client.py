"""
data/alpaca_client.py
Alpaca Market Data Client — Quant Research Terminal
Handles historical and live data fetching from Alpaca API.
"""

import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# ─────────────────────────────────────────────
#  CONFIGURE YOUR ALPACA API KEYS HERE
# ─────────────────────────────────────────────
API_KEY    = "PK5XZX6ZGOXIKPY7MRYIQNYSUC"
SECRET_KEY = "46C7wtA3zqDbmAdrC8St5pTxCRe53QaEcuA61GEVV78L"
# ─────────────────────────────────────────────


class AlpacaClient:
    """
    Wrapper around Alpaca's historical data SDK.
    Provides clean DataFrames for quant model consumption.
    """

    def __init__(self, api_key: str = API_KEY, secret_key: str = SECRET_KEY):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    # ──────────────────────────────────────────
    #  HISTORICAL BARS
    # ──────────────────────────────────────────
    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Day,
        start: str = None,
        end: str = None,
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.

        Parameters
        ----------
        symbol        : Ticker symbol e.g. "AAPL"
        timeframe     : TimeFrame.Day / TimeFrame.Hour / TimeFrame.Minute
        start         : ISO date string "YYYY-MM-DD" (optional)
        end           : ISO date string "YYYY-MM-DD" (optional)
        lookback_days : Days back from today if start not provided

        Returns
        -------
        pd.DataFrame with columns: open, high, low, close, volume, vwap
        """
        if start is None:
            start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed="iex"      # IEX feed — free on all Alpaca accounts
        )

        bars = self.client.get_stock_bars(request)
        df   = bars.df

        # Flatten multi-index if multiple symbols were requested
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        return df[["open", "high", "low", "close", "volume", "vwap"]]

    # ──────────────────────────────────────────
    #  RETURNS SERIES
    # ──────────────────────────────────────────
    def get_returns(self, symbol: str, **kwargs) -> pd.Series:
        """
        Returns daily log-returns for a symbol.
        """
        import numpy as np
        df = self.get_bars(symbol, **kwargs)
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        returns.name = f"{symbol}_log_return"
        return returns

    # ──────────────────────────────────────────
    #  MULTIPLE SYMBOLS
    # ──────────────────────────────────────────
    def get_multi_bars(
        self,
        symbols: list,
        timeframe: TimeFrame = TimeFrame.Day,
        lookback_days: int = 365
    ) -> dict:
        """
        Fetch bars for multiple symbols.
        Returns dict of {symbol: DataFrame}.
        """
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.get_bars(sym, timeframe=timeframe, lookback_days=lookback_days)
            except Exception as e:
                print(f"[AlpacaClient] Warning: Could not fetch {sym} — {e}")
        return result

    # ──────────────────────────────────────────
    #  LATEST QUOTE (real-time snapshot)
    # ──────────────────────────────────────────
    def get_latest_quote(self, symbol: str) -> dict:
        """
        Returns latest bid/ask quote for a symbol.
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote   = self.client.get_stock_latest_quote(request)
        q       = quote[symbol]
        return {
            "symbol"    : symbol,
            "bid_price" : q.bid_price,
            "ask_price" : q.ask_price,
            "bid_size"  : q.bid_size,
            "ask_size"  : q.ask_size,
            "timestamp" : q.timestamp
        }


# ──────────────────────────────────────────────
#  QUICK TEST (run this file directly to verify)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    client = AlpacaClient()
    print("Fetching AAPL daily bars (last 30 days)...")
    df = client.get_bars("AAPL", lookback_days=30)
    print(df.tail(5))

    print("\nFetching log returns...")
    ret = client.get_returns("AAPL", lookback_days=30)
    print(ret.tail(5))