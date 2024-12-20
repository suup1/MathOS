from alpaca_trade_api import REST
import config
from strategy import MovingAverageStrategy
from logger import setup_logger
import pandas as pd
import numpy as np

class TradingBot:
    def __init__(self):
        # Set up API connection and logger
        self.api = REST(config.API_KEY, config.API_SECRET, config.BASE_URL)
        self.logger = setup_logger()
        self.strategy = MovingAverageStrategy(config.SHORT_WINDOW, config.LONG_WINDOW)

    def get_account_info(self):
        try:
            account = self.api.get_account()
            self.logger.info("Account Status: %s", account.status)
        except Exception as e:
            self.logger.error("Error retrieving account info: %s", e)

    def get_historical_data(self, symbol, timeframe='day', limit=200):
            """
            Fetch historical market data for a given symbol and timeframe.

            :param symbol: str - The trading symbol for which to fetch data.
            :param timeframe: str - The timeframe for the data (e.g., "minute", "hour", "day").
            :param limit: int - The maximum number of data points to retrieve.
            :return: pd.DataFrame - The historical market data.
            """
            try:
                data = self.api.get_bars(symbol, timeframe, limit=limit).df
                return data
            except Exception as e:
                self.logger.error("Error fetching historical data: %s", e)
                return pd.DataFrame()

    def get_highest_price(self, symbol):
        """
        Fetch the highest historical price for a given symbol.

        :param symbol: str - The trading symbol for which to fetch the highest price.
        :return: float - The highest historical price.
        """
        try:
            data = self.get_historical_data(symbol)
            highest_price = np.nanmax(data['high'])
            return highest_price
        except Exception as e:
            self.logger.error("Error fetching highest price: %s", e)
            return None

    def execute_trade(self, symbol):
        moon_price = self.get_highest_price(symbol)
        if moon_price:
            data = self.api.get_bars(symbol, 'minute', limit=1).df
            current_price = data['close'].iloc[0]
            if current_price >= 0.97 * moon_price:
                self.logger.info(f"Current price is within 3% of the highest price for {symbol}. Buying signal triggered.")
                signal = 'buy'
            else:
                signal = self.strategy.generate_signals(data)
        if signal == 'buy':
            self.logger.info(f"Signal: {signal} - Buying {config.ORDER_QUANTITY} shares of {symbol}")
            # API call for a buy order
            self.api.submit_order(
                symbol=symbol,
                qty=config.ORDER_QUANTITY,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif signal == 'sell':
            self.logger.info(f"Signal: {signal} - Selling {config.ORDER_QUANTITY} shares of {symbol}")
            # API call for a sell order
            self.api.submit_order(
                symbol=symbol,
                qty=config.ORDER_QUANTITY,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        else:
            self.logger.info("Signal: hold - No trade executed")

        # Fetch data and generate signals
        data = self.get_historical_data(symbol)
        signal = self.strategy.generate_signals(data)

        self.logger.info("Starting bot run")
        self.get_account_info()
        self.execute_trade(config.SYMBOL)

    def run(self):
        self.logger.info("Starting bot run")
        self.get_account_info()
        self.execute_trade(config.SYMBOL)