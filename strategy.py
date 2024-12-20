class MovingAverageStrategy:
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        # Calculate moving averages
        data['SMA50'] = data['close'].rolling(window=self.short_window).mean()
        data['SMA200'] = data['close'].rolling(window=self.long_window).mean()

        # Trading signal based on the moving averages
        if data['SMA50'].iloc[-1] > data['SMA200'].iloc[-1]:
            return 'buy'
        elif data['SMA50'].iloc[-1] < data['SMA200'].iloc[-1]:
            return 'sell'
        else:
            return 'hold'
