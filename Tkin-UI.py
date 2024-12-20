import tkinter as tk
from tkinter import ttk
import alpaca_trade_api as tradeapi
import requests
import threading
import time
from scipy.stats import norm
import numpy as np
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Alpaca API Credentials
API_KEY = ""
SECRET_KEY = ""
BASE_URL = "https://paper-api.alpaca.markets"
NEWS_API_KEY = ""
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

# Black-Scholes Formula
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

# Fetch Stock Prices
def get_stock_price(symbol):
    try:
        barset = api.get_latest_trade(symbol)
        return float(barset.price)
    except Exception:
        return None

# Fetch and Clean Historical Data
def get_historical_data(symbol, period="1mo", interval="1d"):
    valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
    if interval not in valid_intervals:
        print(f"Invalid interval: {interval}. Using default '1d'.")
        interval = "1d"

    try:
        stock_data = yf.download(symbol, period=period, interval=interval, progress=False)

        # Check if data is fetched successfully
        if stock_data is None or stock_data.empty:
            print(f"No price data found for {symbol}.")
            return None

        # Ensure all values are numeric
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        stock_data.dropna(inplace=True)  # Drop rows with NaN values
        stock_data[numeric_columns] = stock_data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        stock_data.dropna(inplace=True)  # Drop rows with invalid numeric data

        return stock_data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Fetch News Articles
def get_stock_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return [(article["title"], article["url"]) for article in articles[:5]]
    except Exception:
        return []

# Place Order
def place_order(symbol, qty, side="buy"):
    try:
        api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="gtc")
        return f"Order placed: {side} {qty} shares of {symbol}."
    except Exception as e:
        return f"Error placing order: {e}"

# UI Code
class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MathOS")
        self.running = False
        self.stocks = []

        # UI Components
        self.setup_inputs()
        self.setup_outputs()
        self.setup_buttons()

    def setup_inputs(self):
        ttk.Label(self.root, text="Symbol:").grid(row=0, column=0)
        self.symbol_entry = ttk.Entry(self.root)
        self.symbol_entry.grid(row=0, column=1)

        ttk.Label(self.root, text="Strike Price:").grid(row=1, column=0)
        self.strike_entry = ttk.Entry(self.root)
        self.strike_entry.grid(row=1, column=1)
        self.strike_entry.insert(0, "150")

        ttk.Label(self.root, text="Volatility:").grid(row=2, column=0)
        self.volatility_entry = ttk.Entry(self.root)
        self.volatility_entry.grid(row=2, column=1)
        self.volatility_entry.insert(0, "0.25")

    def setup_outputs(self):
        ttk.Label(self.root, text="News Feed:").grid(row=3, column=0, columnspan=2)
        self.news_text = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.news_text.grid(row=4, column=0, columnspan=2)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=2)

    def setup_buttons(self):
        self.add_stock_button = ttk.Button(self.root, text="Add Stock", command=self.add_stock)
        self.add_stock_button.grid(row=0, column=2)

        self.start_button = ttk.Button(self.root, text="Start", command=self.start)
        self.start_button.grid(row=1, column=2)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop)
        self.stop_button.grid(row=2, column=2)

        self.trade_button = ttk.Button(self.root, text="Place Trade", command=self.place_trade)
        self.trade_button.grid(row=3, column=2)

    def add_stock(self):
        symbol = self.symbol_entry.get()
        if symbol and symbol not in self.stocks:
            self.stocks.append(symbol)

    def start(self):
        self.running = True
        threading.Thread(target=self.run_analysis).start()

    def stop(self):
        self.running = False

    def place_trade(self):
        symbol = self.symbol_entry.get()
        message = place_order(symbol, qty=1, side="buy")
        self.news_text.insert(tk.END, f"{message}\n")

    def run_analysis(self):
        while self.running:
            for symbol in self.stocks:
                stock_price = get_stock_price(symbol)
                if stock_price:
                    call_price = black_scholes(stock_price, 150, 30/365, 0.05, 0.25)
                    self.news_text.insert(tk.END, f"{symbol} Price: ${stock_price:.2f}, Call: ${call_price:.2f}\n")
                else:
                    self.news_text.insert(tk.END, f"Error fetching price for {symbol}\n")

            self.update_news_feed()
            self.update_chart()
            time.sleep(5)

    def update_news_feed(self):
        query = " OR ".join(self.stocks)
        articles = get_stock_news(query)
        self.news_text.delete(1.0, tk.END)
        for title, url in articles:
            self.news_text.insert(tk.END, f"{title}\n{url}\n\n")

    def plot_chart(self):
        symbol = self.symbol_entry.get()
        interval = self.interval_entry.get()
        data = get_historical_data(symbol, interval=interval)

        if data is not None and not data.empty:
            try:
                mpf.plot(data, type="candle", volume=True, title=f"{symbol} - {interval} Interval", style="yahoo")
            except Exception as e:
                self.news_text.insert(tk.END, f"Error plotting chart: {e}\n")
        else:
            self.news_text.insert(tk.END, f"No valid data for {symbol}.\n")

    def update_chart(self):
        self.ax.clear()
        for symbol in self.stocks:
            prices = [get_stock_price(symbol) for _ in range(5)]
            if all(prices):
                self.ax.plot(prices, label=symbol)
        self.ax.legend()
        self.canvas.draw()

# Run App
root = tk.Tk()
app = TradingApp(root)
root.mainloop()
