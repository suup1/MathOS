import schedule
import time
from bot import TradingBot

# Initialize the bot
bot = TradingBot()

# Function to run the bot
def run_bot():
    bot.run()

# Schedule to run the bot every minute
schedule.every(1).minutes.do(run_bot)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
