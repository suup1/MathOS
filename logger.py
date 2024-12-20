# logger.py
import logging

def setup_logger():
    logger = logging.getLogger("TradingBotLogger")
    logger.setLevel(logging.INFO)

    # File handler to log messages to a file
    file_handler = logging.FileHandler("trading_bot.log")
    file_handler.setLevel(logging.INFO)

    # Logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Adding file handler to the logger
    logger.addHandler(file_handler)

    return logger
