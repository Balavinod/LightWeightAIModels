# simple_yahoo_rsi.py
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta
import logging
import os
import json
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import sys
import asyncio
import aiohttp
import concurrent.futures
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import psutil
import gc
from dotenv import load_dotenv
import json
from schwab import auth, client
from dotenv import load_dotenv
import pprint


# Advanced Logging Configuration
class AdvancedLogger:
    def __init__(self, name='YahooRSI'):
        self.logger = logging.getLogger(name)
        self.setup_logging()

    def setup_logging(self):
        """Setup advanced logging with multiple handlers"""
        # Clear any existing handlers
        self.logger.handlers.clear()

        # Set level
        self.logger.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler (Colored)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # 2. File Handler (Rotates daily)
        file_handler = TimedRotatingFileHandler(
            'yahoo_rsi.log',
            when='D',  # Daily rotation
            interval=1,
            backupCount=7,# Keep 7 days of logs
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # 3. Error Handler (Separate error file)
        error_handler = RotatingFileHandler(
            'yahoo_rsi_errors.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        # 4. JSON Handler for structured data
        json_handler = RotatingFileHandler(
            'yahoo_rsi_data.json',
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8'
        )
        json_handler.setLevel(logging.INFO)
        json_formatter = logging.Formatter('%(message)s')
        json_handler.setFormatter(json_formatter)

        # Add all handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(json_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def log_data_dump(self, symbol, data):
        """Log structured data dump in JSON format"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'DATA_DUMP',
            'symbol': symbol,
            'data': data
        }
        self.logger.info(json.dumps(log_entry, default=str))

    def log_technical_signal(self, symbol, indicator, value, signal):
        """Log technical indicator signals"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'TECHNICAL_SIGNAL',
            'symbol': symbol,
            'indicator': indicator,
            'value': value,
            'signal': signal,
            'level': 'BUY' if signal == 'OVERSOLD' else 'SELL' if signal == 'OVERBOUGHT' else 'NEUTRAL'
        }
        self.logger.info(json.dumps(log_entry))

    def log_performance(self, operation, duration, success=True):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'PERFORMANCE',
            'operation': operation,
            'duration_seconds': duration,
            'success': success
        }
        self.logger.info(json.dumps(log_entry))

    def log_error(self, error_type, message, context=None):
        """Log errors with context"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'ERROR',
            'error_type': error_type,
            'message': message,
            'context': context
        }
        self.logger.error(json.dumps(log_entry))

    def append_new_data_only(df_new_data, file_path, index_label='datetime'):

        # 1. Define the columns to check for duplicates (your unique index/timestamp)
        unique_columns = [index_label]

        # 2. Check if the file already exists
        if os.path.exists(file_path):
            # Read the existing data into a DataFrame
            try:
                df_existing = pd.read_csv(file_path, parse_dates=[index_label], index_col=index_label)

                # Combine the old and new DataFrames
                df_combined = pd.concat([df_existing, df_new_data])

                # Remove duplicates based on the index (keeping the first occurrence)
                # This ensures we only keep unique timestamps
                df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]

                # Write the cleaned, combined DataFrame (overwriting the old file)
                df_combined.to_csv(file_path, index=True, index_label=index_label)
                print(f"Appended new data to {file_path} and removed duplicates.")

            except pd.errors.EmptyDataError:
                # Handle case where file is empty but exists
                df_new_data.to_csv(file_path, index=True, index_label=index_label)
                print(f"File was empty, wrote new data to {file_path}.")

        else:
            # 3. If the file does not exist, simply write the new data
            df_new_data.to_csv(file_path, index=True, index_label=index_label)
            print(f"Created new file: {file_path} with new data.")

class SimpleRSIConfig:
    def __init__(self):
        self.SYMBOL ='/NQ'
        self.LOOKBACK_PERIOD = 40
        self.UPDATE_INTERVAL = 60

        self.STOP_LOSS_POINTS = 20
        self.TAKE_PROFIT_POINTS = 35
        self.MIN_CONFIDENCE = 0.65
        self.INTERVAL = '1m'
        self.PERIOD = 0
        self.file_path = 'C:\\Users\\Administrator\\PycharmProjects\\PythonProject\\backup'

        self.MODEL_WEIGHTS = {
            'xgboost': 0.6,
            'random_forest': 0.4
        }


class SchwabDataFetcher:
    def __init__(self, easy_client, logger, Config):
        self.config = Config
        self.logger = logger
        self.easy_client = easy_client
        self.data_history = []
    def fetch_data(self):
        try:

            start_time = time.time()
            # Create ticker object
            response_object = self.easy_client.get_price_history_every_minute('/NQ',
                                                                        # max allowed for minute data
                                                                        need_extended_hours_data=True
                                                                        # futures trade almost 24h
                                                                       )
            fetch_duration = time.time() - start_time
            hist_data = response_object.json()
            if hist_data.get('empty') != False:
                self.logger.log_error(
                    "DATA_FETCH_ERROR",
                    f"No data returned for {self.config.SYMBOL}",
                    {'symbol': self.config.SYMBOL, 'period': self.config.PERIOD, 'interval': self.config.INTERVAL}
                )
                return None

            data_points = []
            #print(hist_data)
            if 'candles' in hist_data:
                for candle in hist_data['candles']:
                    data_points.append({
                        'datetime': candle['datetime'],
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': candle['volume']
                    })
                    #print(data_points)
                df_hist_data = pd.DataFrame(data_points)
                df_hist_data['datetime'] = pd.to_datetime(df_hist_data['datetime'], unit='ms')
                df_hist_data.set_index('datetime', inplace=True)
                first_ms = df_hist_data.index[0]
                last_ms = df_hist_data.index[-1]
                duration = last_ms - first_ms
                duration_minutes = duration.total_seconds() / 60
                try:
                    self.logger.log_data_dump(self.config.SYMBOL, {
                        'period': duration_minutes,
                        'interval': self.config.INTERVAL,
                        'data_points_count': len(data_points),
                        'fetch_duration': fetch_duration,
                        'latest_price': df_hist_data['close'].iloc[-1] if len(df_hist_data) > 0 else None
                    })
                    print ("DEBUG: log_data_dump executed successfully.")
                except Exception as e:
                    self.logger.logger.error(f"FATAL ERROR during log_data_dump preparation: {e}")
                    print(f"DEBUG: log_data_dump FAILED: {e}")  # Confirmation print
                self.logger.log_performance("DATA_FETCH", fetch_duration, True)
                self.logger.logger.info(f"✅ Fetched {len(df_hist_data)} data points for {self.config.SYMBOL} in {fetch_duration:.2f}s")
            # Log data dump
            #print( data_points )

            return df_hist_data
        except Exception as e:
            fetch_duration = time.time() - start_time
            self.logger.log_error(
                "DATA_FETCH_EXCEPTION",
                str(e),
                {'symbol': self.config.SYMBOL, 'period': self.config.PERIOD, 'interval': self.config.INTERVAL, 'duration': fetch_duration}
            )
            self.logger.log_performance("DATA_FETCH", fetch_duration, False)
            return None
def append_new_data_only(df_new_data, file_path, index_label='datetime'):

        # 1. Define the columns to check for duplicates (your unique index/timestamp)
    unique_columns = [index_label]

        # 2. Check if the file already exists
    if os.path.exists(file_path):
            # Read the existing data into a DataFrame
        try:
            df_existing = pd.read_csv(file_path, parse_dates=[index_label], index_col=index_label)

                # Combine the old and new DataFrames
            df_combined = pd.concat([df_existing, df_new_data])

                # Remove duplicates based on the index (keeping the first occurrence)
                # This ensures we only keep unique timestamps
            df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]

                # Write the cleaned, combined DataFrame (overwriting the old file)
            df_combined.to_csv(file_path, index=True, index_label=index_label)
            print(f"Appended new data to {file_path} and removed duplicates.")

        except pd.errors.EmptyDataError:
                # Handle case where file is empty but exists
            df_new_data.to_csv(file_path, index=True, index_label=index_label)
            print(f"File was empty, wrote new data to {file_path}.")

        else:
            # 3. If the file does not exist, simply write the new data
            df_new_data.to_csv(file_path, index=True, index_label=index_label)
            print(f"Created new file: {file_path} with new data.")
def load_historical_data_from_csv(df_EmptyDataFrame,file_path, index_label='datetime'):
    """
    Loads historical OHLCV data from a CSV file into a pandas DataFrame,
    ensuring the timestamp is parsed correctly as the index.
    """
    if os.path.exists(file_path):
        # Read the CSV file
        df_EmptyDataFrame = pd.read_csv(
            file_path,
            index_col=index_label,  # Use the 'datetime' column as the index
            parse_dates=[index_label] # Crucial: tells pandas to convert the strings to datetime objects
        )
        print(f"Successfully loaded {len(df)} records from {file_path}.")
        return df_EmptyDataFrame
    else:
        print(f"Error: The file {file_path} does not exist.")
        return df_EmptyDataFrame # Return an empty DataFrame
def main():
    import os
    load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")
    config=SimpleRSIConfig()
    logger=AdvancedLogger()

    # Verify environment variables
    api_key = os.getenv('app_key')
    app_secret = os.getenv('app_secret')
    callback_url = os.getenv('callback_url')
    backup_file_path=os.getenv('backup_file_path')

    if not api_key or not app_secret or not callback_url:
        logger.logger.error("Missing required environment variables. Check your .env file.")
        sys.exit(1)

    try:
        logger.logger.info("Attempting authentication with Schwab API...")

        # Create the client
        c = auth.easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path='schwab_token.json'
        )
        logger.logger.info("Client object created successfully")
        system = SchwabDataFetcher(c, logger, config)
        data_list = system.fetch_data()
        #backup_file_path = "C:\\Users\\Administrator\\PycharmProjects\\PythonProject\\backup\\schwab_candles_backup.csv"
        try:
            append_new_data_only(
                data_list,  # Assign by keyword for clarity
                backup_file_path
            )
            logger.logger.info("backup of data is completed")
        except Exception as e:
            logger.logger.error(f"FATAL ERROR during backup of data: {e}")
        #print (data_list)
    except Exception as e:
        logger.logger.error(f"Authentication error: {e}")
        import traceback
if __name__ == "__main__":
    print("Yahoo Finance RSI System")
    main()
