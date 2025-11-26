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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# Telegram
from telegram import Bot
from telegram.error import TelegramError


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


class TechnicalIndicators:
    """Comprehensive technical indicators for 1-minute scalping"""

    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            if len(df) < 50:
                return {}

            self.indicators = {}

            # 1. TREND INDICATORS
            self._calculate_trend_indicators(df)

            # 2. MOMENTUM INDICATORS
            self._calculate_momentum_indicators(df['close'], df['high'], df['low'])

            # 3. VOLATILITY INDICATORS
            self._calculate_volatility_indicators(df['close'], df['high'], df['low'])

            # 4. VOLUME INDICATORS
            self._calculate_volume_indicators(df['close'], df['volume'], df['high'], df['low'])

            return self.indicators

        except Exception as e:
            logging.error(f"Indicator calculation error: {e}")
            return {}

    def _calculate_trend_indicators(self, df):
        """Calculate trend-following indicators"""
        # SMAs (Multiple timeframes)
        self.indicators['sma_5'] = ta.trend.sma_indicator(df['close'], window=5).iloc[-1]
        #print(f"SMA_5: {self.indicators['sma_5']}")
        self.indicators['sma_10'] = ta.trend.sma_indicator(df['close'], window=10).iloc[-1]
        #print(f"SMA_10: {self.indicators['sma_10']}")
        self.indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
        #print(f"SMA_20: {self.indicators['sma_20']}")
        self.indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
        #print(f"SMA_50: {self.indicators['sma_50']}")

        # EMAs
        self.indicators['ema_8'] = ta.trend.ema_indicator(df['close'], window=8).iloc[-1]
        self.indicators['ema_21'] = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]

        # MACD
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        self.indicators['macd'] = macd.macd().iloc[-1]
        self.indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        self.indicators['macd_histogram'] = macd.macd_diff().iloc[-1]

        # VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close'] / 3)
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        self.indicators['vwap'] = vwap.iloc[-1]
        self.indicators['vwap_distance'] = ((df.close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]) * 100

    def _calculate_momentum_indicators(self, close, high, low):
        """Calculate momentum oscillators"""
        # RSI (Multiple timeframes)
        self.indicators['rsi_6'] = ta.momentum.rsi(close, window=6).iloc[-1]  # Fast RSI
        self.indicators['rsi_14'] = ta.momentum.rsi(close, window=14).iloc[-1]
        self.indicators['rsi_21'] = ta.momentum.rsi(close, window=21).iloc[-1]

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
        self.indicators['stoch_k'] = stoch.stoch().iloc[-1]
        self.indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]

        # Williams %R
        self.indicators['williams_r'] = ta.momentum.williams_r(high=high, low=low, close=close).iloc[-1]

        # CCI
        self.indicators['cci'] = ta.trend.CCIIndicator(high=high, low=low, close=close).cci().iloc[-1]

    def _calculate_volatility_indicators(self, close, high, low):
        """Calculate volatility indicators"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        self.indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
        self.indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
        self.indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        self.indicators['bb_position'] = (close.iloc[-1] - self.indicators['bb_lower']) / (
                    self.indicators['bb_upper'] - self.indicators['bb_lower'])
        self.indicators['bb_width'] = self.indicators['bb_upper'] - self.indicators['bb_lower']

        # ATR (Average True Range)
        self.indicators['atr'] = \
        ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().iloc[-1]

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high=high, low=low, close=close)
        self.indicators['kc_upper'] = kc.keltner_channel_hband().iloc[-1]
        self.indicators['kc_lower'] = kc.keltner_channel_lband().iloc[-1]

    def _calculate_volume_indicators(self, close, volume, high, low):
        """Calculate volume-based indicators"""
        # Volume SMA
        self.indicators['volume_sma_20'] = volume.rolling(20).mean().iloc[-1]
        self.indicators['volume_ratio'] = volume.iloc[-1] / self.indicators['volume_sma_20']

        # OBV
        self.indicators['obv'] = \
        ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume().iloc[-1]

        # CMF
        self.indicators['cmf'] = \
        ta.volume.ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume).chaikin_money_flow().iloc[
            -1]

        # MFI
        self.indicators['mfi'] = \
        ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index().iloc[-1]


class TelegramNotifier:
    """Telegram notification system"""

    def __init__(self, bot_token, channel_id):
        self.bot = Bot(token=bot_token)
        self.channel_id = channel_id
        self.signal_count = 0

    def send_scalping_signal(self, signal_data, indicators, ai_prediction):
        """Send comprehensive scalping signal to Telegram"""
        try:
            self.signal_count += 1

            emoji = "🟢" if signal_data['signal'] == 'BUY' else "🔴" if signal_data['signal'] == 'SELL' else "⚪"

            message = f"""
{emoji} **ULTIMATE SCALPING SIGNAL #{self.signal_count}** {emoji}

**🎯 ACTION:** {signal_data['signal']} | **💪 STRENGTH:** {signal_data['strength']}
**💰 PRICE:** ${signal_data['price']:.2f}
**🤖 AI CONFIDENCE:** {ai_prediction['confidence']:.1%}

**📊 TECHNICALS:**
• RSI 14: {indicators.get('rsi_14', 0):.1f}
• MACD: {indicators.get('macd', 0):.3f}
• BB Position: {indicators.get('bb_position', 0):.1%}
• VWAP Dist: {indicators.get('vwap_distance', 0):.2f}%
• Volume: {indicators.get('volume_ratio', 0):.1f}x

**🎯 TARGETS:**
• Take Profit: ${signal_data.get('take_profit', 0):.2f}
• Stop Loss: ${signal_data.get('stop_loss', 0):.2f}

**🤖 AI MODELS:**
• XGBoost: {ai_prediction.get('xgboost', 0):.1%}
• Random Forest: {ai_prediction.get('random_forest', 0):.1%}
• TSLM: {ai_prediction.get('tslm', 0):.1%}

⏰ *Time: {datetime.now().strftime('%H:%M:%S')}*
            """.strip()

            self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode='Markdown'
            )

            logging.info(f"✅ Telegram signal #{self.signal_count} sent")

        except TelegramError as e:
            logging.error(f"❌ Telegram error: {e}")
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
            print(f"df_exisiting data count:{len(df_existing)}")

                # Combine the old and new DataFrames
            df_combined = pd.concat([df_existing, df_new_data])
            print(f"df_combined data count:{len(df_combined)}")

                # Remove duplicates based on the index (keeping the first occurrence)
                # This ensures we only keep unique timestamps
            df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]
            print(f"appended data count:{len(df_combined)}")
                # Write the cleaned, combined DataFrame (overwriting the old file)
            df_combined.to_csv(file_path, index=True, index_label=index_label)
            print(f"Appended new data to {file_path} and removed duplicates and count of combined file :{len(df_combined)}")

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
        print(f"Successfully loaded {len(df_EmptyDataFrame)} records from {file_path}.")
        return df_EmptyDataFrame
    else:
        print(f"Error: The file {file_path} does not exist.")
        return df_EmptyDataFrame # Return an empty DataFrame
def main():
    import os
    import pandas as pd
    load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")
    config=SimpleRSIConfig()
    logger=AdvancedLogger()
    TI=TechnicalIndicators()

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
        try:
            system = SchwabDataFetcher(c, logger, config)
            logger.logger.info("data fetch completed successfully")
        except Exception as e:
            logger.logger.info("Issuing while fetching the data")

        is_running = True
        cycle_count = 0
        try:
            df_EmptyDataFrame = pd.DataFrame()
            logger.logger.info("empty dataframe created successfully")
        except Exception as e:
            logger.logger.info("empty dataframe not created successfully")
        """Start the ultimate scalping system"""
        logger.logger.info("🚀 ULTIMATE SCALPING SYSTEM STARTED")
        logger.logger.info("💻 Optimized for Intel Core Ultra")
        logger.logger.info("🧠 Hybrid AI: TSLM + XGBoost + Random Forest")
        logger.logger.info("📊 Indicators: SMA, VWAP, MACD, RSI, Bollinger Bands")
        logger.logger.info("⏰ 1-minute intervals with Telegram notifications")
        try:
            while is_running:
                if cycle_count == 0:
                    cycle_count = 1
                    try:
                        df_final = load_historical_data_from_csv(df_EmptyDataFrame, backup_file_path,
                                                                        index_label='datetime')
                        logger.logger.info(
                            f"first data set which is going for predictive analysis:{len(df_final)} and {cycle_count}")
                    except Exception as e:
                            logger.logger.error(f"FATAL ERROR during historical backup of data: {e}")
                else:
                    cycle_count += 1
                    cycle_start = time.time()
                    data_list = system.fetch_data()
                    append_new_data_only(
                            data_list,  # Assign by keyword for clarity
                            backup_file_path
                            )

                    df_final = load_historical_data_from_csv(df_EmptyDataFrame, backup_file_path, index_label='datetime')
                    logger.logger.info(f"current data set which is going for predictive analysis:{len(df_final)} and {cycle_count}")
                    logger.logger.info("final dataframe loaded from disk successfully")
                    TechnicalIndicatorsValue = TI.calculate_all_indicators(df_final)
                    for key, value in TechnicalIndicatorsValue.items():
                        logger.logger.info(f"{key}: {value}")
                    cycle_duration = time.time() - cycle_start
                    sleep_time = max(1, 60 - cycle_duration)
                    logger.logger.info(f"⏰ Cycle completed in {cycle_duration:.2f}s")
                    time.sleep(sleep_time)
                    if cycle_count == 10:
                        sys.exit (0)
        except Exception as e:
            logger.logger.error(f"FATAL ERROR during backup of data: {e}")

        '''data_list = system.fetch_data()
        #backup_file_path = "C:\\Users\\Administrator\\PycharmProjects\\PythonProject\\backup\\schwab_candles_backup.csv"
        try:
            append_new_data_only(
                data_list,  # Assign by keyword for clarity
                backup_file_path
            )
            logger.logger.info("backup of data is completed")
        except Exception as e:
            logger.logger.error(f"FATAL ERROR during backup of data: {e}")
        #print (data_list)'''
    except Exception as e:
        logger.logger.error(f"Authentication error: {e}")
        import traceback
if __name__ == "__main__":
    print("Yahoo Finance RSI System")
    main()
