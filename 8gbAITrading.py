import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import TelegramError
import warnings
import gc
import psutil
from schwab import auth, client
import json

warnings.filterwarnings('ignore')

# ============================================
# TELEGRAM BOT CONFIGURATION
# ============================================

# Your Telegram Bot Token from @BotFather
TELEGRAM_BOT_TOKEN="8431923854:AAEb_apNWFMQMJiJA7_sH9SGKAOPLpwYfBU"

# Your Telegram Channel ID (include the @ symbol)
TELEGRAM_CHANNEL_ID="-1003128773566"


# Schwab API Configuration
class Config:
    TELEGRAM_BOT_TOKEN = "*********"
    TELEGRAM_CHANNEL_ID = "**********"

    # Schwab API credentials
    SCHWAB_APP_KEY = "*************"
    SCHWAB_APP_SECRET = "**************"
    SCHWAB_CALLBACK_URL = "********************"

    SYMBOL = "/NQ"  # NASDAQ Futures symbol for Schwab
    TIMEFRAME = "1min"  # Schwab timeframe
    LOOKBACK_PERIOD = 60
    UPDATE_INTERVAL = 60

    # Memory optimization
    MAX_DATA_POINTS = 1000
    ENABLE_LSTM = False
    FEATURE_REDUCTION = True


class SchwabDataStream:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.data_buffer = []
        self.current_data = None
        self.is_authenticated = False
        self.memory_optimizer = MemoryOptimizer()

    async def initialize_client(self):
        """Initialize and authenticate Schwab client"""
        try:
            self.client = auth.easy_client(
                app_key=self.config.SCHWAB_APP_KEY,
                app_secret=self.config.SCHWAB_APP_SECRET,
                callback_url=self.config.SCHWAB_CALLBACK_URL
            )

            # For first-time use, you'll need to complete OAuth flow
            # This is a simplified version - you may need to handle OAuth separately
            await self.client.login()
            self.is_authenticated = True
            print("Schwab client initialized successfully")
            return True

        except Exception as e:
            print(f"Schwab initialization error: {e}")
            return False

    async def get_real_time_data(self):
        """Get real-time data from Schwab API"""
        try:
            if not self.is_authenticated:
                success = await self.initialize_client()
                if not success:
                    return False

            # Get historical data for the symbol
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=4)

            # Convert to milliseconds for Schwab API
            start_epoch = int(start_time.timestamp() * 1000)
            end_epoch = int(end_time.timestamp() * 1000)

            # Get price history from Schwab
            price_history = await self.client.get_price_history(
                symbol=self.config.SYMBOL,
                period_type='day',
                period=1,
                frequency_type='minute',
                frequency=1,
                start_date=start_epoch,
                end_date=end_epoch,
                need_extended_hours_data=True
            )

            if price_history and 'candles' in price_history:
                processed_data = []
                for candle in price_history['candles']:
                    tick_data = {
                        'timestamp': datetime.fromtimestamp(candle['datetime'] / 1000),
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': int(candle['volume'])
                    }
                    processed_data.append(tick_data)

                # Limit buffer size
                self.data_buffer = processed_data[-self.config.MAX_DATA_POINTS:]
                self.current_data = processed_data[-1] if processed_data else None

                return True
            else:
                print("No data received from Schwab API")
                return False

        except Exception as e:
            print(f"Error fetching Schwab data: {e}")
            return False

    async def get_quote(self):
        """Get real-time quote"""
        try:
            if not self.is_authenticated:
                return None

            quote = await self.client.get_quote(self.config.SYMBOL)
            return quote
        except Exception as e:
            print(f"Error getting quote: {e}")
            return None

    async def start_stream(self):
        """Start Schwab data streaming"""
        print("Starting Schwab data stream...")
        success = await self.initialize_client()
        if success:
            await self.get_real_time_data()


class AsyncTelegramBot:
    def __init__(self, bot_token, channel_id):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.application = None
        self.bot = None

    async def initialize(self):
        """Initialize async Telegram bot"""
        try:
            self.bot = Bot(token=self.bot_token)
            self.application = Application.builder().token(self.bot_token).build()

            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("stop", self.stop_command))

            print("Telegram bot initialized successfully")
            return True
        except Exception as e:
            print(f"Telegram bot initialization error: {e}")
            return False

    async def start_command(self, update, context):
        """Handle /start command"""
        await update.message.reply_text(
            "🤖 NASDAQ Futures AI Trading Bot Started\n"
            "Monitoring /NQ with real-time Schwab data"
        )

    async def status_command(self, update, context):
        """Handle /status command"""
        await update.message.reply_text(
            "✅ Bot is running\n"
            "Data Source: Schwab API\n"
            "Symbol: /NQ\n"
            "Status: Active"
        )

    async def stop_command(self, update, context):
        """Handle /stop command"""
        await update.message.reply_text("🛑 Bot stopping...")
        # You can implement proper shutdown logic here

    async def send_signal_async(self, signal):
        """Send signal asynchronously"""
        try:
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}

            message = f"""
{emoji[signal['signal']]} NAS100 Signal (Schwab)

Action: {signal['signal']} | Strength: {signal['strength']}
Confidence: {signal['confidence']:.2%}
Price: ${signal['current_price']:.2f}

TP: ${signal.get('take_profit', 0):.2f}
SL: ${signal.get('stop_loss', 0):.2f}

Time: {signal['timestamp'].strftime('%H:%M:%S')}
Source: Schwab API
            """

            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message
            )
            return True

        except Exception as e:
            print(f"Async Telegram error: {e}")
            return False

    async def send_alert_async(self, alert_type, message):
        """Send alert asynchronously"""
        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message
            )
            return True
        except Exception as e:
            print(f"Async Telegram alert error: {e}")
            return False

    async def start_polling(self):
        """Start async polling for commands"""
        if self.application:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()


class MemoryOptimizer:
    def __init__(self):
        self.memory_threshold = 0.8

    def check_memory(self):
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return memory.percent

    async def cleanup_memory_async(self):
        """Force garbage collection asynchronously"""
        gc.collect()
        await asyncio.sleep(0.1)  # Small delay to allow GC to work

    def should_reduce_features(self):
        """Check if we need to reduce features due to memory pressure"""
        return self.check_memory() > 60


class LightweightFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.essential_indicators = True

    def create_features(self, df):
        """Create only essential technical indicators"""
        if len(df) < 20:
            return df

        # Basic numeric conversion
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        # ESSENTIAL FEATURES ONLY
        df['returns'] = df['close'].pct_change()
        df['price_change'] = df['close'].diff()

        # Moving averages
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume indicator
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # Volatility
        df['volatility'] = df['returns'].rolling(window=10).std()

        self.feature_columns = [col for col in df.columns if col not in
                                ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]

        return df.dropna()

    def create_ultra_light_features(self, df):
        """Ultra-light features for low memory situations"""
        if len(df) < 10:
            return df

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()

        # MINIMAL FEATURES
        df['returns'] = df['close'].pct_change()
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()

        self.feature_columns = [col for col in df.columns if col not in
                                ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask']]

        return df.dropna()


class LightweightAIModels:
    def __init__(self, lookback_period=60, enable_lstm=False):
        self.lookback_period = lookback_period
        self.enable_lstm = enable_lstm
        self.models = {}
        self.ensemble_weights = {
            'xgboost': 0.6,
            'random_forest': 0.4
        }

        if self.enable_lstm:
            self.ensemble_weights = {'xgboost': 0.5, 'random_forest': 0.3, 'lstm': 0.2}
            self._initialize_lstm()

        self._initialize_tree_models()

    def _initialize_lstm(self):
        """Initialize lightweight LSTM"""
        try:
            self.models['lstm'] = tf.keras.Sequential([
                tf.keras.layers.LSTM(32, input_shape=(self.lookback_period, 1)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1, activation='tanh')
            ])
            self.models['lstm'].compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )
        except Exception as e:
            print(f"LSTM initialization failed: {e}")
            self.enable_lstm = False

    def _initialize_tree_models(self):
        """Initialize lightweight tree models"""
        # Lightweight XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=1
        )

        # Lightweight Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )

    async def train_models_async(self, features, targets):
        """Train models asynchronously with memory optimization"""
        try:
            if len(features) < self.lookback_period:
                return

            # Prepare tabular data for tree-based models
            feature_data = features.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask'],
                                         axis=1, errors='ignore')

            if len(feature_data) > len(targets):
                feature_data = feature_data.iloc[:len(targets)]

            if len(feature_data) > 0:
                # Train models (these are CPU-bound, so we run in executor)
                loop = asyncio.get_event_loop()

                # Train XGBoost
                await loop.run_in_executor(None, lambda: self.models['xgboost'].fit(feature_data.values, targets))

                # Train Random Forest
                await loop.run_in_executor(None, lambda: self.models['random_forest'].fit(feature_data.values, targets))

                # Clear memory
                del feature_data
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Model training error: {e}")

    async def predict_async(self, current_data, feature_columns):
        """Generate predictions asynchronously"""
        try:
            predictions = {}
            confidence_scores = {}

            # Tree-based models prediction
            if len(current_data) > 0:
                feature_data = current_data[feature_columns].iloc[-1:].values

                # Run predictions in executor
                loop = asyncio.get_event_loop()

                # XGBoost prediction
                xgb_pred = await loop.run_in_executor(
                    None, lambda: self.models['xgboost'].predict_proba(feature_data)[0][1]
                )
                predictions['xgboost'] = xgb_pred
                confidence_scores['xgboost'] = abs(xgb_pred - 0.5) * 2

                # Random Forest prediction
                rf_pred = await loop.run_in_executor(
                    None, lambda: self.models['random_forest'].predict_proba(feature_data)[0][1]
                )
                predictions['random_forest'] = rf_pred
                confidence_scores['random_forest'] = abs(rf_pred - 0.5) * 2

            # Weighted ensemble prediction
            ensemble_pred = 0
            total_weight = 0

            for model in predictions:
                weight = self.ensemble_weights.get(model, 0)
                ensemble_pred += predictions[model] * weight
                total_weight += weight

            if total_weight > 0:
                ensemble_pred /= total_weight

            # Overall confidence
            avg_confidence = sum(
                confidence_scores.get(model, 0) * self.ensemble_weights.get(model, 0)
                for model in predictions
            )

            return ensemble_pred, avg_confidence, predictions

        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5, 0.0, {}


class AsyncTradingSystem:
    def __init__(self, config):
        self.config = config
        self.data_stream = SchwabDataStream(config)
        self.feature_engineer = LightweightFeatureEngineer()
        self.ai_models = LightweightAIModels(enable_lstm=config.ENABLE_LSTM)
        self.signal_generator = SignalGenerator()
        self.telegram_bot = AsyncTelegramBot(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHANNEL_ID
        )
        self.is_running = False
        self.data_df = pd.DataFrame()
        self.memory_optimizer = MemoryOptimizer()
        self.cycle_count = 0
        self.tasks = set()

    async def start(self):
        """Start the async trading system"""
        self.is_running = True
        print("Starting Async NASDAQ Futures Trading System with Schwab API...")

        # Initialize Telegram bot
        telegram_success = await self.telegram_bot.initialize()
        if telegram_success:
            await self.telegram_bot.send_alert_async('INFO', "🤖 Async NASDAQ Futures AI System Started")

        # Start Schwab data stream
        await self.data_stream.start_stream()

        # Start Telegram polling in background
        telegram_task = asyncio.create_task(self.telegram_bot.start_polling())
        self.tasks.add(telegram_task)
        telegram_task.add_done_callback(self.tasks.discard)

        # Main trading loop
        while self.is_running:
            try:
                self.cycle_count += 1

                # Memory management every 10 cycles
                if self.cycle_count % 10 == 0:
                    await self.memory_optimizer.cleanup_memory_async()

                await self._process_new_data()
                await asyncio.sleep(self.config.UPDATE_INTERVAL)

            except Exception as e:
                print(f"Error in main loop: {e}")
                await asyncio.sleep(10)

    async def _process_new_data(self):
        """Process new data asynchronously"""
        # Check memory before processing
        memory_usage = self.memory_optimizer.check_memory()
        if memory_usage > 85:
            print(f"High memory usage: {memory_usage}% - skipping cycle")
            return

        # Fetch new data from Schwab
        success = await self.data_stream.get_real_time_data()

        if not success or len(self.data_stream.data_buffer) < self.config.LOOKBACK_PERIOD:
            return

        # Convert buffer to DataFrame
        new_data = pd.DataFrame(self.data_stream.data_buffer)
        self.data_df = new_data

        # Choose feature engineering method based on memory
        if memory_usage > 70:
            features_df = self.feature_engineer.create_ultra_light_features(self.data_df.copy())
        else:
            features_df = self.feature_engineer.create_features(self.data_df.copy())

        if len(features_df) < self.config.LOOKBACK_PERIOD:
            return

        # Train models less frequently
        if self.cycle_count % 50 == 0:
            await self._retrain_models(features_df)

        # Generate prediction asynchronously
        prediction, confidence, model_predictions = await self.ai_models.predict_async(
            features_df, self.feature_engineer.feature_columns
        )

        current_price = self.data_df['close'].iloc[-1]

        # Generate signal
        signal = self.signal_generator.generate_signal(
            prediction, confidence, current_price, features_df
        )

        # Send signals asynchronously
        if signal['signal'] != 'HOLD' and confidence > 0.7:
            send_task = asyncio.create_task(
                self.telegram_bot.send_signal_async(signal)
            )
            self.tasks.add(send_task)
            send_task.add_done_callback(self.tasks.discard)

        # Log minimal information
        await self._log_signal_async(signal)

        print(f"Cycle {self.cycle_count} | Mem: {memory_usage}% | "
              f"Price: {current_price:.2f} | Signal: {signal['signal']} | "
              f"Conf: {confidence:.2%}")

        # Clear variables to free memory
        del features_df
        await self.memory_optimizer.cleanup_memory_async()

    async def _retrain_models(self, features_df):
        """Optimized model retraining asynchronously"""
        try:
            print("Retraining models...")

            # Simple target: price increase in next 3 minutes
            lookahead = 3
            targets = (features_df['close'].shift(-lookahead) > features_df['close']).astype(int)
            targets = targets[:-lookahead]

            training_features = features_df.iloc[:-lookahead]

            if len(training_features) > self.config.LOOKBACK_PERIOD:
                await self.ai_models.train_models_async(training_features, targets)
                print("Models retrained successfully")

        except Exception as e:
            print(f"Model retraining error: {e}")

    async def _log_signal_async(self, signal):
        """Minimal logging asynchronously"""
        log_entry = {
            'timestamp': signal['timestamp'],
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'price': signal['current_price']
        }

        # Only log every 10th signal
        if self.cycle_count % 10 == 0:
            try:
                log_df = pd.DataFrame([log_entry])
                # Run file operation in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: log_df.to_csv('async_signals.csv', mode='a', header=False, index=False)
                )
            except Exception as e:
                print(f"Logging error: {e}")

    async def stop(self):
        """Stop the system asynchronously"""
        self.is_running = False

        # Cancel all running tasks
        for task in self.tasks:
            task.cancel()

        await self.telegram_bot.send_alert_async('INFO', "🛑 Async Trading System Stopped")
        print("Stopping system...")


# Keep existing SignalGenerator class (same as before)
class SignalGenerator:
    def __init__(self):
        self.signals = []
        self.stop_loss_pct = 0.002
        self.take_profit_pct = 0.004
        self.min_confidence = 0.6

    def generate_signal(self, prediction, confidence, current_price, features):
        """Simplified signal generation"""
        signal = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'current_price': current_price,
            'signal': 'HOLD',
            'strength': 'NEUTRAL'
        }

        if confidence < self.min_confidence:
            return signal

        # Simple signal logic
        if prediction > 0.6 and confidence > 0.7:
            signal['signal'] = 'BUY'
            signal['strength'] = 'STRONG'
            signal['stop_loss'] = current_price * (1 - self.stop_loss_pct)
            signal['take_profit'] = current_price * (1 + self.take_profit_pct)

        elif prediction > 0.55 and confidence > 0.6:
            signal['signal'] = 'BUY'
            signal['strength'] = 'MEDIUM'
            signal['stop_loss'] = current_price * (1 - self.stop_loss_pct)
            signal['take_profit'] = current_price * (1 + self.take_profit_pct)

        elif prediction < 0.4 and confidence > 0.7:
            signal['signal'] = 'SELL'
            signal['strength'] = 'STRONG'
            signal['stop_loss'] = current_price * (1 + self.stop_loss_pct)
            signal['take_profit'] = current_price * (1 - self.take_profit_pct)

        elif prediction < 0.45 and confidence > 0.6:
            signal['signal'] = 'SELL'
            signal['strength'] = 'MEDIUM'
            signal['stop_loss'] = current_price * (1 + self.stop_loss_pct)
            signal['take_profit'] = current_price * (1 - self.take_profit_pct)

        # Add basic technical context
        if len(features) > 0:
            latest = features.iloc[-1]
            signal['rsi'] = latest.get('rsi', 50)
            signal['volume_ratio'] = latest.get('volume_ratio', 1)

        return signal


async def setup_optimizations():
    """Apply system optimizations"""
    # Set TensorFlow to use CPU only
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Configure pandas for better memory usage
    pd.set_option('mode.chained_assignment', None)

    print("Async system optimizations applied")


async def main():
    # Apply optimizations
    await setup_optimizations()

    config = Config()
    trading_system = AsyncTradingSystem(config)

    try:
        await trading_system.start()
    except KeyboardInterrupt:
        await trading_system.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        await trading_system.stop()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
