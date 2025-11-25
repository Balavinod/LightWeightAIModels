# schwab_1min_scalping.py
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging
import os
import psutil
import gc
from dotenv import load_dotenv
import json
import os
from schwab import auth, client
import asyncio

load_dotenv()


# Memory-optimized logging configuration
class ScalpingLogger:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        """Configure efficient logging for 8GB RAM"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scalping_performance.log', mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ScalpingSystem')

        # Reduce log file size by rotating
        self.performance_logger = logging.getLogger('Performance')
        perf_handler = logging.FileHandler('scalping_signals.log')
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.performance_logger.addHandler(perf_handler)

    def log_memory_usage(self):
        """Log current memory usage"""
        memory = psutil.virtual_memory()
        self.logger.info(f"🧠 Memory: {memory.percent}% used, {memory.available // (1024 ** 3)}GB available")

    def log_scalping_signal(self, signal_data):
        """Log scalping signals efficiently"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal_data['signal'],
            'price': signal_data['price'],
            'confidence': signal_data['confidence'],
            'prediction': signal_data['prediction']
        }
        self.performance_logger.info(f"SIGNAL: {json.dumps(log_entry)}")

    def log_data_fetch(self, symbol, price, fetch_time):
        """Log data fetch performance"""
        self.logger.debug(f"📊 {symbol}: ${price:.2f} | Fetch: {fetch_time:.3f}s")


class ScalpingConfig:
    # Trading Configuration
    SYMBOL = "/NQ"  # NASDAQ Futures
    LOOKBACK_PERIOD = 30  # Reduced for 8GB RAM
    UPDATE_INTERVAL = 60  # 1-minute intervals

    # Risk Management
    STOP_LOSS_POINTS = 20
    TAKE_PROFIT_POINTS = 35
    MIN_CONFIDENCE = 0.65

    # AI Model Configuration (Lightweight)
    MODEL_WEIGHTS = {
        'xgboost': 0.6,
        'random_forest': 0.4
    }


class SchwabDataFetcher:
    def __init__(self, easy_client, config, logger):
        self.client = easy_client
        self.config = config
        self.logger = logger
        self.data_buffer = []
        self.last_price = 0

    async def fetch_realtime_quote(self):
        """Fetch real-time quote using easy_client"""
        try:
            start_time = time.time()

            # Get real-time quote
            response = await self.client.get_quotes(self.config.SYMBOL)
            if response.status == 200:
                quote_response_dict = await response.json()
            else:
                self.logger.error(f"API request failed with status code: {response.status}")

            fetch_time = time.time() - start_time

            if quote_response_dict and self.config.SYMBOL in quote_response_dict:
                quote_data = response[self.config.SYMBOL]

                current_data = {
                    'timestamp': datetime.now(),
                    'bid': quote_data.get('bidPrice', 0),
                    'ask': quote_data.get('askPrice', 0),
                    'last': quote_data.get('lastPrice', 0),
                    'volume': quote_data.get('totalVolume', 0),
                    'symbol': self.config.SYMBOL,
                    'fetch_time': fetch_time
                }

                # Calculate mid-price
                current_data['close'] = (current_data['bid'] + current_data['ask']) / 2
                current_data['high'] = max(current_data['bid'], current_data['ask'])
                current_data['low'] = min(current_data['bid'], current_data['ask'])

                self.last_price = current_data['close']

                # Add to buffer
                self.data_buffer.append(current_data)
                if len(self.data_buffer) > self.config.LOOKBACK_PERIOD:
                    self.data_buffer = self.data_buffer[-self.config.LOOKBACK_PERIOD:]

                self.logger.log_data_fetch(self.config.SYMBOL, current_data['close'], fetch_time)
                return current_data

            return None

        except Exception as e:
            self.logger.logger.error(f"❌ Quote fetch error: {e}")
            return None

    async def fetch_historical_data(self):
        """Fetch 1-minute historical data"""
        try:
            # Get historical data for the current day
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)

            historical_data = await self.client.get_price_history_every_minute(
                symbol=self.config.SYMBOL,
                need_extended_hours_data=True
            )
            if historical_data.status == 200:
                history_quote_response_dict = await historical_data.json()
            else:
                self.logger.error(f"API request failed with status code: {historical_data.status}")
            if historical_data and 'candles' in history_quote_response_dict:
                # Convert to our format
                processed_data = []
                for candle in history_quote_response_dict['candles']:
                    processed_data.append({
                        'timestamp': datetime.fromtimestamp(candle['datetime'] / 1000),
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': candle['volume']
                    })

                self.logger.logger.info(f"📈 Loaded {len(processed_data)} historical data points")
                return processed_data

            return None

        except Exception as e:
            self.logger.logger.error(f"❌ Historical data error: {e}")
            return None


class LightweightFeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    def create_scalping_features(self, df):
        """Create minimal features for 1-minute scalping"""
        if len(df) < 10:
            return df

        try:
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()

            # MINIMAL FEATURE SET for 8GB RAM
            df['returns_1min'] = df['close'].pct_change()
            df['price_momentum'] = df['close'].diff(3)

            # Fast moving averages
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
            df['ema_8'] = ta.trend.ema_indicator(df['close'], window=8)

            # Fast momentum
            df['rsi'] = ta.momentum.rsi(df['close'], window=10)

            # Volume features
            df['volume_sma'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Volatility
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=8)

            self.feature_columns = [col for col in df.columns if col not in
                                    ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'fetch_time']]

            return df.dropna()

        except Exception as e:
            print(f"❌ Feature engineering error: {e}")
            return df


class MemoryOptimizedAIModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self._initialize_lightweight_models()

    def _initialize_lightweight_models(self):
        """Initialize memory-efficient models for 8GB RAM"""
        # Lightweight XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=50,  # Reduced for memory
            max_depth=5,  # Shallower trees
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_jobs=2  # Use 2 cores only
        )

        # Lightweight Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=40,  # Reduced for memory
            max_depth=6,  # Shallower trees
            random_state=42,
            n_jobs=2  # Use 2 cores only
        )

    def train_models(self, features, targets):
        """Memory-efficient training"""
        try:
            if len(features) < 15:
                return

            feature_data = features[self.feature_columns].values

            if len(feature_data) > 0:
                # Train models
                self.models['xgboost'].fit(feature_data, targets)
                self.models['random_forest'].fit(feature_data, targets)

                # Clear memory
                del feature_data
                gc.collect()

        except Exception as e:
            print(f"🤖 Training error: {e}")

    def predict(self, current_data, feature_columns):
        """Fast prediction for scalping"""
        try:
            if len(current_data) == 0:
                return 0.5, 0.0, {}

            feature_data = current_data[feature_columns].iloc[-1:].values

            predictions = {}
            confidence_scores = {}

            # XGBoost prediction
            xgb_pred = self.models['xgboost'].predict_proba(feature_data)[0][1]
            predictions['xgboost'] = xgb_pred
            confidence_scores['xgboost'] = abs(xgb_pred - 0.5) * 2

            # Random Forest prediction
            rf_pred = self.models['random_forest'].predict_proba(feature_data)[0][1]
            predictions['random_forest'] = rf_pred
            confidence_scores['random_forest'] = abs(rf_pred - 0.5) * 2

            # Weighted ensemble
            ensemble_pred = 0
            total_weight = 0

            for model_name, pred in predictions.items():
                weight = self.config.MODEL_WEIGHTS.get(model_name, 0)
                ensemble_pred += pred * weight
                total_weight += weight

            if total_weight > 0:
                ensemble_pred /= total_weight

            # Overall confidence
            overall_confidence = sum(
                confidence_scores.get(model, 0) * self.config.MODEL_WEIGHTS.get(model, 0)
                for model in predictions.keys()
            )

            return ensemble_pred, overall_confidence, predictions

        except Exception as e:
            print(f"🤖 Prediction error: {e}")
            return 0.5, 0.0, {}

    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns


class ScalpingSystem:
    def __init__(self, easy_client):
        self.config = ScalpingConfig()
        self.logger = ScalpingLogger()
        self.data_fetcher = SchwabDataFetcher(easy_client, self.config, self.logger)
        self.feature_engineer = LightweightFeatureEngineer()
        self.ai_models = MemoryOptimizedAIModels(self.config)
        self.is_running = False
        self.cycle_count = 0
        self.signals_generated = 0

    def start(self):
        """Start the 1-minute scalping system"""
        self.logger.logger.info("🚀 Starting 1-Minute Scalping System")
        self.logger.logger.info("💻 Optimized for 8GB RAM + Intel i5")
        self.logger.logger.info(f"🎯 Symbol: {self.config.SYMBOL}")
        self.logger.logger.info(f"⏰ Interval: {self.config.UPDATE_INTERVAL} seconds")

        # Load initial historical data
        self.logger.logger.info("📈 Loading historical data...")
        historical_data = self.data_fetcher.fetch_historical_data()
        if historical_data:
            self.data_fetcher.data_buffer.extend(historical_data[-self.config.LOOKBACK_PERIOD:])
            self.logger.logger.info(f"✅ Loaded {len(historical_data)} historical points")

        self.is_running = True
        self._main_scalping_loop()

    def _main_scalping_loop(self):
        """Main 1-minute scalping loop"""
        while self.is_running:
            try:
                self.cycle_count += 1
                cycle_start = time.time()

                self.logger.logger.info(f"🔄 Cycle {self.cycle_count} - Fetching data...")

                # Fetch real-time data
                current_data = self.data_fetcher.fetch_realtime_quote()

                if current_data and len(self.data_fetcher.data_buffer) >= 15:
                    self._process_scalping_signal()

                # Memory management
                if self.cycle_count % 20 == 0:
                    self.logger.log_memory_usage()
                    gc.collect()

                # Calculate sleep time for exact 60-second intervals
                cycle_duration = time.time() - cycle_start
                sleep_time = max(1, self.config.UPDATE_INTERVAL - cycle_duration)

                self.logger.logger.info(f"⏰ Cycle completed in {cycle_duration:.2f}s, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                self.logger.logger.info("🛑 Stopping system...")
                break
            except Exception as e:
                self.logger.logger.error(f"❌ Cycle error: {e}")
                time.sleep(30)  # Wait 30 seconds on error

    def _process_scalping_signal(self):
        """Process scalping signal"""
        df = pd.DataFrame(self.data_fetcher.data_buffer)
        features_df = self.feature_engineer.create_scalping_features(df)

        if len(features_df) >= 15:
            # Set feature columns
            self.ai_models.set_feature_columns(self.feature_engineer.feature_columns)

            # Train periodically
            if self.cycle_count % 30 == 0:
                targets = (features_df['close'].shift(-1) > features_df['close']).astype(int)
                targets = targets[:-1]
                training_data = features_df.iloc[:-1]
                self.ai_models.train_models(training_data, targets)
                self.logger.logger.info("🤖 Models updated")

            # Get prediction
            prediction, confidence, model_details = self.ai_models.predict(
                features_df, self.feature_engineer.feature_columns
            )

            current_price = self.data_fetcher.last_price

            # Generate signal
            signal = self._generate_signal(prediction, confidence, current_price)

            if signal['signal'] != 'HOLD':
                self.signals_generated += 1
                self._log_and_display_signal(signal, model_details)

    def _generate_signal(self, prediction, confidence, current_price):
        """Generate scalping signal"""
        signal = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'price': current_price,
            'signal': 'HOLD'
        }

        if confidence < self.config.MIN_CONFIDENCE:
            return signal

        # Scalping logic
        if prediction > 0.68 and confidence > 0.70:
            signal.update({
                'signal': 'BUY',
                'stop_loss': current_price - self.config.STOP_LOSS_POINTS,
                'take_profit': current_price + self.config.TAKE_PROFIT_POINTS
            })
        elif prediction < 0.32 and confidence > 0.70:
            signal.update({
                'signal': 'SELL',
                'stop_loss': current_price + self.config.STOP_LOSS_POINTS,
                'take_profit': current_price - self.config.TAKE_PROFIT_POINTS
            })

        return signal

    def _log_and_display_signal(self, signal, model_details):
        """Log and display scalping signal"""
        # Log to file
        self.logger.log_scalping_signal(signal)

        # Display to console
        action_emoji = '🟢' if signal['signal'] == 'BUY' else '🔴'

        print(f"\n🎯 {action_emoji} SCALPING SIGNAL #{self.signals_generated} {action_emoji}")
        print(f"📈 Action: {signal['signal']}")
        print(f"💰 Price: ${signal['price']:.2f}")
        print(f"🎯 Confidence: {signal['confidence']:.1%}")
        print(f"📊 Prediction: {signal['prediction']:.3f}")
        print(f"🎯 Target: ${signal.get('take_profit', 0):.2f}")
        print(f"🛑 Stop: ${signal.get('stop_loss', 0):.2f}")
        print(f"⏰ Time: {signal['timestamp'].strftime('%H:%M:%S')}")
        print("=" * 50)

        self.logger.logger.info(f"✅ Signal #{self.signals_generated}: {signal['signal']} at ${signal['price']:.2f}")

    def stop(self):
        """Stop the system"""
        self.is_running = False
        self.logger.logger.info(f"🛑 System stopped. Generated {self.signals_generated} signals")


# Usage with your easy_client
if __name__ == "__main__":
    # Import and initialize your easy_client
    from schwab import auth  # Adjust import based on your setup

    load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")

# Initialize your easy_client (use your working code)
    api_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = os.getenv('callback_url')  # Your callback URL
    token_path = 'schwab_token.json'

    try:
    # Initialize easy_client (adjust based on your working code)
        c = auth.easy_client(
            api_key=api_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=token_path
        )

    # Test connection
        test_response = c.get_quotes('/NQ')
        if test_response:
            print("✅ Schwab connection successful!")

        # Start scalping system
            system = ScalpingSystem(c)
            system.start()

        else:
            print("❌ Schwab connection failed")

    except Exception as e:
        print(f"❌ Initialization error: {e}")