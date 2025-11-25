# async_scalping_system.py
import pandas as pd
import numpy as np
import time
import asyncio
import aiohttp
import concurrent.futures
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

load_dotenv()


# Async logging configuration
class AsyncScalpingLogger:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        """Configure efficient async logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('async_scalping.log', mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AsyncScalping')

    async def log_async_operation(self, operation, duration, success=True):
        """Log async operation performance"""
        status = "✅" if success else "❌"
        self.logger.info(f"{status} {operation}: {duration:.3f}s")

    def log_memory_usage(self):
        """Log current memory usage"""
        memory = psutil.virtual_memory()
        self.logger.info(f"🧠 Memory: {memory.percent}% used, {memory.available // (1024 ** 3)}GB available")


class AsyncSchwabDataFetcher:
    def __init__(self, easy_client, config, logger):
        self.client = easy_client
        self.config = config
        self.logger = logger
        self.data_buffer = []
        self.last_price = 0
        self.session = None

    async def initialize_session(self):
        """Initialize aiohttp session for async requests"""
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    async def fetch_realtime_quote_async(self):
        """Fetch real-time quote asynchronously"""
        try:
            start_time = time.time()

            # Run synchronous get_quotes in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(
                    pool,
                    self.client.get_quotes,
                    self.config.SYMBOL
                )

            fetch_time = time.time() - start_time

            if response and self.config.SYMBOL in response:
                quote_data = response[self.config.SYMBOL]

                current_data = {
                    'timestamp': datetime.now(),
                    'bid': quote_data.get('bidPrice', 0),
                    'ask': quote_data.get('askPrice', 0),
                    'last': quote_data.get('lastPrice', 0),
                    'volume': quote_data.get('totalVolume', 0),
                    'symbol': self.config.SYMBOL,
                    'fetch_time': fetch_time,
                    'data_type': 'realtime'
                }

                # Calculate mid-price
                current_data['close'] = (current_data['bid'] + current_data['ask']) / 2
                current_data['high'] = max(current_data['bid'], current_data['ask'])
                current_data['low'] = min(current_data['bid'], current_data['ask'])

                self.last_price = current_data['close']

                await self.logger.log_async_operation("Realtime Quote", fetch_time, True)
                return current_data

            await self.logger.log_async_operation("Realtime Quote", fetch_time, False)
            return None

        except Exception as e:
            self.logger.logger.error(f"❌ Async quote fetch error: {e}")
            return None

    async def fetch_historical_data_async(self):
        """Fetch 1-minute historical data asynchronously"""
        try:
            start_time = time.time()

            # Run synchronous get_price_history in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                historical_data = await loop.run_in_executor(
                    pool,
                    self._get_price_history_wrapper,
                    self.config.SYMBOL
                )

            fetch_time = time.time() - start_time

            if historical_data and 'candles' in historical_data:
                processed_data = []
                for candle in historical_data['candles']:
                    processed_data.append({
                        'timestamp': datetime.fromtimestamp(candle['datetime'] / 1000),
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': candle['volume'],
                        'data_type': 'historical'
                    })

                await self.logger.log_async_operation("Historical Data", fetch_time, True)
                self.logger.logger.info(f"📈 Loaded {len(processed_data)} historical points")
                return processed_data

            await self.logger.log_async_operation("Historical Data", fetch_time, False)
            return None

        except Exception as e:
            self.logger.logger.error(f"❌ Async historical data error: {e}")
            return None

    def _get_price_history_wrapper(self, symbol):
        """Wrapper for synchronous price history call"""
        try:
            return self.client.get_price_history(
                symbol=symbol,
                period_type='day',
                period=1,
                frequency_type='minute',
                frequency=1,
                need_extended_hours_data=True
            )
        except Exception as e:
            self.logger.logger.error(f"❌ Price history wrapper error: {e}")
            return None

    async def fetch_all_data_async(self):
        """Fetch both real-time and historical data concurrently"""
        start_time = time.time()

        # Run both fetches concurrently
        realtime_task = asyncio.create_task(self.fetch_realtime_quote_async())
        historical_task = asyncio.create_task(self.fetch_historical_data_async())

        # Wait for both to complete
        realtime_data, historical_data = await asyncio.gather(
            realtime_task,
            historical_task,
            return_exceptions=True
        )

        total_fetch_time = time.time() - start_time

        # Handle results
        results = {}

        if not isinstance(realtime_data, Exception) and realtime_data:
            results['realtime'] = realtime_data
            # Add to buffer
            self.data_buffer.append(realtime_data)
            if len(self.data_buffer) > self.config.LOOKBACK_PERIOD:
                self.data_buffer = self.data_buffer[-self.config.LOOKBACK_PERIOD:]

        if not isinstance(historical_data, Exception) and historical_data:
            results['historical'] = historical_data
            # Replace historical portion of buffer
            self._merge_historical_data(historical_data)

        self.logger.logger.info(f"🔄 All data fetched in {total_fetch_time:.3f}s")
        return results

    def _merge_historical_data(self, historical_data):
        """Merge historical data with real-time buffer"""
        # Keep only the most recent real-time data
        realtime_data = [d for d in self.data_buffer if d.get('data_type') == 'realtime']

        # Combine with historical data (remove duplicates by timestamp)
        historical_df = pd.DataFrame(historical_data)
        if not historical_df.empty:
            # Get unique historical data points
            unique_historical = historical_df.drop_duplicates(subset=['timestamp']).to_dict('records')

            # Combine and sort by timestamp
            combined_data = realtime_data + unique_historical
            combined_data.sort(key=lambda x: x['timestamp'])

            # Keep only the most recent data
            self.data_buffer = combined_data[-self.config.LOOKBACK_PERIOD:]


class AsyncScalpingConfig:
    SYMBOL = "/NQ"
    LOOKBACK_PERIOD = 40
    UPDATE_INTERVAL = 60

    STOP_LOSS_POINTS = 20
    TAKE_PROFIT_POINTS = 35
    MIN_CONFIDENCE = 0.65

    MODEL_WEIGHTS = {
        'xgboost': 0.6,
        'random_forest': 0.4
    }


class AsyncFeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    async def create_features_async(self, df):
        """Create features asynchronously"""
        if len(df) < 10:
            return df

        try:
            # Run feature engineering in thread pool
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    self._create_features_sync,
                    df.copy()
                )
            return result

        except Exception as e:
            print(f"❌ Async feature engineering error: {e}")
            return df

    def _create_features_sync(self, df):
        """Synchronous feature engineering"""
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()

        # Core features
        df['returns_1min'] = df['close'].pct_change()
        df['price_momentum'] = df['close'].diff(3)

        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['ema_8'] = ta.trend.ema_indicator(df['close'], window=8)

        df['rsi'] = ta.momentum.rsi(df['close'], window=10)

        df['volume_sma'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=8)

        self.feature_columns = [col for col in df.columns if col not in
                                ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'fetch_time',
                                 'data_type']]

        return df.dropna()


class AsyncAIModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=2
        )

        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=40,
            max_depth=6,
            random_state=42,
            n_jobs=2
        )

    async def train_models_async(self, features, targets):
        """Train models asynchronously"""
        try:
            if len(features) < 15:
                return

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool,
                    self._train_models_sync,
                    features,
                    targets
                )

        except Exception as e:
            print(f"🤖 Async training error: {e}")

    def _train_models_sync(self, features, targets):
        """Synchronous model training"""
        feature_data = features[self.feature_columns].values

        if len(feature_data) > 0:
            self.models['xgboost'].fit(feature_data, targets)
            self.models['random_forest'].fit(feature_data, targets)

            del feature_data
            gc.collect()

    async def predict_async(self, current_data, feature_columns):
        """Make prediction asynchronously"""
        try:
            if len(current_data) == 0:
                return 0.5, 0.0, {}

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    self._predict_sync,
                    current_data,
                    feature_columns
                )
            return result

        except Exception as e:
            print(f"🤖 Async prediction error: {e}")
            return 0.5, 0.0, {}

    def _predict_sync(self, current_data, feature_columns):
        """Synchronous prediction"""
        feature_data = current_data[feature_columns].iloc[-1:].values

        predictions = {}
        confidence_scores = {}

        xgb_pred = self.models['xgboost'].predict_proba(feature_data)[0][1]
        predictions['xgboost'] = xgb_pred
        confidence_scores['xgboost'] = abs(xgb_pred - 0.5) * 2

        rf_pred = self.models['random_forest'].predict_proba(feature_data)[0][1]
        predictions['random_forest'] = rf_pred
        confidence_scores['random_forest'] = abs(rf_pred - 0.5) * 2

        ensemble_pred = 0
        total_weight = 0

        for model_name, pred in predictions.items():
            weight = self.config.MODEL_WEIGHTS.get(model_name, 0)
            ensemble_pred += pred * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        overall_confidence = sum(
            confidence_scores.get(model, 0) * self.config.MODEL_WEIGHTS.get(model, 0)
            for model in predictions.keys()
        )

        return ensemble_pred, overall_confidence, predictions

    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns


class AsyncScalpingSystem:
    def __init__(self, easy_client):
        self.config = AsyncScalpingConfig()
        self.logger = AsyncScalpingLogger()
        self.data_fetcher = AsyncSchwabDataFetcher(easy_client, self.config, self.logger)
        self.feature_engineer = AsyncFeatureEngineer()
        self.ai_models = AsyncAIModels(self.config)
        self.is_running = False
        self.cycle_count = 0
        self.signals_generated = 0

    async def start(self):
        """Start the async scalping system"""
        self.logger.logger.info("🚀 Starting Async 1-Minute Scalping System")
        self.logger.logger.info("⚡ Concurrent data fetching enabled")

        # Initialize async session
        await self.data_fetcher.initialize_session()

        # Load initial data
        self.logger.logger.info("📈 Loading initial data concurrently...")
        initial_data = await self.data_fetcher.fetch_all_data_async()

        self.is_running = True
        await self._main_async_loop()

    async def _main_async_loop(self):
        """Main async trading loop"""
        while self.is_running:
            try:
                self.cycle_count += 1
                cycle_start = time.time()

                self.logger.logger.info(f"🔄 Cycle {self.cycle_count} - Fetching data concurrently...")

                # Fetch all data asynchronously
                data_results = await self.data_fetcher.fetch_all_data_async()

                if data_results and len(self.data_fetcher.data_buffer) >= 15:
                    await self._process_async_signal()

                # Memory management
                if self.cycle_count % 20 == 0:
                    self.logger.log_memory_usage()
                    gc.collect()

                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(1, self.config.UPDATE_INTERVAL - cycle_duration)

                self.logger.logger.info(f"⏰ Cycle completed in {cycle_duration:.2f}s, sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

            except KeyboardInterrupt:
                self.logger.logger.info("🛑 Stopping async system...")
                break
            except Exception as e:
                self.logger.logger.error(f"❌ Async cycle error: {e}")
                await asyncio.sleep(30)

    async def _process_async_signal(self):
        """Process scalping signal asynchronously"""
        df = pd.DataFrame(self.data_fetcher.data_buffer)

        # Create features asynchronously
        features_df = await self.feature_engineer.create_features_async(df)

        if len(features_df) >= 15:
            # Set feature columns
            self.ai_models.set_feature_columns(self.feature_engineer.feature_columns)

            # Train periodically
            if self.cycle_count % 30 == 0:
                targets = (features_df['close'].shift(-1) > features_df['close']).astype(int)
                targets = targets[:-1]
                training_data = features_df.iloc[:-1]
                await self.ai_models.train_models_async(training_data, targets)
                self.logger.logger.info("🤖 Models updated asynchronously")

            # Get prediction asynchronously
            prediction, confidence, model_details = await self.ai_models.predict_async(
                features_df, self.feature_engineer.feature_columns
            )

            current_price = self.data_fetcher.last_price

            # Generate signal
            signal = await self._generate_signal_async(prediction, confidence, current_price)

            if signal['signal'] != 'HOLD':
                self.signals_generated += 1
                await self._display_signal_async(signal, model_details)

    async def _generate_signal_async(self, prediction, confidence, current_price):
        """Generate signal asynchronously"""
        signal = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'price': current_price,
            'signal': 'HOLD'
        }

        if confidence < self.config.MIN_CONFIDENCE:
            return signal

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

    async def _display_signal_async(self, signal, model_details):
        """Display signal asynchronously"""
        action_emoji = '🟢' if signal['signal'] == 'BUY' else '🔴'

        print(f"\n🎯 {action_emoji} ASYNC SCALPING SIGNAL #{self.signals_generated} {action_emoji}")
        print(f"📈 Action: {signal['signal']}")
        print(f"💰 Price: ${signal['price']:.2f}")
        print(f"🎯 Confidence: {signal['confidence']:.1%}")
        print(f"📊 Prediction: {signal['prediction']:.3f}")
        print(f"🎯 Target: ${signal.get('take_profit', 0):.2f}")
        print(f"🛑 Stop: ${signal.get('stop_loss', 0):.2f}")
        print(f"⏰ Time: {signal['timestamp'].strftime('%H:%M:%S')}")
        print("=" * 50)

        self.logger.logger.info(
            f"✅ Async Signal #{self.signals_generated}: {signal['signal']} at ${signal['price']:.2f}")

    async def stop(self):
        """Stop the async system"""
        self.is_running = False
        await self.data_fetcher.close_session()
        self.logger.logger.info(f"🛑 Async system stopped. Generated {self.signals_generated} signals")


# Main async function
async def main():
    from schwab import auth  # Adjust based on your setup
    load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")

    api_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = 'https://127.0.0.1:8183'
    token_path = 'schwab_token.json'

    try:
        # Initialize easy_client
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

            # Start async system
            system = AsyncScalpingSystem(c)
            await system.start()

        else:
            print("❌ Schwab connection failed")

    except Exception as e:
        print(f"❌ Initialization error: {e}")


if __name__ == "__main__":
    # Install required: pip install aiohttp
    asyncio.run(main())