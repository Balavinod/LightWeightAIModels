from schwab import auth, client
import asyncio
import os
import logging
import json
import sys
from dotenv import load_dotenv
from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")

# Verify environment variables
api_key = os.getenv('app_key')
app_secret = os.getenv('app_secret')
callback_url = os.getenv('callback_url')

if not api_key or not app_secret or not callback_url:
    logger.error("Missing required environment variables. Check your .env file.")
    sys.exit(1)

try:
    logger.info("Attempting authentication with Schwab API...")

    # Create the client
    c = auth.easy_client(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path='schwab_token.json'
    )

    logger.info("Client object created successfully")

    # Import streaming module
    from schwab import streaming


    # Define a detailed handler function for crypto futures data
    def crypto_handler(message):
        """Process and display crypto futures streaming data"""
        try:
            # Print a separator for better visibility
            print("\n" + "=" * 80)
            print(f"CRYPTO FUTURES DATA AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print("=" * 80)

            # Try to format the message as JSON if possible
            if isinstance(message, dict) or isinstance(message, list):
                formatted_message = json.dumps(message, indent=2)
                print(formatted_message)
            else:
                print(f"Raw message: {message}")

            # Extract specific crypto data if available
            if isinstance(message, dict):
                # Look for common data fields in crypto futures data
                for key in ['symbol', 'bidPrice', 'askPrice', 'lastPrice', 'volume',
                            'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'netChange']:
                    if key in message:
                        print(f"{key}: {message[key]}")

                # Extract any symbol-specific information
                for symbol in ['BTC', 'ETH', 'LTC', 'BCH', 'XRP', 'NQ','AAPL']:
                    if symbol in str(message):
                        print(f"\nFound data for {symbol}")

            print("=" * 80 + "\n")
        except Exception as e:
            logger.error(f"Error processing crypto message: {e}")
            print(f"Raw message: {message}")


    async def main():
        try:
            # Create the streaming client
            streamer = streaming.StreamClient(c)

            # Login to the streaming service
            logger.info("Logging in to streaming service...")
            await streamer.login()
            logger.info("Successfully logged in to streaming service")

            # Log available methods for debugging
            stream_methods = [method for method in dir(streamer) if not method.startswith('_')]
            logger.info(f"Available methods: {', '.join(stream_methods)}")

            # Register handlers for crypto futures data
            logger.info("Setting up crypto futures data handlers...")

            # Try different handler registration methods
            if hasattr(streamer, 'add_futures_handler'):
                logger.info("Using add_futures_handler")
                streamer.add_futures_handler(crypto_handler)

            if hasattr(streamer, 'add_levelone_futures_handler'):
                logger.info("Using add_levelone_futures_handler")
                streamer.add_levelone_futures_handler(crypto_handler)

            if hasattr(streamer, 'add_crypto_handler'):
                logger.info("Using add_crypto_handler")
                streamer.add_crypto_handler(crypto_handler)

            # Use the on decorator if available
            if hasattr(streamer, 'on'):
                @streamer.on('LEVELONE_FUTURES')
                def on_futures(msg):
                    print("\nFUTURES DATA RECEIVED:")
                    crypto_handler(msg)

                @streamer.on('CRYPTOCURRENCY')
                def on_crypto(msg):
                    print("\nCRYPTO DATA RECEIVED:")
                    crypto_handler(msg)

                @streamer.on('QUOTE')
                def on_quote(msg):
                    if any(crypto in str(msg) for crypto in ['AAPL']):
                        print("\nCRYPTO QUOTE DATA RECEIVED:")
                        crypto_handler(msg)

            # Subscribe to crypto futures data
            logger.info("Subscribing to crypto futures data...")

            # Common crypto futures symbols
            # Format may vary by broker - these are common formats
            crypto_symbols = [
                '/BTC', '/ETH', '/LTC', '/NQZ25', 'AAPL'  # Common format
            ]

            # Try different subscription methods
            if hasattr(streamer, 'level_one_futures'):
                await streamer.level_one_futures(crypto_symbols)
                logger.info("Subscribed to level_one_futures")

            if hasattr(streamer, 'futures'):
                await streamer.futures(crypto_symbols)
                logger.info("Subscribed to futures")

            if hasattr(streamer, 'crypto'):
                await streamer.crypto(crypto_symbols)
                logger.info("Subscribed to crypto")

            if hasattr(streamer, 'subscribe'):
                # Try different service names for crypto data
                services = ['LEVELONE_FUTURES', 'CRYPTOCURRENCY', 'QUOTE']
                for service in services:
                    try:
                        logger.info(f"Subscribing to {service} with crypto symbols...")
                        await streamer.subscribe(service, crypto_symbols)
                        logger.info(f"Successfully subscribed to {service}")
                    except Exception as e:
                        logger.warning(f"Failed to subscribe to {service}: {e}")

            # Print confirmation message
            print("\n" + "*" * 80)
            print("* CRYPTO FUTURES STREAMING MONITOR ACTIVE")
            print("* Waiting for real-time crypto futures data from Schwab API...")
            print("* Data will appear below as it arrives")
            print("*" * 80 + "\n")

            # Keep the connection alive
            counter = 0
            while True:
                await asyncio.sleep(5)
                counter += 1
                if counter % 12 == 0:  # Every ~60 seconds
                    logger.info("Still listening for crypto futures data...")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")


    if __name__ == '__main__':
        logger.info("Starting main async loop")
        asyncio.run(main())

except Exception as e:
    logger.error(f"Authentication error: {e}")
    import traceback

    logger.error(f"Traceback: {traceback.format_exc()}")