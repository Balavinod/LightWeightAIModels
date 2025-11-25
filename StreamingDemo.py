from schwab import auth, client
import asyncio
import os
import logging
import json
from dotenv import load_dotenv

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
#load_dotenv("C:\\Users\\Administrator\\tradingSignals.env")
load_dotenv("C:\\Users\\Administrator\\PycharmProjects\\PythonProject\\tradingSignals.env")
# Step 1: Check if token file exists and inspect it
token_path = 'schwab_token.json'
if os.path.exists(token_path):
    try:
        with open(token_path, 'r') as f:
            token_data = json.load(f)
            logger.info(
                f"Token file exists. Access token expiry: {token_data.get('access_token_expires_at', 'unknown')}")

            # If token is old, remove it to force re-authentication
            logger.info("Removing old token file to force re-authentication")
            os.remove(token_path)
            logger.info(f"Deleted token file: {token_path}")
    except Exception as e:
        logger.error(f"Error reading token file: {e}")
        logger.info("Removing potentially corrupted token file")
        os.remove(token_path)

# Step 2: Verify environment variables
api_key = os.getenv('app_key')
app_secret = os.getenv('app_secret')
callback_url = os.getenv('callback_url')

logger.info(f"API Key exists: {bool(api_key)}")
logger.info(f"App Secret exists: {bool(app_secret)}")
logger.info(f"Callback URL: {callback_url}")

if not api_key or not app_secret or not callback_url:
    logger.error("Missing required environment variables. Check your .env file.")
    exit(1)

# Step 3: Try authentication with correct parameters
try:
    logger.info("Attempting authentication with Schwab API...")

    # Check the Schwab API documentation for the correct parameters
    # Using client_from_login_flow instead of client_from_manual_flow
    c = auth.client_from_login_flow(
        api_key=api_key,
        api_secret=app_secret,  # Using api_secret instead of app_secret
        callback_url=callback_url,  # Using callback_url instead of redirect_uri
        token_path=token_path
    )

    # Alternative: Try the easy_client method again with explicit parameters
    # c = auth.easy_client(
    #     api_key=api_key,
    #     app_secret=app_secret,
    #     callback_url=callback_url,
    #     token_path=token_path
    # )

    # After authentication, proceed with streaming
    from schwab import streaming


    def my_handler(message):
        print(message)


    async def main():
        try:
            streamer = streaming.StreamClient(c)
            logger.info("Attempting to login to streaming service...")
            await streamer.login()

            logger.info("Adding handlers and subscribing to level one stocks...")
            streamer.add_level_one_stock_handler(my_handler)
            await streamer.level_one_stocks(keys=['AAPL'], fields=[*range(0, 50)])

            while True:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            logger.error(f"Error details: {str(e)}")


    if __name__ == '__main__':
        logger.info("Starting main async loop")
        asyncio.run(main())

except Exception as e:
    logger.error(f"Authentication error: {e}")

    # Provide detailed troubleshooting guidance
    logger.info("\nTROUBLESHOOTING STEPS:")
    logger.info("1. Verify your API credentials in the .env file")
    logger.info("2. Ensure your Schwab developer account is active")
    logger.info("3. Check if your API key has the correct permissions")
    logger.info("4. Verify the callback URL matches exactly what's registered in your Schwab developer account")
    logger.info("5. Make sure your app is approved for the API endpoints you're trying to access")
    logger.info("6. Check the Schwab API documentation for any recent changes to authentication methods")