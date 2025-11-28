from schwab import auth, client
import asyncio
import os
import logging
import json
import sys
from dotenv import load_dotenv
import pprint
from datetime import datetime
import websockets
import pandas as pd
from schwab.client import base as schwab_base


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

def main():
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
    load_dotenv("C:\\Users\\Deppa\\tradingSignals.env")

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
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        import traceback

    #resp = c.get_quotes('/NQ')
    #data = resp.json()
    #pprint.pprint(data)
    resp1 = c.get_price_history_every_minute(
            '/NQ',  # max allowed for minute data
            need_extended_hours_data=True  # futures trade almost 24h
    )
    data1 = resp1.json()
    #pprint.pprint(data1)
    resp2 = c.get_quotes(symbols="/NQ") # futures trade almost 24h
    data_quote = resp2.json()
    df_quotes = pd.DataFrame.from_dict(data_quote, orient='index')
    print(df_quotes.tail())
    data_points = []
#print(hist_data)
    if 'candles' in data1:
        for candle in data1['candles']:
            data_points.append({
                'datetime': datetime.fromtimestamp(candle['datetime'] / 1000),
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            })


    df_hist_data = pd.DataFrame(data_points)
    print(df_hist_data.tail())
    df_hist_data['datetime'] = pd.to_datetime(df_hist_data['datetime'], unit='ms')
    df_hist_data.set_index('datetime', inplace=True)
    backup_file_path ="C:\\Users\\Deppa\\PycharmProjects\\LightWeightAIModels\\backup\\schwab_candles_backup.csv"
    print(backup_file_path)
    try:
        append_new_data_only(
        df_hist_data,  # Assign by keyword for clarity
        backup_file_path,
        index_label='datetime'
        )
        print("backup of data is completed")
    except Exception as e:
        print(f"FATAL ERROR during backup of data: {e}")


    def pprint_first_n_lines(data1, num_lines=20):
        # 1. Format the entire object into a single string
        full_output_string = pprint.pformat(data1)

        # 2. Split the string into a list of individual lines
        lines = full_output_string.splitlines()

        # 3. Slice the list to keep only the first 'num_lines'
        for line in lines[:num_lines]:
            print(line)

        # Optional: Add an indicator that the output was truncated
        if len(lines) > num_lines:
            print(f"... Output truncated after {num_lines} lines ...")


    pprint_first_n_lines(data1, num_lines=20)
    pprint_first_n_lines(data_quote, num_lines=20)

if __name__ == "__main__":
    main()

