import requests
import pprint
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

access_token = "I0.b2F1dGgyLmNkYy5zY2h3YWIuY29t.iHP0rlONrwVJBVfHX3KA3oVLOaPWMBH8YoRCgXNI6DI@" # Must be a valid token from authentication
symbols = "NQZ24,/NQH25,/MNQZ24" # Comma-separated string of symbols with slashes

url = f"https://api.schwabapi.com/marketdata/v1/quotes?symbols=%2FNQ&fields=quote%2Creference&indicative=false"

headers = {
    "Authorization": f"Bearer {access_token}"
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    pprint.pprint(data)
    quote = response.json()
    if quote.get("assetMainType") == "FUTURE":
            logging.info(f"🟢 {symbol} Quote:")
            logging.info(f"   Last:    {quote['lastPrice']}")
            logging.info(f"   Bid:     {quote['bidPrice']} x {quote['bidSize']}")
            logging.info(f"   Ask:     {quote['askPrice']} x {quote['askSize']}")
            logging.info(f"   Change:  {quote['netChange']} ({quote['netPercentChangeInDouble']}%)")
            logging.info(f"   Volume:  {quote['totalVolume']}")
            logging.info(f"   Open:    {quote['openPrice']}")
            logging.info(f"   High:    {quote['highPrice']}")
            logging.info(f"   Low:     {quote['lowPrice']}")
            logging.info(f"   Close:   {quote['closePrice']} (previous)")
            logging.info(f"   Time:    {quote['tradeTime']}")
    else:
        logging.info("IF not working")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")