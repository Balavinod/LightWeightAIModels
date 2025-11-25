from schwab.auth import easy_client
from schwab.client import Client
import pickle

# Put your API key and app secret from https://developer.schwab.com
API_KEY = "czjhDYKBQOxEJGoJbRCI9YYoor6z68lP2ustFXCYGT25pDGC"
APP_SECRET = "PacTQniBEOUfCfXa6Z7krvAqLbfFCrrAiRKCpKWxVkOgzsUxo5ar1jNrC91Ngk4w"

if __name__ == '__main__':
# This opens browser, you log in once → saves tokens forever
    client = easy_client(
        API_KEY,
        APP_SECRET,
        callback_url="https://127.0.0.1:8183",  # default
        token_path="schwab_token.json"    # saves refresh token here automatically
)

# Test it
    print(client.get_account_numbers().json())