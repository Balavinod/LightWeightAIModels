import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Download historical data for a symbol (e.g., 'AAPL')
ticker_symbol = 'NQ-F'
start_date = '2020-01-01'
end_date = '2025-01-01'

df_historical = yf.download(ticker_symbol, start=start_date, end=end_date)
df_historical.dropna(inplace=True) # Good practice to handle missing data

# 2. Your custom feature preparation step
# Assuming your prepare_features function takes a DataFrame
# This needs to be a function call that takes your df and returns the NumPy array
historical_features = ai_models.prepare_features(TechnicalIndicatorsValue, df_final)

# 3. Fit and save the scaler
scaler = StandardScaler()
scaler.fit(historical_features)
joblib.dump(scaler, 'scaler.pkl')

print(f"Historical features shape: {historical_features.shape}")
print("Scaler saved to scaler.pkl")