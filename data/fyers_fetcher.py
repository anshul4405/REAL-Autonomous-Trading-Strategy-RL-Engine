import os
from fyers_apiv3 import fyersModel

class FyersDataFetcher:
    """Class to authenticate with Fyers API (Free) for Indian Stock Market data."""
    
    def __init__(self):
        self.client_id = os.getenv("FYERS_APP_ID")
        self.secret_key = os.getenv("FYERS_SECRET_ID")
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        
        if self.client_id and self.access_token:
            self.fyers = fyersModel.FyersModel(client_id=self.client_id, is_async=False, token=self.access_token, log_path="")
            print("Fyers API Authenticated Successfully.")
        else:
            print("WARNING: Fyers Keys not fully set up in .env")
            self.fyers = None

    def fetch_historical_data(self, symbol="NSE:RELIANCE-EQ", resolution="5", range_from="2023-10-01", range_to="2023-10-30"):
        if not self.fyers: return None
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1"
        }
        response = self.fyers.history(data=data)
        if response.get("s") == "ok":
            import pandas as pd
            df = pd.DataFrame(response["candles"], columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df.set_index('date', inplace=True)
            return df
        return None
