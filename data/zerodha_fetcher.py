import os
import pandas as pd
from kiteconnect import KiteConnect

class ZerodhaDataFetcher:
    """Class to authenticate with Zerodha API and fetch Indian Stock Market historical/live data."""
    
    def __init__(self, api_key=None, access_token=None):
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")
        
        if not self.api_key:
            print("WARNING: KITE_API_KEY not found in environment!")
            
        self.kite = KiteConnect(api_key=self.api_key)
        
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            print("Zerodha Kite API authenticated using provided Access Token.")
        else:
            print("WARNING: No Access Token provided. Historical data fetching will fail unless authenticated manually.")
            print(f"Login URL: {self.kite.login_url()}")
            
    def fetch_historical_data(self, instrument_token, from_date, to_date, interval="minute"):
        """
        Fetches true exchange historical intraday data.
        Intervals: 'minute', '5minute', '15minute', 'day'
        """
        print(f"Fetching {interval} data for instrument {instrument_token} from {from_date} to {to_date}...")
        try:
            records = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('date', inplace=True)
                # Ensure the columns match our existing feature pipeline 'Open', 'High', 'Low', 'Close', 'Volume'
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data from Zerodha: {e}")
            return pd.DataFrame()
            
    def get_instrument_token(self, trading_symbol, exchange="NSE"):
        """Helper to get token required for historical database lookup."""
        try:
            instruments = self.kite.instruments(exchange)
            for instr in instruments:
                if instr['tradingsymbol'] == trading_symbol:
                    return instr['instrument_token']
            print(f"Instrument {trading_symbol} not found on {exchange}.")
        except Exception as e:
            print(f"Error fetching instruments. Have you logged in and entered the access token?: {e}")
        return None
