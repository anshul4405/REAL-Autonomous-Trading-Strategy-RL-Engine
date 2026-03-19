import yfinance as yf
import pandas as pd
import os

class HistoricalDataFetcher:
    def __init__(self, ticker="AAPL", start_date="2020-01-01", end_date="2023-01-01", interval="1d"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval

    def fetch_data(self, save_csv=True, output_dir="data"):
        print(f"Fetching {self.interval} data for {self.ticker} from {self.start_date} to {self.end_date}...")
        try:
            # yfinance download
            data = yf.download(
                tickers=self.ticker, 
                start=self.start_date, 
                end=self.end_date, 
                interval=self.interval,
                progress=False
            )
            
            if data.empty:
                print(f"No data fetched for {self.ticker}.")
                return pd.DataFrame()
                
            # Handle MultiIndex columns from recent yfinance updates
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
                
            # Drop any missing values purely for cleaner start
            data.dropna(inplace=True)
            
            if save_csv:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{self.ticker}_{self.interval}_historical.csv")
                data.to_csv(file_path)
                print(f"Data saved to {file_path}")
                
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

class RealTimeDataFetcher:
    """
    Scaffolding class for future broker integrations.
    Supports Upstox / Zerodha Kite based on real-world requirements.
    """
    def __init__(self, api_key=None, secret_key=None, broker="zerodha"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.broker = broker
        
    def authenticate(self):
        print(f"Authenticating with {self.broker.capitalize()}...")
        # Add actual authentication logic here
        pass
        
    def get_live_price(self, ticker):
        print(f"Fetching live price for {ticker} from {self.broker.capitalize()} API...")
        # Placeholder
        return None

if __name__ == "__main__":
    fetcher = HistoricalDataFetcher(ticker="AAPL", start_date="2021-01-01", end_date="2022-01-01")
    df = fetcher.fetch_data()
    print(df.head())
