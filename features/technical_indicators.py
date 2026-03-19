import pandas as pd
import numpy as np

def add_moving_averages(df: pd.DataFrame, short_window=10, long_window=50):
    df[f'MA_{short_window}'] = df['Close'].rolling(window=short_window).mean()
    df[f'MA_{long_window}'] = df['Close'].rolling(window=long_window).mean()
    return df

def add_rsi(df: pd.DataFrame, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, slow=26, fast=12, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def add_bollinger_bands(df: pd.DataFrame, window=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * num_std)
    df['BB_Lower'] = df['BB_Middle'] - (std * num_std)
    return df

def add_derived_features(df: pd.DataFrame):
    df['Price_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Price_Return'].rolling(window=10).std()
    
    if 'MA_10' in df.columns and 'MA_50' in df.columns:
        df['Trend_Strength'] = (df['MA_10'] - df['MA_50']) / df['Close']
    else:
        df['Trend_Strength'] = np.nan
        
    return df

def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all technical indicators and derived features to the raw dataframe.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_derived_features(df)
    
    df.dropna(inplace=True)
    return df
