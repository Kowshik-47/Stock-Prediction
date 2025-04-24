import pandas as pd
from datetime import datetime

def preprocess_data(data, start_date):
    """
    Preprocess stock data: filter dates, add features.
    
    Args:
        data (pd.DataFrame): Raw stock data with 'date', 'open', 'high', 'low', 'close', 'volume'
        start_date (datetime): Reference date for 'days' feature
    
    Returns:
        pd.DataFrame: Processed data with additional features
    """
    # Convert date to datetime
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    if data['date'].isnull().any():
        raise ValueError("Invalid date format in data")
    
    # Filter for 2022
    end_date = datetime(2022, 12, 30)
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
    
    # Filter prices between $13 and $25
    data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
    
    # Add features
    data['days'] = (data['date'] - start_date).dt.days
    data['lag1'] = data['close'].shift(1)
    data['lag2'] = data['close'].shift(2)
    data['ma7'] = data['close'].rolling(window=7).mean()
    
    return data
