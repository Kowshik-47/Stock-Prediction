import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

# Load and preprocess data
def load_and_preprocess_data():
    try:
        if not os.path.exists(DATA_PATH):
            logger.error(f"Data file not found at {DATA_PATH}")
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isnull().any():
            logger.error("Some dates could not be parsed in stock.csv")
            raise ValueError("Some dates could not be parsed in stock.csv")
        # Filter for close prices between $13 and $25 (optional)
        data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
        # Use earliest date as start_date
        data = data.sort_values('date').copy()
        start_date = data['date'].min()
        data['days'] = (data['date'] - start_date).dt.days
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['ma7'] = data['close'].rolling(window=7).mean()
        logger.info("Data loaded and preprocessed successfully")
        return data, start_date
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

# Train model
def train_model():
    data, start_date = load_and_preprocess_data()
    feature_columns = ['days', 'open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7']
    
    # Drop rows with NaN values
    data = data.dropna()
    
    X = data[feature_columns]
    y = data['close']
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save model and metadata
    saved_data = {
        'model': model,
        'start_date': start_date,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'feature_columns': feature_columns
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(saved_data, f)
    
    logger.info(f"Model saved to {MODEL_PATH}")
    logger.info(f"R²: {r_squared:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_model()