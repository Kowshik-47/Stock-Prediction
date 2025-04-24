import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
from datetime import datetime
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

# Load and preprocess data
def load_and_preprocess_data():
    try:
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isnull().any():
            raise ValueError("Some dates could not be parsed in stock.csv")
        # Filter for close prices between $13 and $25
        data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
        start_date = datetime(2022, 1, 3)
        end_date = datetime(2022, 12, 30)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
        data['days'] = (data['date'] - start_date).dt.days
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['ma7'] = data['close'].rolling(window=7).mean()
        return data, start_date
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")

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
    
    print(f"Model saved to {MODEL_PATH}")
    print(f"R²: {r_squared:.3f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_model()
