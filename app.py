import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Define file paths relative to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock.csv')

# Load the pre-trained model and metadata
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        return (
            saved_data['model'],
            saved_data['start_date'],
            saved_data['r_squared'],
            saved_data['mae'],
            saved_data['rmse'],
            saved_data.get('feature_columns', ['days', 'open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7'])
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None

# Load and preprocess historical data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isnull().any():
            raise ValueError("Some dates could not be parsed in stock.csv.")
        # Filter for single stock (close prices between $13 and $25)
        data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
        start_date = datetime(2022, 1, 3)
        end_date = datetime(2022, 12, 30)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)].copy()
        data['days'] = (data['date'] - start_date).dt.days
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['ma7'] = data['close'].rolling(window=7).mean()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load model and data
model, start_date, r_squared, mae, rmse, feature_columns = load_model()
data = load_data()

# Streamlit app
st.title("Stock Price Prediction")
st.write("Enter a date in 2022 to predict the stock closing price using a pre-trained Random Forest model.")

if model is None or start_date is None or data is None or feature_columns is None:
    st.error("Failed to initialize the model or data. Please check the model.pkl and stock.csv files.")
else:
    # Date input
    st.subheader("Select a Date for Prediction")
    input_date = st.date_input(
        "Choose a date",
        min_value=start_date,
        max_value=datetime(2022, 12, 31),
        value=datetime(2022, 12, 31)
    )
    
    # Process prediction
    try:
        input_date = pd.to_datetime(input_date)
        days = (input_date - start_date).days
        
        # Find the closest previous date in the data
        prev_data = data[data['date'] <= input_date].tail(1)
        if prev_data.empty:
            st.error("No historical data available before the selected date.")
        else:
            prev_row = prev_data.iloc[0]
            
            # Prepare features for prediction
            input_features = []
            for col in feature_columns:
                if col == 'days':
                    input_features.append(days)
                elif col in ['open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7']:
                    input_features.append(prev_row[col] if col in prev_row else prev_row['close'])
                else:
                    raise ValueError(f"Unknown feature: {col}")
            
            input_features = np.array([input_features])
            
            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Display prediction and metrics
            st.subheader("Prediction Result")
            st.markdown(f"**Predicted Closing Price on {input_date.strftime('%Y-%m-%d')}:** ${prediction:.2f}")
            st.markdown(f"**Model R² Score:** {r_squared:.3f}")
            st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}")
            st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
            
            # Plot historical data and prediction
            st.subheader("Historical Data and Prediction")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(data['date'], data['close'], label='Historical Prices', color='blue')
            
            # Plot prediction point
            ax.scatter([input_date], [prediction], color='green', s=100, label='Prediction')
            
            ax.set_title(f'Stock Closing Price Prediction (R² = {r_squared:.3f})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price ($)')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Create a WSGI callable for gunicorn
server = st.server.server
