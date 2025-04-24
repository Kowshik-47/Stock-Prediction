import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting
import matplotlib.pyplot as plt
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Define file paths relative to the app root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock.csv')

# Load model and metadata
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        logger.info("Model loaded successfully")
        return (
            saved_data['model'],
            saved_data['start_date'],
            saved_data['r_squared'],
            saved_data['mae'],
            saved_data['rmse'],
            saved_data.get('feature_columns', ['days', 'open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7'])
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None, None

# Load and preprocess data
def load_data():
    try:
        if not os.path.exists(DATA_PATH):
            logger.error(f"Data file not found at {DATA_PATH}")
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isnull().any():
            logger.error("Some dates could not be parsed in stock.csv")
            raise ValueError("Some dates could not be parsed in stock.csv")
        # Filter for close prices between $13 and $25 (optional, can be removed for broader data)
        data = data[(data['close'] >= 13) & (data['close'] <= 25)].copy()
        # No date filtering to allow all available years
        data = data.sort_values('date').copy()
        start_date = data['date'].min()  # Use earliest date in dataset
        data['days'] = (data['date'] - start_date).dt.days
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['ma7'] = data['close'].rolling(window=7).mean()
        logger.info("Data loaded and preprocessed successfully")
        return data, start_date
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, str(e)

# Load model and data
model, model_start_date, r_squared, mae, rmse, feature_columns = load_model()
data, data_start_date = load_data()
start_date = model_start_date if model_start_date else data_start_date

# Generate plot
def generate_plot(input_date, prediction, data):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['date'], data['close'], label='Historical Prices', color='blue')
        ax.scatter([input_date], [prediction], color='green', s=100, label='Prediction')
        ax.set_title(f'Stock Closing Price Prediction (R² = {r_squared:.3f})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price ($)')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        logger.info("Plot generated successfully")
        return plot_data
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise

@app.route('/')
def index():
    logger.info("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or data is None or isinstance(data, tuple):
        error = "Failed to load model or data" if model is None else data[1]
        logger.error(f"Predict endpoint error: {error}")
        return jsonify({'error': error}), 500
    
    try:
        input_date_str = request.form['date']
        input_date = pd.to_datetime(input_date_str)
        
        # Check if input_date is within the data range
        min_date = data['date'].min()
        max_date = data['date'].max()
        if input_date < min_date or input_date > max_date + pd.Timedelta(days=365):
            logger.warning(f"Input date {input_date_str} outside available data range ({min_date} to {max_date})")
            return jsonify({'error': f'Date must be between {min_date.date()} and one year after {max_date.date()}'}), 400
        
        # Ensure enough historical data for features (at least 7 days for ma7)
        earliest_required_date = input_date - pd.Timedelta(days=7)
        if earliest_required_date < min_date:
            logger.warning(f"Insufficient historical data for date {input_date_str}")
            return jsonify({'error': f'Insufficient historical data before {input_date_str}. Earliest supported date is {min_date + pd.Timedelta(days=7)}'}), 400
        
        days = (input_date - start_date).days
        prev_data = data[data['date'] <= input_date].tail(1)
        
        if prev_data.empty:
            logger.warning(f"No historical data available before date: {input_date_str}")
            return jsonify({'error': 'No historical data available before the selected date'}), 400
        
        prev_row = prev_data.iloc[0]
        input_features = []
        for col in feature_columns:
            if col == 'days':
                input_features.append(days)
            elif col in ['open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7']:
                value = prev_row[col] if col in prev_row and pd.notnull(prev_row[col]) else prev_row['close']
                input_features.append(value)
            else:
                logger.error(f"Unknown feature: {col}")
                return jsonify({'error': f'Unknown feature: {col}'}), 500
        
        input_features = np.array([input_features])
        prediction = model.predict(input_features)[0]
        
        # Generate plot
        plot_data = generate_plot(input_date, prediction, data)
        
        logger.info(f"Prediction made for date {input_date_str}: {prediction}")
        return jsonify({
            'prediction': round(prediction, 2),
            'r_squared': round(r_squared, 3),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'plot': plot_data
        })
    
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    logger.info(f"Serving static file: {filename}")
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=False)