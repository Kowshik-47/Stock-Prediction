from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plotting
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'stock.csv')

# Load model and metadata
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
        return None, None, None, None, None, None, str(e)

# Load and preprocess data
def load_data():
    try:
        data = pd.read_csv(DATA_PATH)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isnull().any():
            raise ValueError("Some dates could not be parsed in stock.csv")
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
        return None, str(e)

# Load model and data
model, start_date, r_squared, mae, rmse, feature_columns = load_model()
data = load_data()

# Generate plot
def generate_plot(input_date, prediction, data):
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
    return plot_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or data is None or isinstance(data, tuple):
        error = "Failed to load model or data" if model is None else data[1]
        return jsonify({'error': error}), 500
    
    try:
        input_date_str = request.form['date']
        input_date = pd.to_datetime(input_date_str)
        if input_date < start_date or input_date > datetime(2022, 12, 31):
            return jsonify({'error': 'Date must be in 2022'}), 400
        
        days = (input_date - start_date).days
        prev_data = data[data['date'] <= input_date].tail(1)
        
        if prev_data.empty:
            return jsonify({'error': 'No historical data available before the selected date'}), 400
        
        prev_row = prev_data.iloc[0]
        input_features = []
        for col in feature_columns:
            if col == 'days':
                input_features.append(days)
            elif col in ['open', 'high', 'low', 'volume', 'lag1', 'lag2', 'ma7']:
                input_features.append(prev_row[col] if col in prev_row else prev_row['close'])
            else:
                return jsonify({'error': f'Unknown feature: {col}'}), 500
        
        input_features = np.array([input_features])
        prediction = model.predict(input_features)[0]
        
        # Generate plot
        plot_data = generate_plot(input_date, prediction, data)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'r_squared': round(r_squared, 3),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'plot': plot_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
