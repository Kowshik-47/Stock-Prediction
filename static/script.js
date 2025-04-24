document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const date = document.getElementById('date').value;
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    
    // Reset UI
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    errorDiv.textContent = '';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: new FormData(document.getElementById('prediction-form'))
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('prediction-text').textContent = 
                `Predicted Closing Price on ${date}: $${data.prediction.toFixed(2)}`;
            document.getElementById('r-squared').textContent = `Model R² Score: ${data.r_squared.toFixed(3)}`;
            document.getElementById('mae').textContent = `Mean Absolute Error (MAE): ${data.mae.toFixed(2)}`;
            document.getElementById('rmse').textContent = `Root Mean Squared Error (RMSE): ${data.rmse.toFixed(2)}`;
            document.getElementById('plot').src = `data:image/png;base64,${data.plot}`;
            resultDiv.classList.remove('hidden');
        } else {
            errorDiv.textContent = data.error || 'An error occurred';
            errorDiv.classList.remove('hidden');
        }
    } catch (err) {
        errorDiv.textContent = 'Failed to connect to the server';
        errorDiv.classList.remove('hidden');
    }
});