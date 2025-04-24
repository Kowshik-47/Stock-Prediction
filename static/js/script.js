document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const dateInput = document.getElementById('dateInput').value;
    const resultDiv = document.getElementById('result');
    const plotImg = document.getElementById('plot');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `date=${encodeURIComponent(dateInput)}`
        });
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error:', data.error);
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            plotImg.style.display = 'none';
        } else {
            resultDiv.innerHTML = `
                <p>Predicted Closing Price: $${data.prediction}</p>
                <p>R²: ${data.r_squared}</p>
                <p>MAE: ${data.mae}</p>
                <p>RMSE: ${data.rmse}</p>
            `;
            plotImg.src = `data:image/png;base64,${data.plot}`;
            plotImg.style.display = 'block';
        }
    } catch (error) {
        console.error('Network Error:', error);
        resultDiv.innerHTML = `<p style="color: red;">Network Error: ${error.message}</p>`;
        plotImg.style.display = 'none';
    }
});

// Optional: Set min/max dates for the date input
document.getElementById('dateInput').setAttribute('min', '2022-01-01'); // Adjust based on stock.csv
document.getElementById('dateInput').setAttribute('max', '2022-12-31'); // Adjust based on stock.csv
