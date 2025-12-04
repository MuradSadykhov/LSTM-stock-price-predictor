# LSTM Stock Price Predictor
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Finance](https://img.shields.io/badge/Finance-ML%20Model-green)
A neural network project that predicts next-day stock closing prices using engineered time-series features and a multi-feature LSTM deep learning architecture.

This model is trained on real historical stock market data (Yahoo Finance API) and evaluates prediction accuracy on unseen test data.

---

## Project Highlights

| Feature | Details |
|--------|---------|
| Model Type | LSTM Neural Network (10 engineered features) |
| Data Source | Yahoo Finance |
| Prediction Target | Next-day closing price |
| Training Window | 7+ years of daily price data |
| Model Performance | 📉 Low RMSE + High directional accuracy |
| Customization | Predict *any* stock by changing the ticker |

---

## Model Examples

### Apple (AAPL)

- **Mean Absolute Error (MAE):** ~3.68  
- **Root Mean Squared Error (RMSE):** ~5.23  
- **Avg % Error:** ~1.67%  
- **Predictions within ±10% of actual:** **99.1%** of test days

 Sample Output:
#### AAPL Actual vs Predicted Pricing
Below is the model’s actual vs predicted pricing for AAPL — notice the strong trend alignment across the full year:
<img src="images/AAPL_prediction.png" width="800">

---

###  Disney (DIS)

- **Mean Absolute Error (MAE):** ~4.49  
- **Root Mean Squared Error (RMSE):** ~5.42  
- **Avg % Error:** ~4.17%  
- **Predictions within ±10% of actual:** **92.7%** of test days

 Sample Output:
#### DIS Actual vs Predicted Pricing
The DIS chart shows slightly higher prediction error due to increased volatility in price movements:
<img src="images/DIS_prediction.png" width="800">

---

## How It Works

✔ Scales numerical features using MinMaxScaler  
✔ Creates supervised time-series structure  
✔ Trains deep LSTM layers to detect price patterns  
✔ Predicts future prices using sliding windows  
✔ Inverse transforms predictions to USD values  

---

## Technologies Used

| Category | Packages |
|--------|---------|
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy, Scikit-Learn |
| Visualization | Matplotlib |
| Data Source | yfinance |

---

## Future Improvements

- Incorporate volume & macro-economic data
- Expand prediction horizon (1–7 days ahead)
- Hyperparameter tuning (Bayesian Optimization)
- Add transformer-based architecture for comparison

---

## Installation

Clone repo:
```sh
git clone https://github.com/MuradSadykhov/LSTM-stock-price-predictor.git
cd LSTM-stock-price-predictor
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
pip install -r requirements.txt
## Contact

If you'd like to connect or discuss the project:

**Murad Sadykhov**  
Email: [sdsmura@gmail.com]  
LinkedIn: https://www.linkedin.com/in/murad-sadykhov/

---

## License
This project is open-source under the MIT License — feel free to fork and explore!
