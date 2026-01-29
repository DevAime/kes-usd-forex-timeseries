# KES Exchange Rate Time Series Analysis (2015-2025)

A comprehensive statistical and machine learning analysis of the Kenyan Shilling exchange rate against major international currencies (USD, EUR, GBP) using advanced time series methods and deep learning forecasting.

>  **Live Demo**: [Interactive Dashboard](https://kes-usd-forex-timeseries-rbgqw6jkakwstauwsptzhu.streamlit.app/)

## Table of Contents

- [Overview](#overview)
- [Live Dashboard](#live-dashboard)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Components](#analysis-components)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Interactive Dashboard](#interactive-dashboard)
- [Data Source](#data-source)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Overview

This project provides a complete end-to-end analysis of Kenyan Shilling (KES) exchange rates over a 10-year period (September 2015 - September 2025). The analysis combines traditional econometric methods with state-of-the-art machine learning techniques to deliver accurate forecasts and actionable insights for:

- Corporate treasury departments
- Import/export businesses
- Financial institutions
- Portfolio managers
- Policy makers

**Key Achievement:** LSTM neural networks achieved 95-99% accuracy (R²) in exchange rate prediction, dramatically outperforming traditional statistical methods.

<img width="1919" height="853" alt="image" src="https://github.com/user-attachments/assets/ccb74c3b-760d-4c85-9ba9-5275327e797d" />


## Live Dashboard

 **Access the live interactive dashboard here**: [https://kes-usd-forex-timeseries-rbgqw6jkakwstauwsptzhu.streamlit.app/](https://kes-usd-forex-timeseries-rbgqw6jkakwstauwsptzhu.streamlit.app/)

The dashboard provides:
- Real-time currency selection and analysis
- Interactive visualizations with zoom and hover capabilities
- LSTM-powered 6-month forecasts
- Strategic recommendations by stakeholder type
- Mobile-responsive design

<img width="1919" height="857" alt="image" src="https://github.com/user-attachments/assets/d86afba6-6092-4230-98d1-a8b4c2ec6fc2" />

## Project Structure

```
kes-exchange-rate-analysis/
│
├── data/
│   ├── USD_KES Historical Data.csv      
│   ├── EUR_KES Historical Data.csv      
│   └── GBP_KES Historical Data.csv      
│
├── kesforextimeseries.ipynb             # Main Jupyter notebook with complete analysis
├── app.py                               # Streamlit web dashboard
├── requirements.txt                     
├── README.md                           
                         
```

## Features

### Comprehensive Statistical Analysis

- **Stationarity Testing**: ADF and KPSS tests with first-order differencing
- **Time Series Decomposition**: Trend, seasonal, and noise component separation
- **Correlation Analysis**: Price and returns correlation matrices
- **Volatility Modeling**: Rolling 30-day standard deviation analysis
- **Seasonality Detection**: Monthly pattern identification
- **Signal-to-Noise Ratio**: Forecast reliability assessment

<img width="1400" height="1200" alt="image" src="https://github.com/user-attachments/assets/db7c2d03-34d6-43ca-8abe-422e69a55357" />

<img width="1400" height="1200" alt="image" src="https://github.com/user-attachments/assets/d98fd857-2278-4364-a245-adfc5a4f9f7c" />

<img width="1400" height="1200" alt="image" src="https://github.com/user-attachments/assets/1a1a74f8-fd0b-43ff-b659-cf7dc00127a4" />


*Time series decomposition showing trend, seasonal, and residual components*

### Advanced Machine Learning Forecasting

- **Random Forest Regressor**: Ensemble learning with 200 trees
- **Gradient Boosting**: Sequential error correction approach
- **LSTM Neural Networks**: Deep learning for temporal sequences
- **Performance Metrics**: MAE, RMSE, R², MAPE evaluation
- **6-Month Forecasts**: Extended predictions with confidence intervals

<img width="1828" height="449" alt="newplot (3)" src="https://github.com/user-attachments/assets/f9cca254-2b54-43b2-bc6a-e8b99d9202f9" />

*Comparison of Random Forest, Gradient Boosting, and LSTM performance*

### Interactive Visualizations

- Plotly-based interactive charts
- Normalized comparison (base year 2015 = 100)
- Yearly performance bar charts
- Monthly seasonality patterns
- Time series decomposition plots
- Actual vs predicted comparisons
- LSTM forecast visualizations with confidence bands


## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab (for notebook)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/kes-exchange-rate-analysis.git
cd kes-exchange-rate-analysis
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.11.0
statsmodels>=0.14.0
scipy>=1.9.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
streamlit>=1.25.0
jupyter>=1.0.0
```

## Usage

### Accessing the Live Dashboard

Simply visit: [https://kes-usd-forex-timeseries-rbgqw6jkakwstauwsptzhu.streamlit.app/](https://kes-usd-forex-timeseries-rbgqw6jkakwstauwsptzhu.streamlit.app/)

No installation required! The dashboard is fully hosted and accessible from any device.

### Running the Jupyter Notebook

1. **Start Jupyter:**
```bash
jupyter notebook kesforextimeseries.ipynb
```

2. **Run all cells sequentially** to perform complete analysis

3. **Customize analysis** by modifying parameters in configuration cells

### Running the Dashboard Locally

1. **Launch Streamlit app:**
```bash
streamlit run app.py
```

2. **Access in browser:** Opens automatically at `http://localhost:8501`

3. **Navigate pages:** Use sidebar menu to explore different analysis sections

### Quick Start Example

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
usd_data = pd.read_csv('USD_KES Historical Data.csv')
usd_data['Date'] = pd.to_datetime(usd_data['Date'])
usd_data = usd_data.sort_values('Date')

# Fit SARIMA model
model = SARIMAX(usd_data['Price'], order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()

# Generate forecast
forecast = results.get_forecast(steps=180)
print(forecast.predicted_mean)
```

## Analysis Components

### 1. Data Preprocessing

- Date parsing and sorting
- Missing value handling (forward-fill, backward-fill)
- Returns calculation (daily percentage changes)
- Feature engineering for ML models

### 2. Stationarity Analysis

**Price Level Tests:**
- ADF Test: All p-values >0.60 (non-stationary)
- KPSS Test: All p-values <0.01 (non-stationary)

**After First-Order Differencing:**
- ADF Test: All p-values <0.001 (stationary)
- KPSS Test: All p-values >0.05 (stationary)
- Validates SARIMA(p,1,q) differencing parameter

### 3. Decomposition

**Components Extracted:**
- Trend: Long-term directional movement (2-2.5% annual depreciation)
- Seasonal: 12-month recurring patterns
- Residual: Random noise after removing predictable patterns

**Signal-to-Noise Ratios:**
- USD/KES: 4.56 (Strong signal, most predictable)
- GBP/KES: 2.91 (Moderate signal)
- EUR/KES: 2.52 (Weak signal, least predictable)

### 4. Correlation Analysis

**Price Correlations:** Very high (>0.90) - limited diversification
**Returns Correlations:** Moderate (0.30-0.68) - some tactical opportunities

<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/88e8012d-c559-4bef-821c-44b122cb2a16" />
*Correlation analysis revealing relationship patterns between currency pairs*

### 5. Trend Analysis

- Linear trend: Quantifies long-term depreciation
- Polynomial trend: Captures acceleration/deceleration
- Trend strength: 52.8-68.9% (USD strongest)

### 6. Seasonality

**Depreciation Months (High FX Demand):**
- January: +1-4% (year-end demand)
- April-May: +2-5% (tax payments, school fees)
- December: +2-6% (holiday imports)

**Appreciation Months (High FX Supply):**
- February-March: -3 to -9% (diaspora remittances)
- September: -2 to -3% (export earnings)

## Machine Learning Models

### Model Architectures

**Random Forest:**
- 200 decision trees
- Features: 30 lags + rolling statistics (7d, 30d) + EMAs
- Max depth: 20
- Min samples split: 5

**Gradient Boosting:**
- 200 estimators
- Learning rate: 0.1
- Max depth: 5
- Subsample ratio: 0.8

**LSTM Neural Network:**
- Architecture: Input(60) → LSTM(50) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(25) → Dense(1)
- Optimizer: Adam
- Loss: MSE
- Early stopping: Patience 10


### Model Performance

| Currency | Model | MAE | RMSE | R² | MAPE (%) |
|----------|-------|-----|------|-----|----------|
| USD/KES | Random Forest | 4.18 | 5.02 | 0.73 | 3.04 |
| USD/KES | Gradient Boosting | 4.70 | 5.31 | 0.70 | 3.45 |
| USD/KES | **LSTM** | **1.37** | **2.17** | **0.95** | **1.01** |
| EUR/KES | Random Forest | 4.44 | 5.99 | 0.71 | 2.92 |
| EUR/KES | Gradient Boosting | 4.90 | 6.45 | 0.66 | 3.24 |
| EUR/KES | **LSTM** | **1.00** | **1.49** | **0.98** | **0.68** |
| GBP/KES | Random Forest | 4.97 | 6.87 | 0.64 | 2.78 |
| GBP/KES | Gradient Boosting | 5.15 | 7.09 | 0.62 | 2.89 |
| GBP/KES | **LSTM** | **0.83** | **1.20** | **0.99** | **0.47** |

**Key Achievement:** LSTM achieved 60-80% improvement over traditional SARIMA methods.

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/c7b84a55-58bf-4526-80b9-b6c08ce722a2" />

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/805c62d7-fcb6-4cb0-9fdd-c39a36c9db48" />

*Comprehensive comparison of model performance across all metrics*

## Results

### Historical Performance (2015-2025)

| Currency Pair | Total Depreciation | Annualized Rate | Volatility | Trend Strength |
|---------------|-------------------|-----------------|------------|----------------|
| USD/KES | +22.0% | +2.06%/year | 0.21% | 68.9% (Strong) |
| EUR/KES | +28.0% | +2.49%/year | 0.51% | 64.0% (Moderate) |
| GBP/KES | +10.0% | +0.84%/year | 0.61% | 52.8% (Weak) |

<img width="1579" height="531" alt="image" src="https://github.com/user-attachments/assets/bcb8741a-ecc6-43ed-ae0b-9658facbee8e" />
*10-year historical performance showing three distinct phases*

### Three Historical Phases

1. **Stability (2015-2019):** Gradual depreciation, low volatility
2. **Crisis (2020-2023):** COVID-19 impact, 2023 peak (27-35% depreciation)
3. **Recovery (2024-2025):** Sharp correction (18-23% appreciation), stabilization

### LSTM 6-Month Forecasts

| Currency | Current Rate | 6M Forecast | Expected Change |
|----------|--------------|-------------|-----------------|
| USD/KES | 129.20 | 149.61 | +15.80% |
| EUR/KES | 151.51 | 168.96 | +11.52% |
| GBP/KES | 173.48 | 197.52 | +13.86% |

**Interpretation:** All forecasts project continued depreciation, with USD showing highest expected movement.

### Strategic Recommendations

**For Corporate Treasurers:**
- Hedge 60-80% of 6-month exposure
- Prioritize USD hedging (highest forecast depreciation)
- Use LSTM forecasts for decision-making

**For Importers:**
- Execute forward contracts immediately
- Negotiate USD-denominated contracts
- Front-load imports in Oct-Nov (before seasonal peak)

**For Exporters:**
- Delay EUR/GBP conversions until favorable periods
- Accelerate USD collections
- Diversify currency mix: 40% USD, 30% EUR, 20% GBP

**For Investors:**
- Hedge 70-80% of KES-denominated assets
- Favor USD-denominated securities
- Reduce KES exposure when volatility >0.8%



## Data Source

All exchange rate data sourced from the **Central Bank of Kenya (CBK)** official website: [www.centralbank.go.ke](http://www.centralbank.go.ke)

**Data Specifications:**
- **Frequency:** Daily indicative rates
- **Period:** September 29, 2015 - September 29, 2025
- **Observations:** ~2,610 per currency
- **Quality:** Official CBK rates based on market transactions
- **Variables:** Date, Price (Close), Open, High, Low, Volume, Change %

## Key Findings

1. **Long-term Trend:** KES depreciates 2-2.5% annually against major currencies
2. **2023 Crisis:** Exceptional year with 27-35% depreciation (not the new normal)
3. **Current Valuation:** Rates fairly valued near long-term trend lines
4. **Volatility Status:** Normalized to historical levels after crisis period
5. **USD Advantage:** Most stable (0.21% volatility), strongest trend (68.9%)
6. **Seasonal Patterns:** Exploitable for 2-4% annual savings through timing
7. **ML Superiority:** LSTM achieved 95-99% accuracy vs 40-65% for SARIMA
8. **Forecast Outlook:** 11-16% depreciation expected over next 6 months


## Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Additional Features:**
   - Incorporate macroeconomic indicators (interest rates, inflation)
   - Add sentiment analysis from news sources
   - Implement real-time data feeds
   - Develop high-frequency (intraday) analysis

2. **Model Enhancements:**
   - Attention mechanisms for LSTM
   - Transformer architectures
   - Ensemble methods combining multiple models
   - Bayesian approaches for uncertainty quantification

3. **Visualization Improvements:**
   - Additional interactive dashboards
   - Real-time monitoring alerts
   - Custom report generation
   - Mobile app development

4. **Documentation:**
   - Tutorial videos
   - Academic paper write-up
   - API documentation
   - Use case examples

### How to Contribute

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Guidelines

- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation
- Ensure reproducibility with random seeds



## Contact


**Email**: aimemuganga07@gmail.com


**Project Link**: https://github.com/DevAime/kes-usd-forex-timeseries.git



**Past performance does not guarantee future results.** Users should:
- Conduct independent due diligence
- Consult qualified financial advisors
- Consider their specific circumstances
- Understand the risks of foreign exchange trading

The authors assume no liability for decisions made based on this analysis.

---



⭐ **Star this repository** if you found it helpful!
