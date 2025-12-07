"""
KES Exchange Rate Analysis Dashboard
Streamlit Application for Interactive Time Series Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="KES Exchange Rate Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
    }
    .highlight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    summary_stats = {
        'USD': {'current': 129.20, 'start': 105.90, 'change': 22.0, 'volatility': 0.21, 
                'trend': 2.06, 'snr': 4.56, 'mae_sarima': 8.96, 'mae_lstm': 1.37, 
                'r2_lstm': 0.9527, 'mape_lstm': 1.01, 'forecast_6m': 149.61, 'forecast_change': 15.80},
        'EUR': {'current': 151.51, 'start': 118.40, 'change': 28.0, 'volatility': 0.51, 
                'trend': 2.49, 'snr': 2.52, 'mae_sarima': 7.49, 'mae_lstm': 1.00, 
                'r2_lstm': 0.9824, 'mape_lstm': 0.68, 'forecast_6m': 168.96, 'forecast_change': 11.52},
        'GBP': {'current': 173.48, 'start': 157.70, 'change': 10.0, 'volatility': 0.61, 
                'trend': 0.84, 'snr': 2.91, 'mae_sarima': 14.02, 'mae_lstm': 0.83, 
                'r2_lstm': 0.9890, 'mape_lstm': 0.47, 'forecast_6m': 197.52, 'forecast_change': 13.86}
    }
    
    years = list(range(2015, 2026))
    historical_data = pd.DataFrame({
        'Year': years,
        'USD': [100, 100.2, 100.9, 99.6, 99.1, 106.8, 110.7, 120.7, 148, 122, 122],
        'EUR': [100, 97, 112, 106, 103, 121, 116, 120, 151, 116, 129],
        'GBP': [100, 84, 95, 88, 92, 102, 105, 103, 138, 112, 110]
    })
    
    yearly_performance = pd.DataFrame({
        'Year': ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025'],
        'USD': [-2.9, 0.2, 0.7, -1.3, -0.5, 7.7, 3.6, 9.1, 27.2, -17.6, -0.1],
        'EUR': [-6.2, -3.0, 15.5, -5.8, -2.7, 17.4, -4.1, 3.4, 31.4, -22.8, 13.6],
        'GBP': [-5.5, -16.2, 10.9, -6.7, 3.5, 11.2, 2.5, -2.1, 34.5, -19.0, 7.1]
    })
    
    lstm_forecast_data = pd.DataFrame({
        'Month': ['Current', '1M', '2M', '3M', '4M', '5M', '6M'],
        'USD': [129.20, 133.18, 136.91, 140.58, 144.02, 147.06, 149.61],
        'EUR': [151.51, 154.62, 157.83, 160.93, 163.86, 166.55, 168.96],
        'GBP': [173.48, 177.70, 181.52, 185.47, 189.50, 193.54, 197.52]
    })
    
    seasonality_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'USD': [0.012, -0.044, -0.008, 0.026, -0.001, 0.012, 0.026, 0.013, 0.016, 0.010, 0.023, 0.008],
        'EUR': [0.024, -0.085, 0.028, 0.042, 0.001, 0.029, 0.039, 0.019, -0.026, -0.025, 0.020, 0.055],
        'GBP': [0.039, -0.086, 0.010, 0.048, -0.022, -0.031, 0.030, -0.006, -0.021, 0.007, 0.066, 0.021]
    })
    
    ml_comparison = pd.DataFrame({
        'Currency': ['USD/KES', 'USD/KES', 'USD/KES', 
                     'EUR/KES', 'EUR/KES', 'EUR/KES',
                     'GBP/KES', 'GBP/KES', 'GBP/KES'],
        'Model': ['Random Forest', 'Gradient Boosting', 'LSTM'] * 3,
        'MAE': [4.18, 4.70, 1.37, 4.44, 4.90, 1.00, 4.97, 5.15, 0.83],
        'R²': [0.73, 0.70, 0.95, 0.71, 0.66, 0.98, 0.64, 0.62, 0.99],
        'MAPE': [3.04, 3.45, 1.01, 2.92, 3.24, 0.68, 2.78, 2.89, 0.47]
    })
    
    return summary_stats, historical_data, yearly_performance, lstm_forecast_data, seasonality_data, ml_comparison

summary_stats, historical_data, yearly_performance, lstm_forecast_data, seasonality_data, ml_comparison = load_data()

# Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select Section:",
        ["Overview", "Historical Analysis", "Forecasts", "Machine Learning", "Insights"]
    )
    
    st.markdown("---")
    st.subheader("Data Source")
    st.info("Central Bank of Kenya\nwww.centralbank.go.ke")
    
    st.subheader("Period")
    st.success("Sep 2015 - Sep 2025\n(10 Years)")
    
    st.markdown("---")
    st.subheader("Quick Stats")
    for currency in ['USD', 'EUR', 'GBP']:
        change = summary_stats[currency]['change']
        st.metric(
            f"{currency}/KES",
            f"{summary_stats[currency]['current']:.2f}",
            f"{change:+.1f}%",
            delta_color="inverse"
        )

# Main content
st.title("KES Exchange Rate Analysis (2015-2025)")
st.markdown("**Comprehensive Time Series & Machine Learning Forecasting**")
st.markdown("---")

# OVERVIEW PAGE
if page == "Overview":
    st.header("Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### USD/KES")
        st.metric("Current Rate", f"KES {summary_stats['USD']['current']:.2f}")
        st.metric("10-Year Change", f"{summary_stats['USD']['change']:+.1f}%", delta_color="inverse")
        st.metric("Volatility", f"{summary_stats['USD']['volatility']:.2f}%")
        st.metric("Annual Trend", f"{summary_stats['USD']['trend']:+.2f}%/yr")
        st.success("**Most Stable**")
    
    with col2:
        st.markdown("### EUR/KES")
        st.metric("Current Rate", f"KES {summary_stats['EUR']['current']:.2f}")
        st.metric("10-Year Change", f"{summary_stats['EUR']['change']:+.1f}%", delta_color="inverse")
        st.metric("Volatility", f"{summary_stats['EUR']['volatility']:.2f}%")
        st.metric("Annual Trend", f"{summary_stats['EUR']['trend']:+.2f}%/yr")
        st.warning("**Moderate Risk**")
    
    with col3:
        st.markdown("### GBP/KES")
        st.metric("Current Rate", f"KES {summary_stats['GBP']['current']:.2f}")
        st.metric("10-Year Change", f"{summary_stats['GBP']['change']:+.1f}%", delta_color="inverse")
        st.metric("Volatility", f"{summary_stats['GBP']['volatility']:.2f}%")
        st.metric("Annual Trend", f"{summary_stats['GBP']['trend']:+.2f}%/yr")
        st.error("**Highest Risk**")
    
    st.markdown("---")
    
    st.info("""
    **About This Analysis:**
    
    Comprehensive 10-year analysis combining traditional time series methods (SARIMA) with 
    machine learning (Random Forest, Gradient Boosting, LSTM). Data from Central Bank of Kenya. 
    Analysis includes stationarity testing, decomposition, correlation analysis, and forecasting.
    """)
    
    st.subheader("Normalized Exchange Rate Comparison (2015 = 100)")
    st.caption("Values above 100 indicate KES depreciation")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data['Year'], y=historical_data['USD'],
        mode='lines+markers', name='USD/KES',
        line=dict(color='#FF6B6B', width=3), marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_data['Year'], y=historical_data['EUR'],
        mode='lines+markers', name='EUR/KES',
        line=dict(color='#4ECDC4', width=3), marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=historical_data['Year'], y=historical_data['GBP'],
        mode='lines+markers', name='GBP/KES',
        line=dict(color='#45B7D1', width=3), marker=dict(size=8)
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                  annotation_text="Baseline (2015)")
    
    fig.update_layout(height=500, hovermode='x unified', 
                      xaxis_title="Year", yaxis_title="Index (2015 = 100)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Historical Trends:**
        - All pairs show KES depreciation over 10 years
        - 2023 crisis: 27-35% peak depreciation
        - 2024 recovery: 18-23% correction
        - Current: Stabilized near trend lines
        """)
    
    with col2:
        st.markdown("""
        **Risk Profile:**
        - USD/KES: Lowest volatility (0.21%)
        - EUR/KES: Moderate volatility (0.51%)
        - GBP/KES: Highest volatility (0.61%)
        - Brexit and UK politics increase GBP risk
        """)

# HISTORICAL ANALYSIS PAGE
elif page == "Historical Analysis":
    st.header("Historical Performance")
    
    st.subheader("Yearly Exchange Rate Performance")
    st.caption("Positive = KES depreciation; Negative = KES appreciation")
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=yearly_performance['Year'], y=yearly_performance['USD'],
                         name='USD/KES', marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(x=yearly_performance['Year'], y=yearly_performance['EUR'],
                         name='EUR/KES', marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(x=yearly_performance['Year'], y=yearly_performance['GBP'],
                         name='GBP/KES', marker_color='#45B7D1'))
    
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=500, barmode='group', hovermode='x unified',
                      xaxis_title="Year", yaxis_title="Yearly Change (%)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Key Events:**
    - 2016: Brexit caused GBP/KES drop (-16.2%)
    - 2020: COVID-19 impact (7-17% depreciation)
    - 2023: Crisis year (27-35% depreciation)
    - 2024: Recovery year (-18 to -23% appreciation)
    """)
    
    st.markdown("---")
    
    st.subheader("Monthly Seasonality Pattern")
    st.caption("Average daily returns by month")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=seasonality_data['Month'], y=seasonality_data['USD']*100,
                             mode='lines+markers', name='USD/KES',
                             line=dict(color='#FF6B6B', width=3), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=seasonality_data['Month'], y=seasonality_data['EUR']*100,
                             mode='lines+markers', name='EUR/KES',
                             line=dict(color='#4ECDC4', width=3), marker=dict(size=10)))
    fig.add_trace(go.Scatter(x=seasonality_data['Month'], y=seasonality_data['GBP']*100,
                             mode='lines+markers', name='GBP/KES',
                             line=dict(color='#45B7D1', width=3), marker=dict(size=10)))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450, hovermode='x unified',
                      xaxis_title="Month", yaxis_title="Average Return (%)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Depreciation Months:**
        - January: Year-end demand
        - April-May: Tax, school fees
        - December: Holiday imports
        
        **Strategy:** Hedge in these months
        """)
    
    with col2:
        st.info("""
        **Appreciation Months:**
        - February-March: Remittances
        - September: Export earnings
        
        **Strategy:** Convert forex in these months
        """)

# FORECASTS PAGE
elif page == "Forecasts":
    st.header("LSTM 6-Month Forecasts")
    
    st.info("""
    **LSTM Neural Network Forecasting:**
    - Best performing model (R² 0.95-0.99)
    - Trained on 60-day sequences
    - 2-layer architecture with dropout
    - Validated on 20% test data
    """)
    
    selected_currency = st.selectbox("Select Currency:", ["USD/KES", "EUR/KES", "GBP/KES"])
    currency_code = selected_currency.split('/')[0]
    
    st.subheader(f"{selected_currency} Forecast")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lstm_forecast_data['Month'],
        y=lstm_forecast_data[currency_code],
        mode='lines+markers',
        name='LSTM Forecast',
        line=dict(color='#FF6B6B' if currency_code=='USD' else '#4ECDC4' if currency_code=='EUR' else '#45B7D1', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(255,107,107,0.1)' if currency_code=='USD' else 'rgba(78,205,196,0.1)' if currency_code=='EUR' else 'rgba(69,183,209,0.1)'
    ))
    
    fig.update_layout(height=450, hovermode='x',
                      xaxis_title="Forecast Horizon", 
                      yaxis_title=f"Exchange Rate (KES per {currency_code})")
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_rate = summary_stats[currency_code]['current']
    forecast_end = summary_stats[currency_code]['forecast_6m']
    change = summary_stats[currency_code]['forecast_change']
    mae = summary_stats[currency_code]['mae_lstm']
    
    with col1:
        st.metric("Current Rate", f"{current_rate:.2f}")
    with col2:
        st.metric("6M Forecast", f"{forecast_end:.2f}")
    with col3:
        st.metric("Expected Change", f"{change:+.2f}%", delta_color="inverse")
    with col4:
        st.metric("Model MAE", f"{mae:.2f}")
    
    st.markdown("---")
    
    st.subheader("All Currencies - 6-Month Outlook")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### USD/KES")
        st.metric("Current", f"{summary_stats['USD']['current']:.2f}")
        st.metric("6M Forecast", f"{summary_stats['USD']['forecast_6m']:.2f}")
        st.metric("Change", f"{summary_stats['USD']['forecast_change']:+.2f}%", delta_color="inverse")
        st.caption(f"R²: {summary_stats['USD']['r2_lstm']:.4f}")
    
    with col2:
        st.markdown("#### EUR/KES")
        st.metric("Current", f"{summary_stats['EUR']['current']:.2f}")
        st.metric("6M Forecast", f"{summary_stats['EUR']['forecast_6m']:.2f}")
        st.metric("Change", f"{summary_stats['EUR']['forecast_change']:+.2f}%", delta_color="inverse")
        st.caption(f"R²: {summary_stats['EUR']['r2_lstm']:.4f}")
    
    with col3:
        st.markdown("#### GBP/KES")
        st.metric("Current", f"{summary_stats['GBP']['current']:.2f}")
        st.metric("6M Forecast", f"{summary_stats['GBP']['forecast_6m']:.2f}")
        st.metric("Change", f"{summary_stats['GBP']['forecast_change']:+.2f}%", delta_color="inverse")
        st.caption(f"R²: {summary_stats['GBP']['r2_lstm']:.4f}")
    
    st.warning("""
    **Forecast Limitations:**
    - Assumes historical patterns continue
    - Cannot predict policy changes or black swan events
    - Accuracy decreases with forecast horizon
    - Use as guidance, not certainty
    """)

# MACHINE LEARNING PAGE
elif page == "Machine Learning":
    st.header("Machine Learning Model Comparison")
    
    st.markdown("""
    **Models Evaluated:**
    - **Random Forest:** Ensemble of 200 decision trees
    - **Gradient Boosting:** Sequential error correction
    - **LSTM Neural Network:** Deep learning for sequences
    """)
    
    st.subheader("Model Performance Comparison")
    
    # MAE Comparison
    fig_mae = px.bar(ml_comparison, x='Currency', y='MAE', color='Model',
                     barmode='group', title='Mean Absolute Error (Lower = Better)',
                     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    fig_mae.update_layout(height=400)
    st.plotly_chart(fig_mae, use_container_width=True)
    
    # R² Comparison
    fig_r2 = px.bar(ml_comparison, x='Currency', y='R²', color='Model',
                    barmode='group', title='R² Score (Higher = Better)',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    fig_r2.update_layout(height=400)
    st.plotly_chart(fig_r2, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("LSTM Superiority")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### USD/KES")
        st.metric("LSTM R²", "0.9527")
        st.metric("LSTM MAE", "1.37 KES")
        st.metric("LSTM MAPE", "1.01%")
        st.success("95% variance explained")
    
    with col2:
        st.markdown("#### EUR/KES")
        st.metric("LSTM R²", "0.9824")
        st.metric("LSTM MAE", "1.00 KES")
        st.metric("LSTM MAPE", "0.68%")
        st.success("98% variance explained")
    
    with col3:
        st.markdown("#### GBP/KES")
        st.metric("LSTM R²", "0.9890")
        st.metric("LSTM MAE", "0.83 KES")
        st.metric("LSTM MAPE", "0.47%")
        st.success("99% variance explained")
    
    st.info("""
    **Key Findings:**
    - LSTM achieved R² scores of 0.95-0.99 (near-perfect predictions)
    - 60-80% improvement over SARIMA baseline
    - Prediction errors typically <1 KES
    - GBP/KES best predicted despite highest volatility
    """)
    
    st.markdown("---")
    
    st.subheader("Model Comparison Table")
    
    st.dataframe(ml_comparison.style.highlight_min(subset=['MAE', 'MAPE'], color='lightgreen')
                                    .highlight_max(subset=['R²'], color='lightgreen'),
                 use_container_width=True)

# INSIGHTS PAGE
else:
    st.header("Key Insights & Recommendations")
    
    st.subheader("Signal-to-Noise Ratio")
    st.caption("Measures trend strength (higher = more predictable)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("USD/KES", f"{summary_stats['USD']['snr']:.2f}")
        st.progress(summary_stats['USD']['snr'] / 5.0)
        st.success("Strong Signal")
    
    with col2:
        st.metric("EUR/KES", f"{summary_stats['EUR']['snr']:.2f}")
        st.progress(summary_stats['EUR']['snr'] / 5.0)
        st.warning("Weak Signal")
    
    with col3:
        st.metric("GBP/KES", f"{summary_stats['GBP']['snr']:.2f}")
        st.progress(summary_stats['GBP']['snr'] / 5.0)
        st.info("Moderate Signal")
    
    st.markdown("---")
    
    st.subheader("Strategic Recommendations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Treasurers", "Importers", "Exporters", "Investors"])
    
    with tab1:
        st.markdown("""
        ### For Corporate Treasurers
        
        **Immediate Actions:**
        - Hedge 60-80% of 6-month exposure
        - Prioritize USD (highest forecast depreciation +15.8%)
        - Use LSTM forecasts for decision-making
        
        **Strategy:**
        - Implement layered hedging (staggered maturities)
        - Monitor 30-day volatility
        - Rebalance monthly
        """)
    
    with tab2:
        st.markdown("""
        ### For Importers
        
        **Cost Management:**
        - Execute forward contracts immediately
        - Negotiate USD-denominated contracts
        - Front-load imports before further depreciation
        
        **Pricing:**
        - Add 2-3% FX escalation clauses
        - Review pricing quarterly
        - Import in Oct-Nov (before Jan peak)
        """)
    
    with tab3:
        st.markdown("""
        ### For Exporters
        
        **Revenue Optimization:**
        - Favorable environment for KES earnings
        - Delay forex conversions (benefit from depreciation)
        - Diversify currency mix (40% USD, 30% EUR, 20% GBP)
        
        **Timing:**
        - Accelerate USD collections
        - Hold EUR/GBP receivables
        """)
    
    with tab4:
        st.markdown("""
        ### For Investors
        
        **Exposure Management:**
        - Hedge 70-80% of KES assets
        - Favor USD-denominated securities
        - Monitor volatility thresholds
        
        **Strategy:**
        - Reduce KES when volatility >0.8%
        - Use LSTM for tactical allocation
        """)
    
    st.markdown("---")
    
    st.subheader("Key Takeaways")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Positive Findings:**
        - Long-term trend predictable (2-2.5% annual)
        - LSTM forecasts highly accurate (R² >0.95)
        - Seasonal patterns exploitable (2-4% savings)
        - USD/KES most stable and reliable
        - Volatility normalized post-crisis
        """)
    
    with col2:
        st.warning("""
        **Risk Factors:**
        - All forecasts project 11-16% depreciation
        - GBP highest volatility (0.61%)
        - Limited cross-currency diversification
        - Cannot predict black swan events
        - Policy changes can override forecasts
        """)

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** Central Bank of Kenya (www.centralbank.go.ke)  
**Analysis Period:** September 29, 2015 - September 29, 2025  
**Methods:** SARIMA, Random Forest, Gradient Boosting, LSTM Neural Networks  
**Disclaimer:** For educational purposes only. Not financial advice. Consult professionals before decisions.
""")
st.caption("2025 KES Exchange Rate Analysis | Comprehensive Time Series & ML Forecasting")