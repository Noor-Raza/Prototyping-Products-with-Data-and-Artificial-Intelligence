import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import cohere
from custom_theme import apply_custom_theme, display_header, display_footer, enhance_sidebar
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------- Session State Management ---------------- #
def initialize_session_state():
    """Initialize session state variables for persistent storage across reruns"""
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if "active_stock" not in st.session_state:
        st.session_state.active_stock = None

def add_to_watchlist(ticker):
    """Add a stock to the watchlist"""
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        return True
    return False

def remove_from_watchlist(ticker):
    """Remove a stock from the watchlist"""
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)
        return True
    return False

def set_active_stock(ticker):
    """Set the currently active stock"""
    st.session_state.active_stock = ticker

def get_active_stock():
    """Get the currently active stock"""
    return st.session_state.active_stock

# Initialize session state
initialize_session_state()

# ---------------- Page Configuration ---------------- #
st.set_page_config(page_title="Stock Forecasting", layout="wide")
apply_custom_theme()
enhance_sidebar()

# ---------------- App Title ---------------- #
display_header("🔮 Stock Forecasting Insights", 
              "Predict future stock price movements with advanced modeling")

# ---------------- Sidebar ---------------- #
st.sidebar.header("📊 Forecast Settings")

# Get current active stock
current_stock = get_active_stock()

# Stock selection section
st.sidebar.subheader("Select Stock")

# Standard stock selection
POPULAR_STOCKS = [
    "AAPL - Apple Inc.", "MSFT - Microsoft Corporation", "GOOGL - Alphabet Inc.",
    "AMZN - Amazon.com Inc.", "NVDA - NVIDIA Corporation", "META - Meta Platforms Inc.",
    "TSLA - Tesla Inc.", "JPM - JPMorgan Chase & Co.", "JNJ - Johnson & Johnson",
    "V - Visa Inc.", "Other (Enter Ticker)"
]

stock_choice = st.sidebar.selectbox("Select Stock", options=POPULAR_STOCKS, index=0 if current_stock is None else next((i for i, s in enumerate(POPULAR_STOCKS) if s.startswith(current_stock)), 10))
ticker = st.sidebar.text_input("Enter Custom Ticker", value=current_stock if stock_choice == "Other (Enter Ticker)" else "") if stock_choice == "Other (Enter Ticker)" else stock_choice.split()[0]

# Button to apply selection
if st.sidebar.button("Apply Selection"):
    set_active_stock(ticker)
    st.experimental_rerun()

# Initialize Cohere client with API key from environment variables
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Check if API key is available
if not COHERE_API_KEY:
    st.warning("Cohere API key not found. Some AI-powered features will be limited.")
    co = None
else:
    co = cohere.Client(COHERE_API_KEY)

# Display additional stock information
try:
    info = yf.Ticker(ticker).info
    if 'longName' in info and 'sector' in info:
        st.sidebar.markdown(f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:5px; margin-top:15px;">
            <h3 style="margin:0;">{info.get('longName')}</h3>
            <p>Sector: {info.get('sector')}<br>
            Industry: {info.get('industry')}</p>
        </div>
        """, unsafe_allow_html=True)
except:
    pass

# Watchlist section
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Watchlist")

# Add/Remove from watchlist
if ticker:
    if ticker in st.session_state.watchlist:
        if st.sidebar.button(f"Remove {ticker} from Watchlist"):
            if remove_from_watchlist(ticker):
                st.sidebar.success(f"Removed {ticker} from watchlist")
    else:
        if st.sidebar.button(f"Add {ticker} to Watchlist"):
            if add_to_watchlist(ticker):
                st.sidebar.success(f"Added {ticker} to watchlist")

# Display watchlist
if st.session_state.watchlist:
    st.sidebar.markdown("Your stocks:")
    
    # Create a 2-column layout for watchlist items
    cols = st.sidebar.columns(2)
    
    for i, watched_ticker in enumerate(st.session_state.watchlist):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(f"{watched_ticker}", key=f"watchlist_{watched_ticker}"):
                set_active_stock(watched_ticker)
                st.experimental_rerun()
else:
    st.sidebar.info("Your watchlist is empty. Add stocks to track them easily.")

st.sidebar.markdown("---")

# Rest of the sidebar settings
forecast_days = st.sidebar.slider("Days to Forecast", min_value=3, max_value=30, value=7)

end_date = date.today() - timedelta(days=1)
start_date = st.sidebar.date_input("Start Date", value=end_date - timedelta(days=365), max_value=end_date)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    include_weekends = st.checkbox("Include weekends in forecast", value=False)
    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, 0.001, 
                                      help="Controls flexibility of the trend (higher = more flexible)")
    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 1.0, 0.01,
                                      help="Controls strength of seasonality (higher = stronger seasonal patterns)")

@st.cache_data
def get_data(ticker, start):
    df = yf.download(ticker, start=start)
    return df.reset_index()

# Add confidence interval explanation
def add_confidence_explanation():
    """Add an expandable explanation of confidence intervals"""
    with st.expander("ℹ️ Understanding Confidence Intervals & Forecast Metrics"):
        st.markdown("""
        ### Confidence Intervals
        The shaded area in the forecast represents the 80% confidence interval:
        
        - **Upper Bound**: There's a 10% chance the price will be higher than this line
        - **Lower Bound**: There's a 10% chance the price will be lower than this line
        - **Forecast Line**: The most likely price path based on historical patterns
        
        ### Forecast Evaluation Metrics
        - **RMSE** (Root Mean Squared Error): Measures the average magnitude of forecast errors (lower is better)
        - **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values
        - **MAPE** (Mean Absolute Percentage Error): Average percentage difference between predicted and actual values
        
        *Note: These metrics are calculated using a test period to evaluate forecast accuracy.*
        """)

# Add scenario analysis section
def add_scenario_analysis(ticker, forecast, forecast_days):
    """Add a scenario analysis section"""
    st.subheader("📊 Scenario Analysis")
    with st.expander("What is Scenario Analysis?"):
        st.markdown("""
        Scenario Analysis shows three potential price outcomes:
        - **Bullish**: Upper bound of the forecast (optimistic scenario)
        - **Baseline**: Central forecast prediction (most likely scenario)
        - **Bearish**: Lower bound of the forecast (pessimistic scenario)
        
        Percentages indicate expected price changes from current value.
        """)
    
    # Create scenarios
    last_price = forecast["yhat"].iloc[-forecast_days-1]
    baseline = forecast["yhat"].iloc[-1]
    bullish = forecast["yhat_upper"].iloc[-1]
    bearish = forecast["yhat_lower"].iloc[-1]
    
    # Calculate percentage changes
    baseline_pct = ((baseline - last_price) / last_price) * 100
    bullish_pct = ((bullish - last_price) / last_price) * 100
    bearish_pct = ((bearish - last_price) / last_price) * 100
    
    # Create columns for scenarios
    bull_col, base_col, bear_col = st.columns(3)
    
    with bull_col:
        st.markdown(f"""
        <div style="background-color:#d4edda; border-radius:10px; padding:15px; text-align:center;">
            <h3 style="color:#155724;">Bullish</h3>
            <h2 style="color:#155724;">${bullish:.2f}</h2>
            <p style="font-size:18px;">+{bullish_pct:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with base_col:
        st.markdown(f"""
        <div style="background-color:#e2e3e5; border-radius:10px; padding:15px; text-align:center;">
            <h3 style="color:#383d41;">Baseline</h3>
            <h2 style="color:#383d41;">${baseline:.2f}</h2>
            <p style="font-size:18px;">{baseline_pct:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with bear_col:
        st.markdown(f"""
        <div style="background-color:#f8d7da; border-radius:10px; padding:15px; text-align:center;">
            <h3 style="color:#721c24;">Bearish</h3>
            <h2 style="color:#721c24;">${bearish:.2f}</h2>
            <p style="font-size:18px;">{bearish_pct:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# Dynamic forecast insight based on actual forecast results
def generate_llm_insight_via_api(df_prophet, forecast, forecast_days, ticker):
    recent_price = df_prophet["y"].iloc[-1]
    predicted_price = forecast["yhat"].iloc[-1]
    price_change = (predicted_price - recent_price) / recent_price * 100

    # Construct the prompt for Cohere's API to generate the insight
    prompt = f"""
    Given the forecast for {ticker} in the next {forecast_days} days, 
    the most recent closing price is ${recent_price:.2f} and the predicted closing price is ${predicted_price:.2f}.
    This is a {price_change:.2f}% change.
    
    Provide a detailed and insightful analysis of this stock's forecast based on the given information.
    The analysis should include possible reasons for the trend, any relevant market factors, and investor sentiment.
    """

    # Request insight from Cohere API
    response = co.generate(
        model='command-light',  # Cohere's best model
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )

    # Return the generated insight
    return response.generations[0].text.strip()

def generate_comprehensive_insights(df_prophet, forecast, forecast_days, ticker, metrics=None):
    """
    Generate streamlined yet comprehensive investment insights that incorporate all available 
    forecasting data on the page.
    
    Parameters:
    -----------
    df_prophet : DataFrame
        Historical stock data used for prediction
    forecast : DataFrame
        Prophet model forecast results
    forecast_days : int
        Number of days forecasted
    ticker : str
        Stock ticker symbol
    metrics : dict, optional
        Dictionary containing model evaluation metrics (RMSE, MAE, MAPE)
    
    Returns:
    --------
    str
        Comprehensive investment insights text
    """
    # Extract key price information
    recent_price = df_prophet["y"].iloc[-1]
    predicted_price = forecast["yhat"].iloc[-1]
    price_change = (predicted_price - recent_price) / recent_price * 100
    
    # Extract scenario analysis data
    bullish_price = forecast["yhat_upper"].iloc[-1]
    bearish_price = forecast["yhat_lower"].iloc[-1]
    
    bullish_change = ((bullish_price - recent_price) / recent_price) * 100
    bearish_change = ((bearish_price - recent_price) / recent_price) * 100
    
    # Extract seasonality insights
    try:
        # Weekly seasonality
        weekly_seasonality = forecast[["ds", "weekly"]].copy()
        weekly_seasonality["DayOfWeek"] = pd.to_datetime(weekly_seasonality["ds"]).dt.day_name()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        weekly_avg = weekly_seasonality[weekly_seasonality["DayOfWeek"].isin(weekday_order)].groupby("DayOfWeek")["weekly"].mean().reindex(weekday_order)
        
        strong_day = weekly_avg.idxmax()
        weak_day = weekly_avg.idxmin()
        
        # Monthly seasonality
        monthly_seasonality = forecast[["ds", "yearly"]].copy()
        monthly_seasonality["Month"] = pd.to_datetime(monthly_seasonality["ds"]).dt.strftime('%B')
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_avg = monthly_seasonality.groupby("Month")["yearly"].mean().reindex(month_order)
        
        strong_month = monthly_avg.idxmax()
        weak_month = monthly_avg.idxmin()
        
        seasonality_info = f"Seasonality analysis suggests strongest performance on {strong_day}s and in {strong_month}, with weakest performance on {weak_day}s and in {weak_month}."
    except:
        seasonality_info = "Seasonality patterns could not be determined with the available data."
    
    # Extract model evaluation metrics
    if metrics:
        model_accuracy = 100 - metrics.get("mape", 0)
        mae_value = metrics.get("mae", 0)
        model_quality = "high" if model_accuracy > 90 else "moderate" if model_accuracy > 80 else "fair" if model_accuracy > 70 else "limited"
        
        evaluation_info = f"Forecast reliability: {model_quality} ({model_accuracy:.1f}% accuracy, average error of ${mae_value:.2f})."
    else:
        evaluation_info = "Forecast reliability information is not available."
    
    # Construct the prompt for Cohere's API with all the information
    prompt = f"""
    Create a comprehensive investment summary for {ticker} that combines all forecasting insights:
    
    FORECAST SUMMARY:
    - Current price: ${recent_price:.2f}
    - Forecasted price ({forecast_days}-day): ${predicted_price:.2f} ({price_change:.2f}%)
    - Bullish scenario: ${bullish_price:.2f} ({bullish_change:.2f}%)
    - Bearish scenario: ${bearish_price:.2f} ({bearish_change:.2f}%)
    
    MODEL EVALUATION:
    {evaluation_info}
    
    SEASONALITY:
    {seasonality_info}
    
    Write a concise, professional investment analysis that:
    1. Summarizes the {forecast_days}-day price outlook with confidence level
    2. Highlights potential bullish and bearish scenarios
    3. Incorporates seasonality trends
    4. Notes key levels to watch
    5. Provides overall investment takeaway
    
    Keep the analysis straightforward and focused on actionable insights for an investor.
    """

    # Request insight from Cohere API
    try:
        response = co.generate(
            model='command-light',
            prompt=prompt + "\n\nImportant: Ensure your response ends with a complete sentence.",
            max_tokens=300,
            temperature=0.7
        )
        
        generated_text = response.generations[0].text.strip()
        
        # Check if the text ends with a complete sentence (ends with period, question mark, or exclamation point)
        if not generated_text[-1] in ['.', '?', '!']:
            generated_text += "."
            
        return generated_text
    except Exception as e:
        # Fallback in case of API error
        return f"""
        {ticker} {forecast_days}-Day Forecast Summary:
        
        The model predicts a {price_change:.2f}% {'increase' if price_change > 0 else 'decrease'} from the current ${recent_price:.2f} to ${predicted_price:.2f}.
        
        Bullish scenario: ${bullish_price:.2f} ({bullish_change:.2f}%)
        Bearish scenario: ${bearish_price:.2f} ({bearish_change:.2f}%)
        
        {seasonality_info}
        
        {evaluation_info}
        
        This analysis is based on historical patterns and should be considered alongside broader market conditions and company fundamentals.
        """

# ---------------- Forecasting Logic ---------------- #
if ticker:
    st.markdown(f"### 📈 Forecasting Close Price for {ticker.upper()} (Prophet Model)")

    try:
        df = get_data(ticker, start_date)
        if df.empty or "Close" not in df.columns or "Date" not in df.columns:
            st.warning("No data available or invalid metric.")
        else:
            df_prophet = df[["Date", "Close"]].dropna()
            df_prophet.columns = ["ds", "y"]

            model = Prophet(
                daily_seasonality=True, 
                yearly_seasonality=True,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=forecast_days)
            
            # Handle weekends
            if not include_weekends:
                future = future[future['ds'].dt.weekday < 5]  # Exclude weekends
                
            forecast = model.predict(future)

            # Forecast Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="markers", name="Actual Close", marker=dict(color='black', size=6)))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color='orange', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(width=0)))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", fill='tonexty', line=dict(width=0)))
            fig.update_layout(
                title=f"{forecast_days}-Day Forecast of {ticker.upper()} Close Price", 
                xaxis_title="Date", 
                yaxis_title="Close Price", 
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add confidence interval explanation
            add_confidence_explanation()
            
            # Add scenario analysis
            add_scenario_analysis(ticker, forecast, forecast_days)

            # --- Model Evaluation --- #
            st.markdown("---")
            st.subheader("⚙️ Model Evaluation")
            with st.expander("Understanding Evaluation Metrics"):
                st.markdown("""
                Model Evaluation metrics help assess forecast accuracy:
                - **RMSE**: Root Mean Squared Error (lower is better)
                - **MAE**: Mean Absolute Error (average prediction error in dollars)
                - **MAPE**: Mean Absolute Percentage Error (average prediction error percentage)
                
                These are calculated by testing the model against a portion of known historical data.
                """)
            
            train = df_prophet[:-forecast_days]
            test = df_prophet[-forecast_days:]
            model_eval = Prophet(
                daily_seasonality=True, 
                yearly_seasonality=True,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            model_eval.fit(train)
            future_eval = model_eval.make_future_dataframe(periods=forecast_days)
            
            if not include_weekends:
                future_eval = future_eval[future_eval['ds'].dt.weekday < 5]
                
            forecast_eval = model_eval.predict(future_eval)

            y_true = test['y'].values
            y_pred = forecast_eval['yhat'].tail(forecast_days).values

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            # Display metrics in a better format
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">RMSE</div>
                    <div class="metric-value">${rmse:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metrics_col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">MAE</div>
                    <div class="metric-value">${mae:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with metrics_col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-title">MAPE</div>
                    <div class="metric-value">{mape:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            try:
                # Create tabs for seasonality patterns
                seasonality_tabs = st.tabs(["Weekly Seasonality", "Monthly Seasonality"])
                
                with seasonality_tabs[0]:
                    # Weekly Seasonality
                    st.subheader("📅 Weekly Seasonality Pattern")
                    weekly_seasonality = forecast[["ds", "weekly"]].copy()
                    weekly_seasonality["DayOfWeek"] = pd.to_datetime(weekly_seasonality["ds"]).dt.day_name()
                    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                    weekly_avg = weekly_seasonality[weekly_seasonality["DayOfWeek"].isin(weekday_order)].groupby("DayOfWeek")["weekly"].mean().reindex(weekday_order)

                    weekly_fig = go.Figure()
                    weekly_fig.add_trace(go.Bar(x=weekly_avg.index, y=weekly_avg.values, marker_color="orange"))
                    weekly_fig.update_layout(
                        title="Average Weekly Seasonality Effect (Trading Days Only)",
                        xaxis_title="Day of Week",
                        yaxis_title="Effect",
                        template="plotly_white"
                    )
                    st.plotly_chart(weekly_fig, use_container_width=True)
                
                with seasonality_tabs[1]:
                    # Monthly Seasonality
                    st.subheader("📆 Monthly Seasonality Pattern")
                    monthly_seasonality = forecast[["ds", "yearly"]].copy()
                    monthly_seasonality["Month"] = pd.to_datetime(monthly_seasonality["ds"]).dt.strftime('%B')
                    month_order = [
                        "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"
                    ]
                    monthly_avg = monthly_seasonality.groupby("Month")["yearly"].mean().reindex(month_order)

                    monthly_fig = go.Figure()
                    monthly_fig.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, mode="lines+markers", line=dict(color='blue')))
                    monthly_fig.update_layout(
                        title="Average Monthly Seasonality Effect",
                        xaxis_title="Month",
                        yaxis_title="Effect",
                        template="plotly_white"
                    )
                    st.plotly_chart(monthly_fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error rendering seasonal plots: {e}")

            # --- LLM-Generated Comprehensive Insight --- #
            st.markdown("---")
            st.markdown("### 🧠 Investment Insights & Key Takeaways")

            with st.spinner("Generating comprehensive analysis..."):
                # Prepare metrics for the LLM
                metrics = {
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape
                }
                
                # Generate the comprehensive insights
                insight_text = generate_comprehensive_insights(
                    df_prophet, 
                    forecast, 
                    forecast_days, 
                    ticker, 
                    metrics
                )
                
                # Display the insights in a clean, prominent format
                st.markdown(f"""
                <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border-left:5px solid #2E7D32;">
                    <h4 style="margin-top:0;">📈 {ticker} - {forecast_days}-Day Investment Outlook</h4>
                    <p style="font-size:16px;">{insight_text}</p>
                    <div style="font-size:12px; color:#666; margin-top:15px; padding-top:10px; border-top:1px solid #ddd;">
                        <strong>Note:</strong> This analysis combines forecast data, scenario analysis, model evaluation, and seasonality patterns.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display key price levels in a clean, simple format
                last_price = df_prophet["y"].iloc[-1]
                baseline = forecast["yhat"].iloc[-1]
                bullish = forecast["yhat_upper"].iloc[-1]
                bearish = forecast["yhat_lower"].iloc[-1]
                
                # Create columns for key price levels
                st.markdown("#### Key Price Levels")
                
                key_levels_col1, key_levels_col2, key_levels_col3, key_levels_col4 = st.columns(4)
                
                with key_levels_col1:
                    st.metric(
                        label="Current Price", 
                        value=f"${last_price:.2f}"
                    )
                
                with key_levels_col2:
                    st.metric(
                        label="Target Price", 
                        value=f"${baseline:.2f}", 
                        delta=f"{((baseline - last_price) / last_price * 100):.2f}%"
                    )
                
                with key_levels_col3:
                    st.metric(
                        label="Support", 
                        value=f"${bearish:.2f}"
                    )
                
                with key_levels_col4:
                    st.metric(
                        label="Resistance", 
                        value=f"${bullish:.2f}"
                    )

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please select a stock or enter a ticker symbol to begin forecasting.")

# Display footer
display_footer()