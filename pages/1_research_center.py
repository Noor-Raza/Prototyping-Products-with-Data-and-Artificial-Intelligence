import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import requests
import json
import random
from textblob import TextBlob
import math
import random
import hashlib
import re
import os
from dotenv import load_dotenv
import cohere
load_dotenv()

# Add the parent directory to system path to import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_theme import (
    apply_custom_theme, display_header, display_footer, enhance_sidebar,
    display_metric, display_card, display_stock_card, display_alert,
    display_watchlist
)

from utils import (
    initialize_session_state, update_stock_context, get_active_stock,
    add_to_watchlist, remove_from_watchlist, get_stock_data, get_stock_info,
    get_market_indices, get_market_movers, get_economic_calendar,
    format_large_number
)

# Function to generate company description using Cohere API
def generate_company_description(ticker, company_name, business_summary):
    """Generate an enhanced company description using Cohere API"""
    api_key = "TvrgUHF3GKzAB5sYBHK7UkHApcr2VZ0nJnBkNATD"
    api_url = "https://api.cohere.ai/v1/generate"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Create a prompt for Cohere
    prompt = f"""Write a concise, single-paragraph analysis of {company_name} ({ticker}) from a stock research perspective.
    Business Summary: {business_summary}
    
    Include key business areas and market position. Keep it brief (max 100 words) but make it insightful for investors.
    Ensure your analysis ends with a complete sentence.
    """
    
    payload = {
        "model": "command",
        "prompt": prompt,
        "max_tokens": 150,  # Reduced for brevity
        "temperature": 0.7,
        "k": 0,
        "stop_sequences": [],
        "return_likelihoods": "NONE"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for unsuccessful requests
        
        result = response.json()
        description = result.get("generations", [{}])[0].get("text", "").strip()
        
        # Ensure description ends with a complete sentence (ends with period, question mark, or exclamation point)
        if description and not description[-1] in ['.', '?', '!']:
            description += '.'
            
        return description
    except Exception as e:
        st.error(f"Error generating company description: {str(e)}")
        return business_summary  # Fallback to original business summary

# Function to generate key insights using Cohere API
def generate_key_insights(ticker, stock_data, metrics, returns):
    """Generate key insights from the stock data using Cohere API"""
    api_key = "TvrgUHF3GKzAB5sYBHK7UkHApcr2VZ0nJnBkNATD"
    api_url = "https://api.cohere.ai/v1/generate"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Calculate some additional metrics to feed to Cohere
    recent_days = min(30, len(stock_data))
    
    # Convert Series values to Python native types to avoid formatting issues
    latest_price = float(stock_data['Close'].iloc[-1])
    first_price = float(stock_data['Close'].iloc[-min(recent_days, len(stock_data))])
    price_change = (latest_price / first_price - 1) * 100  # percentage
    avg_volume = float(stock_data['Volume'][-recent_days:].mean())
    
    # Calculate volatility properly
    returns_series = stock_data['Close'].pct_change().dropna()
    volatility = float(returns_series.std() * (252 ** 0.5) * 100)  # Annualized volatility as percentage
    
    data_summary = {
        "ticker": ticker,
        "latest_price": f"${latest_price:.2f}",
        "price_change_30d": f"{price_change:.2f}%",
        "avg_volume": f"{avg_volume:.0f}",
        "volatility": f"{volatility:.2f}%",
        "returns": returns,
        "metrics": {k: v for k, v in metrics.items() if k in ["52 Week High", "52 Week Low", "P/E Ratio", "Market Cap", "Dividend Yield"]}
    }
    
    # Create a prompt for Cohere
    prompt = f"""As a stock market analyst, provide 3 key insights about {ticker} based on the following data:

    Current Price: {data_summary['latest_price']}
    30-Day Price Change: {data_summary['price_change_30d']}
    Average Volume: {data_summary['avg_volume']}
    Annualized Volatility: {data_summary['volatility']}
    
    Returns:
    {"".join([f"- {k}: {v}\n" for k, v in data_summary['returns'].items()])}
    
    Key Metrics:
    {"".join([f"- {k}: {v}\n" for k, v in data_summary['metrics'].items()])}
    
    Format your response as 3 bullet points with key investment insights. Each bullet should be concise (1-2 sentences) and provide an actionable insight or risk assessment. Make sure each bullet point ends with a complete sentence.
    """
    
    payload = {
        "model": "command",
        "prompt": prompt,
        "max_tokens": 250,  # Reduced for brevity
        "temperature": 0.7,
        "k": 0,
        "stop_sequences": [],
        "return_likelihoods": "NONE"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        insights = result.get("generations", [{}])[0].get("text", "").strip()
        
        # Process each bullet point to ensure it ends with a complete sentence
        lines = insights.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                if not line[-1] in ['.', '?', '!']:
                    line += '.'
                processed_lines.append(line)
                
        return '\n'.join(processed_lines)
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return "Unable to generate insights at this time. Please try again later."
    
    # Function to generate executive summary using Cohere API
def generate_executive_summary(ticker, stock_data, stock_info, metrics, returns, health_score=None, peer_data=None):
    """Generate an executive summary with actionable insights using Cohere API"""
    api_key = "TvrgUHF3GKzAB5sYBHK7UkHApcr2VZ0nJnBkNATD"  # Using your existing key
    api_url = "https://api.cohere.ai/v1/generate"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Gather key data from various tabs
    # Price information
    latest_price = float(stock_data['Close'].iloc[-1])
    first_price = float(stock_data['Close'].iloc[0])
    price_change = (latest_price / first_price - 1) * 100  # percentage
    
    # Valuation metrics
    pe_ratio = metrics.get("P/E Ratio", "N/A")
    market_cap = metrics.get("Market Cap", "N/A")
    
    # Performance returns
    one_month_return = returns.get("1M", "N/A")
    three_month_return = returns.get("3M", "N/A")
    six_month_return = returns.get("6M", "N/A")
    
    # Financial health if available
    financial_health = "Not available"
    health_score_value = None
    if health_score and isinstance(health_score, (int, float)):
        health_score_value = health_score
        if health_score > 70:
            financial_health = "Strong"
        elif health_score > 50:
            financial_health = "Moderate"
        else:
            financial_health = "Concerning"
    
    # Profitability metrics
    profit_margin = stock_info.get("profitMargins", "N/A")
    revenue_growth = stock_info.get("revenueGrowth", "N/A")
    
    # Debt metrics
    debt_to_equity = stock_info.get("debtToEquity", "N/A")
    
    # Dividend information
    dividend_yield = stock_info.get("dividendYield", 0)
    dividend_info = f"{dividend_yield*100:.2f}%" if dividend_yield else "No dividend"
    
    # Peer comparison insights
    peer_insights = ""
    if peer_data is not None and isinstance(peer_data, dict) and ticker in peer_data:
        # Find out if the stock is outperforming peers in key metrics
        peer_tickers = list(peer_data.keys())
        peer_tickers.remove(ticker)
        
        # Compare P/E ratio if available
        pe_values = [data.get('P/E Ratio', 0) for peer, data in peer_data.items() if peer != ticker and data.get('P/E Ratio', 0) > 0]
        if pe_values and isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
            avg_pe = sum(pe_values) / len(pe_values)
            pe_comparison = f"P/E ratio is {'lower' if pe_ratio < avg_pe else 'higher'} than peer average ({pe_ratio:.2f} vs {avg_pe:.2f})"
            peer_insights += pe_comparison + ". "
        
        # Compare profit margins if available
        margin_values = [data.get('Profit Margin', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Profit Margin', 0) > 0]
        if margin_values and isinstance(profit_margin, (int, float)) and profit_margin > 0:
            avg_margin = sum(margin_values) / len(margin_values)
            margin_comparison = f"Profit margin is {'higher' if profit_margin > avg_margin else 'lower'} than peer average ({profit_margin:.2%} vs {avg_margin:.2%})"
            peer_insights += margin_comparison + ". "
    
    # Create a comprehensive prompt for Cohere
    prompt = f"""As an investment analyst, provide a concise summary and actionable insights for {ticker} based on the following data:

    Stock Overview:
    - Price: ${latest_price:.2f} ({'+' if price_change > 0 else ''}{price_change:.2f}% since analysis period start)
    - Market Cap: {format_large_number(market_cap) if isinstance(market_cap, (int, float)) else market_cap}
    - P/E Ratio: {pe_ratio}
    
    Performance:
    - 1-Month Return: {one_month_return}
    - 3-Month Return: {three_month_return}
    - 6-Month Return: {six_month_return}
    
    Financials:
    - Financial Health: {financial_health}
    - Profit Margin: {f"{profit_margin:.2%}" if isinstance(profit_margin, (int, float)) else profit_margin}
    - Revenue Growth: {f"{revenue_growth:.2%}" if isinstance(revenue_growth, (int, float)) else revenue_growth}
    - Debt-to-Equity: {f"{debt_to_equity:.2f}" if isinstance(debt_to_equity, (int, float)) else debt_to_equity}
    - Dividend: {dividend_info}
    
    Peer Comparison:
    {peer_insights if peer_insights else "No peer comparison data available."}
    
    Your analysis should include:
    1. A brief executive summary (2-3 sentences)
    2. Key strengths of the company (2 bullet points)
    3. Primary risks or concerns (2 bullet points)
    4. A clear investment recommendation (Buy, Hold, or Sell)
    5. Three specific action steps an investor should take

    For the recommendation:
    - Suggest Buy if the company shows strong financial health, good growth prospects, and reasonable valuation
    - Suggest Hold if the outlook is mixed or the valuation seems fair
    - Suggest Sell if there are significant concerns about financials, growth, or the valuation seems excessive

    Format each section with clear headings. Keep the analysis concise, evidence-based, and actionable.
    """
    
    payload = {
        "model": "command",
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.7,
        "k": 0,
        "stop_sequences": [],
        "return_likelihoods": "NONE"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        summary = result.get("generations", [{}])[0].get("text", "").strip()
        
        return summary
    except Exception as e:
        st.error(f"Error generating executive summary: {str(e)}")
        return "Unable to generate summary at this time. Please try again later."

# ---------------- Helpers ---------------- #
def get_last_trading_day():
    today = datetime.now()
    weekday = today.weekday()
    if weekday == 5:  # Saturday
        return today - timedelta(days=1)
    elif weekday == 6:  # Sunday
        return today - timedelta(days=2)
    return today

def fetch_stock_data(ticker, start_date=None, end_date=None):
    if end_date is None:
        end_date = get_last_trading_day()
    if start_date is None:
        start_date = end_date - timedelta(days=365)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    return df

def calculate_returns(df):
    returns = {}
    periods = {"1M": 21, "3M": 63, "6M": 126}  # Removed 1Y return
    close = df["Close"].reset_index(drop=True)

    for label, days in periods.items():
        if len(close) >= days:
            start_price = float(close.iloc[-days])
            end_price = float(close.iloc[-1])
            ret = ((end_price - start_price) / start_price) * 100
            returns[label] = f"{ret:.2f}%"
        else:
            returns[label] = "N/A"
    return returns

def format_y_axis(value, pos):
    """Format y-axis values for better readability"""
    if value >= 1e9:
        return f"${value*1e-9:.1f}B"
    elif value >= 1e6:
        return f"${value*1e-6:.1f}M"
    elif value >= 1e3:
        return f"${value*1e-3:.1f}K"
    else:
        return f"${value:.2f}"

def plot_price_with_ma(df, ticker):
    if df.empty or len(df) < 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Not enough data to plot", ha='center', va='center')
        plt.tight_layout()
        return fig
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    
    # Create a figure with better styling
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f9f9f9')
    ax.set_facecolor('#f9f9f9')
    
    # Use pandas built-in plotting which handles datetime indexes properly
    df['Close'].plot(ax=ax, label='Close Price', color='#1f77b4', linewidth=2)
    df['MA20'].plot(ax=ax, label='MA 20', color='#ff7f0e', linewidth=1.5, alpha=0.8)
    df['MA50'].plot(ax=ax, label='MA 50', color='#2ca02c', linewidth=1.5, alpha=0.8)
    
    # Format the plot
    ax.set_title(f"{ticker} Price & Moving Averages", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Enhanced legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, facecolor='white')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(df) // 250)))
    plt.xticks(rotation=45)
    
    # Format y-axis for currency
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
    
    # Add chart border
    for spine in ax.spines.values():
        spine.set_edgecolor('#dddddd')
    
    plt.tight_layout()
    return fig

def plot_volume(df):
    if df.empty or 'Volume' not in df:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No volume data available", ha='center', va='center')
        plt.tight_layout()
        return fig
    
    # Create a figure with better styling
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#f9f9f9')
    ax.set_facecolor('#f9f9f9')
    
    # Method 1: Use pandas built-in plotting (handles dates automatically)
    df['Volume'].plot(kind='bar', ax=ax, color='#1f77b4', alpha=0.7, width=0.8)
    
    # Format the plot
    ax.set_title("Trading Volume", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volume", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Format x-axis - reduce number of date labels to avoid crowding
    num_ticks = min(10, len(df))
    step_size = max(1, len(df) // num_ticks)
    tick_positions = range(0, len(df), step_size)
    tick_labels = [df.index[i].strftime('%b %d') for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    # Format y-axis for large numbers
    def volume_formatter(x, pos):
        if x >= 1e9:
            return f'{x*1e-9:.1f}B'
        elif x >= 1e6:
            return f'{x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.1f}K'
        return f'{x:.0f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(volume_formatter))
    
    # Add chart border
    for spine in ax.spines.values():
        spine.set_edgecolor('#dddddd')
    
    plt.tight_layout()
    return fig

def get_key_metrics(info):
    return {
        "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Volatility (Beta)": info.get("beta", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "Dividend Yield": f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get("dividendYield") else "N/A",
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Website": info.get("website", "N/A")
    }

# ---------------- Theme & Page Setup ---------------- #
st.set_page_config(page_title="Research Center", layout="wide")
apply_custom_theme()
enhance_sidebar()
display_header("🔬 Research Center", "Explore in-depth stock performance and company insights")

# ---------------- Main Layout ---------------- #
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()

# Date Range Filter
st.sidebar.markdown("### Date Range")

# Predefined date ranges
last_day = get_last_trading_day()
date_ranges = {
    "1 Year": last_day - timedelta(days=365),  # Changed to default
    "1 Month": last_day - timedelta(days=30),
    "3 Months": last_day - timedelta(days=90),
    "6 Months": last_day - timedelta(days=180),
    "2 Years": last_day - timedelta(days=730),
    "5 Years": last_day - timedelta(days=1825),
    "Custom": None  # For custom date selection
}

# Create radio buttons for predefined ranges - set default to 1 Year
selected_range = st.sidebar.radio("Select time period:", list(date_ranges.keys()), index=0)

# Handle custom date selection
if selected_range == "Custom":
    # Custom date range inputs
    start_date = st.sidebar.date_input(
        "Start Date",
        value=last_day - timedelta(days=365),
        max_value=last_day
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=last_day,
        max_value=datetime.now()
    )
else:
    # Use predefined range
    start_date = date_ranges[selected_range]
    end_date = last_day

# Fetch data with date filter applied
stock_data = fetch_stock_data(ticker, start_date, end_date)
stock_info = yf.Ticker(ticker).info

tabs = st.tabs(["Overview", "Financials", "Peer Comparison",  "News & Insights", "Summary & Action"])

# ----------- OVERVIEW TAB ----------- #
with tabs[0]:
    # Display current date range
    st.subheader(f"📈 {ticker} Overview")
    date_range_text = f"Data from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}"
    st.caption(date_range_text)
    
    # Create a two-column layout for better organization
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Plot price + MA
        price_fig = plot_price_with_ma(stock_data, ticker)
        st.pyplot(price_fig, use_container_width=True)
        
        # Plot volume
        volume_fig = plot_volume(stock_data)
        st.pyplot(volume_fig, use_container_width=True)
    
    with col2:
        # Performance Summary in a card-like container
        st.markdown("### 📊 Performance Summary")
        with st.container(border=True):
            returns = calculate_returns(stock_data)
            for label, value in returns.items():
                # Color-code returns (green for positive, red for negative)
                if value != "N/A" and float(value.strip('%')) >= 0:
                    st.markdown(f"**{label} Return:** <span style='color:green'>{value}</span>", unsafe_allow_html=True)
                elif value != "N/A":
                    st.markdown(f"**{label} Return:** <span style='color:red'>{value}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{label} Return:** {value}")
        
        # Key Metrics in a card-like container
        st.markdown("### 🧮 Key Metrics")
        with st.container(border=True):
            metrics = get_key_metrics(stock_info)
            for key, val in metrics.items():
                if key in ["Market Cap", "52 Week High", "52 Week Low"] and val != "N/A":
                    # Format large numbers for better readability
                    try:
                        num_val = float(val)
                        if num_val >= 1e9:
                            val = f"${num_val/1e9:.2f}B"
                        elif num_val >= 1e6:
                            val = f"${num_val/1e6:.2f}M"
                        elif num_val >= 1e3:
                            val = f"${num_val/1e3:.2f}K"
                    except:
                        pass
                st.markdown(f"**{key}:** {val}")
    
    # Company Description using Cohere API
    st.markdown("### 🏢 Company Description")
    with st.container(border=True):
        company_name = stock_info.get("shortName", ticker)
        business_summary = stock_info.get("longBusinessSummary", f"No description available for {ticker}.")
        
        with st.spinner("Generating enhanced company description..."):
            enhanced_description = generate_company_description(
                ticker, 
                company_name, 
                business_summary
            )
        
        st.markdown(enhanced_description)
    
    # Key Insights using Cohere API
    st.markdown("### 🔑 Key Insights")
    with st.container(border=True):
        with st.spinner("Generating key insights..."):
            insights = generate_key_insights(
                ticker,
                stock_data,
                metrics,
                returns
            )
        
        st.markdown(insights)

# ----------- FINANCIALS TAB ----------- #
with tabs[1]:
    st.subheader(f"💰 {ticker} Financial Analysis")
    
    # Add a high-level financial health meter at the top
    try:
        # Calculate a simple financial health score based on available metrics
        health_score = 0
        health_factors = 0
        
        # Check debt metrics
        if isinstance(stock_info.get("debtToEquity"), (int, float)):
            debt_to_equity = stock_info.get("debtToEquity")
            if debt_to_equity < 0.3:
                health_score += 100
            elif debt_to_equity < 0.5:
                health_score += 80
            elif debt_to_equity < 1:
                health_score += 60
            elif debt_to_equity < 1.5:
                health_score += 40
            else:
                health_score += 20
            health_factors += 1
        
        # Check profitability
        if isinstance(stock_info.get("returnOnEquity"), (int, float)):
            roe = stock_info.get("returnOnEquity")
            if roe > 0.2:
                health_score += 100
            elif roe > 0.15:
                health_score += 80
            elif roe > 0.1:
                health_score += 60
            elif roe > 0.05:
                health_score += 40
            elif roe > 0:
                health_score += 20
            else:
                health_score += 0
            health_factors += 1
        
        # Check current ratio
        if isinstance(stock_info.get("currentRatio"), (int, float)):
            current_ratio = stock_info.get("currentRatio")
            if current_ratio > 2:
                health_score += 100
            elif current_ratio > 1.5:
                health_score += 80
            elif current_ratio > 1:
                health_score += 60
            elif current_ratio > 0.8:
                health_score += 40
            else:
                health_score += 20
            health_factors += 1
        
        # Calculate final score
        if health_factors > 0:
            final_health_score = health_score / health_factors
            
            # Display a gauge chart for financial health
            health_categories = {
                (0, 30): {"label": "Poor", "color": "red"},
                (30, 50): {"label": "Caution", "color": "orange"},
                (50, 70): {"label": "Moderate", "color": "yellow"},
                (70, 90): {"label": "Good", "color": "lightgreen"},
                (90, 101): {"label": "Excellent", "color": "green"}
            }
            
            # Find the health category
            health_label = "N/A"
            health_color = "gray"
            for score_range, category in health_categories.items():
                if score_range[0] <= final_health_score < score_range[1]:
                    health_label = category["label"]
                    health_color = category["color"]
            
            # Create a visual meter
            st.markdown("### 🏥 Financial Health Score")
            health_col1, health_col2, health_col3 = st.columns([1, 3, 1])
            
            with health_col2:
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="background: linear-gradient(to right, red, orange, yellow, lightgreen, green); 
                         height: 20px; border-radius: 10px; position: relative; margin-top: 10px;">
                        <div style="position: absolute; left: {final_health_score}%; top: -20px; transform: translateX(-50%);">
                            <div style="font-size: 24px;">▼</div>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span>Poor</span>
                        <span>Caution</span>
                        <span>Moderate</span>
                        <span>Good</span>
                        <span>Excellent</span>
                    </div>
                    <div style="margin-top: 15px; font-size: 20px; font-weight: bold; color: {health_color};">
                        {health_label} ({final_health_score:.1f}/100)
                    </div>
                </div>
                """, unsafe_allow_html=True)
    except:
        st.info("Financial health score could not be calculated due to insufficient data.")
    
    # Navigation tabs within Financial Analysis - makes it more organized
    fin_tabs = st.tabs(["Key Ratios", "Statements", "Growth & Cash Flow", "Dividends"])
    
    # ----------- KEY RATIOS TAB ----------- #
    with fin_tabs[0]:
        # Financial Ratios Section with better visualization
        st.markdown("### 📊 Financial Ratios & Metrics")
        
        # Create three columns for different ratio categories
        ratio_col1, ratio_col2, ratio_col3 = st.columns(3)
        
        with ratio_col1:
            st.markdown("""
            <div style="background-color: #f0f7ff; padding: 10px; border-left: 5px solid #1565C0; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #1565C0;">Valuation Ratios</h4>
            </div>
            """, unsafe_allow_html=True)
            
            valuation_ratios = {
                "P/E Ratio": stock_info.get("trailingPE", "N/A"),
                "Forward P/E": stock_info.get("forwardPE", "N/A"),
                "PEG Ratio": stock_info.get("pegRatio", "N/A"),
                "Price/Sales": stock_info.get("priceToSalesTrailing12Months", "N/A"),
                "Price/Book": stock_info.get("priceToBook", "N/A"),
                "Enterprise/Revenue": stock_info.get("enterpriseToRevenue", "N/A"),
                "Enterprise/EBITDA": stock_info.get("enterpriseToEbitda", "N/A")
            }
            
            # Add benchmark ranges for some ratios
            benchmarks = {
                "P/E Ratio": {"low": 0, "med": 15, "high": 25},
                "PEG Ratio": {"low": 0, "med": 1, "high": 2},
                "Price/Book": {"low": 0, "med": 2, "high": 3},
            }
            
            for key, val in valuation_ratios.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    val_formatted = f"{val:.2f}"
                    
                    # Add visual indicator if benchmark exists
                    if key in benchmarks:
                        benchmark = benchmarks[key]
                        if val < benchmark["med"]:
                            indicator = "🟢" # Green - potentially undervalued
                            tooltip = "Potentially undervalued"
                        elif val < benchmark["high"]:
                            indicator = "🟡" # Yellow - fair value
                            tooltip = "Fairly valued"
                        else:
                            indicator = "🔴" # Red - potentially overvalued
                            tooltip = "Potentially overvalued"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span>{val_formatted} <span title="{tooltip}">{indicator}</span></span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span>{val_formatted}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**{key}:** {val}")
        
        with ratio_col2:
            st.markdown("""
            <div style="background-color: #f0fff0; padding: 10px; border-left: 5px solid #2E7D32; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #2E7D32;">Profitability Ratios</h4>
            </div>
            """, unsafe_allow_html=True)
            
            profitability_ratios = {
                "Profit Margin": stock_info.get("profitMargins", "N/A"),
                "Operating Margin": stock_info.get("operatingMargins", "N/A"),
                "Return on Assets": stock_info.get("returnOnAssets", "N/A"),
                "Return on Equity": stock_info.get("returnOnEquity", "N/A"),
                "Revenue Growth": stock_info.get("revenueGrowth", "N/A"),
                "Earnings Growth": stock_info.get("earningsGrowth", "N/A"),
                "Gross Margins": stock_info.get("grossMargins", "N/A")
            }
            
            for key, val in profitability_ratios.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    # Format as percentage
                    val_formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                    
                    # Add visual indicator based on value
                    if val > 0.15:
                        indicator = "🟢" # Excellent
                        tooltip = "Excellent"
                    elif val > 0.08:
                        indicator = "🟢" # Good
                        tooltip = "Good"
                    elif val > 0.03:
                        indicator = "🟡" # Average
                        tooltip = "Average"
                    elif val > 0:
                        indicator = "🟠" # Below average
                        tooltip = "Below average"
                    else:
                        indicator = "🔴" # Poor
                        tooltip = "Poor"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span><strong>{key}:</strong></span>
                        <span>{val_formatted} <span title="{tooltip}">{indicator}</span></span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**{key}:** {val}")
        
        with ratio_col3:
            st.markdown("""
            <div style="background-color: #fff8e1; padding: 10px; border-left: 5px solid #FF8F00; border-radius: 5px;">
                <h4 style="margin-top: 0; color: #FF8F00;">Financial Health</h4>
            </div>
            """, unsafe_allow_html=True)
            
            financial_health = {
                "Current Ratio": stock_info.get("currentRatio", "N/A"),
                "Quick Ratio": stock_info.get("quickRatio", "N/A"),
                "Debt/Equity": stock_info.get("debtToEquity", "N/A"),
                "Interest Coverage": stock_info.get("interestCoverage", "N/A"),
                "Debt/EBITDA": stock_info.get("debtToEBITDA", "N/A"),
                "Beta": stock_info.get("beta", "N/A"),
                "52-Week Change": stock_info.get("52WeekChange", "N/A")
            }
            
            for key, val in financial_health.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    # Format based on metric type
                    if key in ["52-Week Change"]:
                        val_formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                        # Color based on performance
                        if val > 0:
                            color = "green"
                        else:
                            color = "red"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span style="color: {color};">{val_formatted}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif key == "Debt/Equity":
                        val_formatted = f"{val:.2f}"
                        # Add visual indicator based on debt level
                        if val < 0.3:
                            indicator = "🟢" # Low debt - excellent
                            tooltip = "Low debt"
                        elif val < 0.6:
                            indicator = "🟢" # Moderate debt - good
                            tooltip = "Moderate debt"
                        elif val < 1.0:
                            indicator = "🟡" # Average debt
                            tooltip = "Average debt"
                        elif val < 1.5:
                            indicator = "🟠" # High debt
                            tooltip = "High debt"
                        else:
                            indicator = "🔴" # Very high debt
                            tooltip = "Very high debt"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span>{val_formatted} <span title="{tooltip}">{indicator}</span></span>
                        </div>
                        """, unsafe_allow_html=True)
                    elif key == "Current Ratio" or key == "Quick Ratio":
                        val_formatted = f"{val:.2f}"
                        # Add visual indicator based on liquidity
                        if val > 2:
                            indicator = "🟢" # Excellent liquidity
                            tooltip = "Excellent liquidity"
                        elif val > 1.5:
                            indicator = "🟢" # Good liquidity
                            tooltip = "Good liquidity"
                        elif val > 1:
                            indicator = "🟡" # Adequate liquidity
                            tooltip = "Adequate liquidity"
                        elif val > 0.8:
                            indicator = "🟠" # Concerning liquidity
                            tooltip = "Concerning liquidity"
                        else:
                            indicator = "🔴" # Poor liquidity
                            tooltip = "Poor liquidity"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span>{val_formatted} <span title="{tooltip}">{indicator}</span></span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        val_formatted = f"{val:.2f}"
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span><strong>{key}:</strong></span>
                            <span>{val_formatted}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**{key}:** {val}")
    
    # ----------- STATEMENTS TAB ----------- #
    with fin_tabs[1]:
        # Financial Summary Section
        st.markdown("### 📊 Financial Statement Highlights")
        
        # Create cards for financial highlights
        income_col, balance_col = st.columns(2)
        
        with income_col:
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 10px; color: #333;">
                    <span style="margin-right: 10px;">📈</span> Income Statement
                </h4>
            """, unsafe_allow_html=True)
            
            income_stats = {
                "Revenue": stock_info.get("totalRevenue", "N/A"),
                "Revenue Per Share": stock_info.get("revenuePerShare", "N/A"),
                "Gross Profit": stock_info.get("grossProfits", "N/A"),
                "EBITDA": stock_info.get("ebitda", "N/A"),
                "Net Income": stock_info.get("netIncomeToCommon", "N/A"),
                "EPS (TTM)": stock_info.get("trailingEps", "N/A"),
                "EPS (Forward)": stock_info.get("forwardEps", "N/A")
            }
            
            for key, val in income_stats.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    # Format large numbers
                    if key in ["Revenue", "Gross Profit", "EBITDA", "Net Income"]:
                        if abs(val) >= 1e9:
                            val_formatted = f"${val/1e9:.2f}B"
                        elif abs(val) >= 1e6:
                            val_formatted = f"${val/1e6:.2f}M"
                        elif abs(val) >= 1e3:
                            val_formatted = f"${val/1e3:.2f}K"
                        else:
                            val_formatted = f"${val:.2f}"
                    else:
                        val_formatted = f"${val:.2f}"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">{val_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with balance_col:
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; border-bottom: 1px solid #ddd; padding-bottom: 10px; color: #333;">
                    <span style="margin-right: 10px;">📊</span> Balance Sheet
                </h4>
            """, unsafe_allow_html=True)
            
            balance_stats = {
                "Total Cash": stock_info.get("totalCash", "N/A"),
                "Total Debt": stock_info.get("totalDebt", "N/A"),
                "Total Cash Per Share": stock_info.get("totalCashPerShare", "N/A"),
                "Book Value Per Share": stock_info.get("bookValue", "N/A"),
                "Total Assets": stock_info.get("totalAssets", "N/A"),
                "Current Assets": stock_info.get("currentAssets", "N/A"),
                "Market Cap": stock_info.get("marketCap", "N/A")
            }
            
            for key, val in balance_stats.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    # Format large numbers
                    if key in ["Total Cash", "Total Debt", "Total Assets", "Current Assets", "Market Cap"]:
                        if abs(val) >= 1e9:
                            val_formatted = f"${val/1e9:.2f}B"
                        elif abs(val) >= 1e6:
                            val_formatted = f"${val/1e6:.2f}M"
                        elif abs(val) >= 1e3:
                            val_formatted = f"${val/1e3:.2f}K"
                        else:
                            val_formatted = f"${val:.2f}"
                    else:
                        val_formatted = f"${val:.2f}"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">{val_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add visualization of debt-to-equity and assets
        st.markdown("### 💵 Debt vs. Equity")
        
        try:
            # Get debt and equity data
            total_debt = stock_info.get("totalDebt")
            total_equity = stock_info.get("totalStockholderEquity")
            total_assets = stock_info.get("totalAssets")
            
            if isinstance(total_debt, (int, float)) and isinstance(total_equity, (int, float)) and total_debt > 0 and total_equity > 0:
                debt_equity_data = [
                    {"category": "Debt", "value": total_debt},
                    {"category": "Equity", "value": total_equity}
                ]
                
                # Create a horizontal bar chart
                debt_equity_df = pd.DataFrame(debt_equity_data)
                
                # Format for display
                debt_equity_df["formatted_value"] = debt_equity_df["value"].apply(
                    lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else (
                        f"${x/1e6:.2f}M" if x >= 1e6 else (
                            f"${x/1e3:.2f}K" if x >= 1e3 else f"${x:.2f}"
                        )
                    )
                )
                
                # Calculate percentages
                total = debt_equity_df["value"].sum()
                debt_equity_df["percentage"] = (debt_equity_df["value"] / total * 100).round(1)
                
                # Display as horizontal stacked bar
                st.markdown(f"""
                <div style="margin-top: 20px;">
                    <div style="background-color: #f8f9fa; height: 40px; border-radius: 5px; position: relative; overflow: hidden; display: flex;">
                        <div style="background-color: #ff6b6b; width: {debt_equity_df.iloc[0]['percentage']}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                            Debt {debt_equity_df.iloc[0]['percentage']}%
                        </div>
                        <div style="background-color: #4ecdc4; width: {debt_equity_df.iloc[1]['percentage']}%; height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                            Equity {debt_equity_df.iloc[1]['percentage']}%
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <div>
                            <span style="color: #ff6b6b; font-weight: bold;">■</span> Debt: {debt_equity_df.iloc[0]['formatted_value']}
                        </div>
                        <div>
                            <span style="color: #4ecdc4; font-weight: bold;">■</span> Equity: {debt_equity_df.iloc[1]['formatted_value']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show debt-to-equity ratio with gauge
                debt_equity_ratio = total_debt / total_equity
                de_ratio_text = f"{debt_equity_ratio:.2f}"
                
                # Debt-to-equity interpretation
                if debt_equity_ratio < 0.3:
                    de_status = "Very Low"
                    de_color = "#4ecdc4"  # Green
                elif debt_equity_ratio < 0.6:
                    de_status = "Low"
                    de_color = "#a5d6a7"  # Light green
                elif debt_equity_ratio < 1.0:
                    de_status = "Moderate"
                    de_color = "#ffe082"  # Yellow
                elif debt_equity_ratio < 1.5:
                    de_status = "High"
                    de_color = "#ffab91"  # Orange
                else:
                    de_status = "Very High"
                    de_color = "#ff6b6b"  # Red
                
                st.markdown(f"""
                <div style="margin-top: 20px; text-align: center;">
                    <div style="font-size: 16px; margin-bottom: 5px;">Debt-to-Equity Ratio</div>
                    <div style="font-size: 24px; font-weight: bold; color: {de_color};">{de_ratio_text}</div>
                    <div style="color: {de_color};">{de_status} Debt Level</div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.info("Insufficient data to visualize debt and equity structure.")
        except:
            st.info("Unable to visualize debt and equity due to missing data.")
    
    # ----------- GROWTH & CASH FLOW TAB ----------- #
    with fin_tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; color: #1565C0; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                    <span style="margin-right: 10px;">📈</span> Growth Metrics
                </h4>
            """, unsafe_allow_html=True)
            
            growth_metrics = {
                "Revenue Growth (YoY)": stock_info.get("revenueGrowth", "N/A"),
                "Earnings Growth (YoY)": stock_info.get("earningsGrowth", "N/A"),
                "Free Cash Flow Growth": stock_info.get("freeCashflow", "N/A"),
                "Quarterly Earnings Growth": stock_info.get("earningsQuarterlyGrowth", "N/A"),
                "Quarterly Revenue Growth": stock_info.get("revenueQuarterlyGrowth", "N/A")
            }
            
            for key, val in growth_metrics.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    if key in ["Revenue Growth (YoY)", "Earnings Growth (YoY)", "Quarterly Earnings Growth", "Quarterly Revenue Growth"]:
                        # Format as percentage with color coding
                        val_formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                        
                        # Color code based on growth value
                        if val > 0.15:
                            color = "#4caf50"  # Green
                            icon = "↑"
                        elif val > 0:
                            color = "#8bc34a"  # Light green
                            icon = "↑"
                        elif val > -0.05:
                            color = "#ffc107"  # Yellow
                            icon = "↓"
                        else:
                            color = "#f44336"  # Red
                            icon = "↓"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                            <span>{key}</span>
                            <span style="font-weight: bold; color: {color};">{icon} {val_formatted}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Format as currency for Free Cash Flow Growth
                        if abs(val) >= 1e9:
                            val_formatted = f"${val/1e9:.2f}B"
                        elif abs(val) >= 1e6:
                            val_formatted = f"${val/1e6:.2f}M"
                        elif abs(val) >= 1e3:
                            val_formatted = f"${val/1e3:.2f}K"
                        else:
                            val_formatted = f"${val:.2f}"
                        
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                            <span>{key}</span>
                            <span style="font-weight: bold;">{val_formatted}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #f0fff0; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
                <h4 style="margin-top: 0; color: #2E7D32; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                    <span style="margin-right: 10px;">💰</span> Cash Flow Analysis
                </h4>
            """, unsafe_allow_html=True)
            
            cash_metrics = {
                "Free Cash Flow": stock_info.get("freeCashflow", "N/A"),
                "Operating Cash Flow": stock_info.get("operatingCashflow", "N/A"),
                "Levered Free Cash Flow": stock_info.get("leveredFreeCashFlow", "N/A")
            }
            
            for key, val in cash_metrics.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    # Format large numbers
                    if abs(val) >= 1e9:
                        val_formatted = f"${val/1e9:.2f}B"
                    elif abs(val) >= 1e6:
                        val_formatted = f"${val/1e6:.2f}M"
                    elif abs(val) >= 1e3:
                        val_formatted = f"${val/1e3:.2f}K"
                    else:
                        val_formatted = f"${val:.2f}"
                    
                    # Add color based on positivity
                    color = "#4caf50" if val > 0 else "#f44336"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold; color: {color};">{val_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Calculate FCF Yield if we have market cap
            try:
                fcf = stock_info.get("freeCashflow")
                market_cap = stock_info.get("marketCap")
                
                if isinstance(fcf, (int, float)) and isinstance(market_cap, (int, float)) and market_cap > 0:
                    fcf_yield = fcf / market_cap
                    
                    # Color based on FCF yield value
                    if fcf_yield > 0.08:
                        fcf_color = "#4caf50"  # Excellent
                        fcf_status = "Excellent"
                    elif fcf_yield > 0.05:
                        fcf_color = "#8bc34a"  # Good
                        fcf_status = "Good"
                    elif fcf_yield > 0.02:
                        fcf_color = "#ffc107"  # Moderate
                        fcf_status = "Moderate"
                    elif fcf_yield > 0:
                        fcf_color = "#ff9800"  # Low
                        fcf_status = "Low"
                    else:
                        fcf_color = "#f44336"  # Negative
                        fcf_status = "Negative"
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>Free Cash Flow Yield</span>
                        <span style="font-weight: bold; color: {fcf_color};">{fcf_yield:.2%} ({fcf_status})</span>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                pass
            
            # Cash Flow Per Share
            try:
                ocf = stock_info.get("operatingCashflow")
                shares_outstanding = stock_info.get("sharesOutstanding")
                
                if isinstance(ocf, (int, float)) and isinstance(shares_outstanding, (int, float)) and shares_outstanding > 0:
                    ocf_per_share = ocf / shares_outstanding
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>Operating Cash Flow Per Share</span>
                        <span style="font-weight: bold;">${ocf_per_share:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            except:
                pass
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Add Efficiency Metrics in a card
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #ddd;">
            <h4 style="margin-top: 0; color: #FF8F00; border-bottom: 1px solid #ddd; padding-bottom: 10px;">
                <span style="margin-right: 10px;">⚙️</span> Efficiency Metrics
            </h4>
        """, unsafe_allow_html=True)
        
        # Create a two-column layout for efficiency metrics
        eff_col1, eff_col2 = st.columns(2)
        
        with eff_col1:
            efficiency_metrics1 = {
                "Return on Invested Capital": stock_info.get("returnOnCapital", "N/A"),
                "Gross Profit Margin": stock_info.get("grossMargins", "N/A"),
                "Operating Margin": stock_info.get("operatingMargins", "N/A"),
            }
            
            for key, val in efficiency_metrics1.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    val_formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                    
                    # Add color based on value
                    if val > 0.15:
                        color = "#4caf50"  # Excellent
                    elif val > 0.08:
                        color = "#8bc34a"  # Good
                    elif val > 0:
                        color = "#ffc107"  # Moderate
                    else:
                        color = "#f44336"  # Poor
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold; color: {color};">{val_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with eff_col2:
            efficiency_metrics2 = {
                "Net Profit Margin": stock_info.get("profitMargins", "N/A"),
                "Asset Turnover": stock_info.get("assetTurnover", "N/A"),
                "Inventory Turnover": stock_info.get("inventoryTurnover", "N/A"),
            }
            
            for key, val in efficiency_metrics2.items():
                if val != "N/A" and isinstance(val, (int, float)):
                    if key != "Asset Turnover" and key != "Inventory Turnover":
                        val_formatted = f"{val:.2%}" if abs(val) < 10 else f"{val:.2f}"
                        
                        # Add color based on value
                        if val > 0.15:
                            color = "#4caf50"  # Excellent
                        elif val > 0.08:
                            color = "#8bc34a"  # Good
                        elif val > 0:
                            color = "#ffc107"  # Moderate
                        else:
                            color = "#f44336"  # Poor
                    else:
                        val_formatted = f"{val:.2f}"
                        color = "#000000"  # No specific color
                    
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold; color: {color};">{val_formatted}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>{key}</span>
                        <span style="font-weight: bold;">N/A</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ----------- DIVIDENDS TAB ----------- #
    with fin_tabs[3]:
        div_yield = stock_info.get("dividendYield", 0)
        
        if div_yield and isinstance(div_yield, (int, float)) and div_yield > 0:
            # Create card layout for dividend info
            st.markdown("""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #c8e6c9;">
                <h3 style="margin-top: 0; color: #2e7d32; text-align: center;">Dividend Profile</h3>
            """, unsafe_allow_html=True)
            
            # Display the yield prominently
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 16px; color: #555;">Annual Dividend Yield</div>
                <div style="font-size: 36px; font-weight: bold; color: #2e7d32;">{div_yield * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create 3-column layout for dividend metrics
            div_col1, div_col2, div_col3 = st.columns(3)
            
            with div_col1:
                div_rate = stock_info.get("dividendRate", 0)
                if isinstance(div_rate, (int, float)):
                    st.markdown(f"""
                    <div style="text-align: center; background-color: #f1f8e9; padding: 15px; border-radius: 10px;">
                        <div style="font-size: 14px; color: #555;">Annual Dividend</div>
                        <div style="font-size: 24px; font-weight: bold; color: #2e7d32;">${div_rate:.2f}</div>
                        <div style="font-size: 12px; color: #777;">per share</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with div_col2:
                payout_ratio = stock_info.get("payoutRatio", 0)
                if isinstance(payout_ratio, (int, float)):
                    # Determine status based on payout ratio
                    if payout_ratio < 0.3:
                        payout_status = "Very Safe"
                        payout_color = "#4caf50"  # Green
                    elif payout_ratio < 0.5:
                        payout_status = "Safe"
                        payout_color = "#8bc34a"  # Light green
                    elif payout_ratio < 0.75:
                        payout_status = "Adequate"
                        payout_color = "#ffc107"  # Yellow
                    else:
                        payout_status = "High"
                        payout_color = "#ff9800"  # Orange
                    
                    st.markdown(f"""
                    <div style="text-align: center; background-color: #f1f8e9; padding: 15px; border-radius: 10px;">
                        <div style="font-size: 14px; color: #555;">Payout Ratio</div>
                        <div style="font-size: 24px; font-weight: bold; color: {payout_color};">{payout_ratio * 100:.1f}%</div>
                        <div style="font-size: 12px; color: #777;">{payout_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with div_col3:
                ex_div_date = stock_info.get("exDividendDate", "N/A")
                if ex_div_date != "N/A" and isinstance(ex_div_date, (int, float)):
                    # Convert timestamp to date
                    from datetime import datetime
                    try:
                        ex_date = datetime.fromtimestamp(ex_div_date).strftime('%b %d, %Y')
                    except:
                        ex_date = "N/A"
                else:
                    ex_date = "N/A"
                
                st.markdown(f"""
                <div style="text-align: center; background-color: #f1f8e9; padding: 15px; border-radius: 10px;">
                    <div style="font-size: 14px; color: #555;">Ex-Dividend Date</div>
                    <div style="font-size: 20px; font-weight: bold; color: #2e7d32;">{ex_date}</div>
                    <div style="font-size: 12px; color: #777;">Last date to qualify</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Close the main card
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add dividend safety assessment in a separate card
            st.markdown("""
            <div style="background-color: #f1f8e9; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #c8e6c9;">
                <h4 style="margin-top: 0; color: #2e7d32;">Dividend Safety Assessment</h4>
            """, unsafe_allow_html=True)
            
            # Create a simple scoring system for dividend safety
            safety_score = 0
            safety_factors = 0
            
            # Check payout ratio
            if isinstance(payout_ratio, (int, float)):
                if payout_ratio < 0.3:
                    safety_score += 3
                elif payout_ratio < 0.5:
                    safety_score += 2
                elif payout_ratio < 0.75:
                    safety_score += 1
                safety_factors += 1
            
            # Check debt to equity
            debt_to_equity = stock_info.get("debtToEquity")
            if isinstance(debt_to_equity, (int, float)):
                if debt_to_equity < 0.3:
                    safety_score += 3
                elif debt_to_equity < 0.6:
                    safety_score += 2
                elif debt_to_equity < 1:
                    safety_score += 1
                safety_factors += 1
            
            # Check cash flow
            if isinstance(stock_info.get("operatingCashflow"), (int, float)) and isinstance(div_rate, (int, float)) and isinstance(stock_info.get("sharesOutstanding"), (int, float)):
                # Calculate dividend coverage by operating cash flow
                total_annual_div = div_rate * stock_info.get("sharesOutstanding")
                ocf = stock_info.get("operatingCashflow")
                if ocf > 0 and total_annual_div > 0:
                    div_coverage = ocf / total_annual_div
                    if div_coverage > 3:
                        safety_score += 3
                    elif div_coverage > 2:
                        safety_score += 2
                    elif div_coverage > 1:
                        safety_score += 1
                    safety_factors += 1
            
            # Calculate overall safety score
            if safety_factors > 0:
                overall_score = safety_score / safety_factors
                
                if overall_score >= 2.5:
                    safety_text = "Very Safe"
                    safety_color = "#4caf50"  # Green
                    description = "Dividend appears very secure with strong coverage and sustainable payout ratio."
                elif overall_score >= 1.5:
                    safety_text = "Safe"
                    safety_color = "#8bc34a"  # Light green
                    description = "Dividend appears stable with adequate coverage and reasonable payout ratio."
                elif overall_score >= 0.8:
                    safety_text = "Adequate"
                    safety_color = "#ffc107"  # Yellow
                    description = "Dividend coverage is acceptable but may be at risk during economic downturns."
                else:
                    safety_text = "At Risk"
                    safety_color = "#ff9800"  # Orange
                    description = "Dividend may be at risk of reduction if financial conditions deteriorate."
                
                # Display safety assessment
                st.markdown(f"""
                <div style="display: flex; margin-bottom: 20px;">
                    <div style="width: 120px; height: 120px; border-radius: 50%; background-color: {safety_color}; 
                         display: flex; align-items: center; justify-content: center; margin-right: 20px;">
                        <span style="font-size: 22px; font-weight: bold; color: white;">{safety_text}</span>
                    </div>
                    <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                        <p style="margin: 0 0 10px 0; font-size: 16px; font-weight: bold;">{safety_text} Dividend</p>
                        <p style="margin: 0; color: #555;">{description}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display factors contributing to assessment
                st.markdown("<h5 style='margin-top: 0;'>Factors Considered</h5>", unsafe_allow_html=True)
                
                factors_html = ""
                
                # Payout ratio factor
                if isinstance(payout_ratio, (int, float)):
                    status = "Excellent" if payout_ratio < 0.3 else ("Good" if payout_ratio < 0.5 else ("Fair" if payout_ratio < 0.75 else "High"))
                    color = "#4caf50" if payout_ratio < 0.3 else ("#8bc34a" if payout_ratio < 0.5 else ("#ffc107" if payout_ratio < 0.75 else "#ff9800"))
                    factors_html += f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>Payout Ratio ({payout_ratio * 100:.1f}%)</span>
                        <span style="font-weight: bold; color: {color};">{status}</span>
                    </div>
                    """
                
                # Debt to equity factor
                if isinstance(debt_to_equity, (int, float)):
                    status = "Excellent" if debt_to_equity < 0.3 else ("Good" if debt_to_equity < 0.6 else ("Fair" if debt_to_equity < 1 else "High"))
                    color = "#4caf50" if debt_to_equity < 0.3 else ("#8bc34a" if debt_to_equity < 0.6 else ("#ffc107" if debt_to_equity < 1 else "#ff9800"))
                    factors_html += f"""
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                        <span>Debt-to-Equity ({debt_to_equity:.2f})</span>
                        <span style="font-weight: bold; color: {color};">{status}</span>
                    </div>
                    """
                
                # Cash flow coverage factor
                if isinstance(stock_info.get("operatingCashflow"), (int, float)) and isinstance(div_rate, (int, float)) and isinstance(stock_info.get("sharesOutstanding"), (int, float)):
                    total_annual_div = div_rate * stock_info.get("sharesOutstanding")
                    ocf = stock_info.get("operatingCashflow")
                    if ocf > 0 and total_annual_div > 0:
                        div_coverage = ocf / total_annual_div
                        status = "Excellent" if div_coverage > 3 else ("Good" if div_coverage > 2 else ("Fair" if div_coverage > 1 else "Weak"))
                        color = "#4caf50" if div_coverage > 3 else ("#8bc34a" if div_coverage > 2 else ("#ffc107" if div_coverage > 1 else "#ff9800"))
                        factors_html += f"""
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee;">
                            <span>Cash Flow Coverage ({div_coverage:.2f}x)</span>
                            <span style="font-weight: bold; color: {color};">{status}</span>
                        </div>
                        """
                
                st.markdown(factors_html, unsafe_allow_html=True)
            else:
                st.markdown("<p>Insufficient data to assess dividend safety.</p>", unsafe_allow_html=True)
            
            # Close the safety assessment card
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Historical context note
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #e0e0e0;">
                <h4 style="margin-top: 0; color: #424242;">⚠️ Note on Dividend Data</h4>
                <p style="margin-bottom: 0;">
                    The dividend information shown here is based on the most recent data available. Historical dividend patterns, 
                    growth rates, and consistency are important factors to consider when evaluating dividend stocks. Consider researching 
                    the company's dividend history for a more complete picture.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # No dividend card
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; text-align: center; margin-top: 30px;">
                <h3 style="margin-top: 0; color: #616161;">No Dividend Information</h3>
                <p style="color: #757575; margin-bottom: 0;">
                    This company does not appear to pay a dividend based on available data.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add possible reasons
            st.markdown("""
            <div style="margin-top: 30px;">
                <h4>Common Reasons Companies Don't Pay Dividends:</h4>
                <ul>
                    <li><strong>Growth Strategy</strong>: Company may reinvest profits for expansion rather than distributing to shareholders</li>
                    <li><strong>Early Stage</strong>: Newer companies often focus on growth and market share before initiating dividends</li>
                    <li><strong>Industry Norms</strong>: Some sectors (like technology) traditionally favor stock buybacks over dividends</li>
                    <li><strong>Financial Flexibility</strong>: Retaining earnings provides flexibility for investments, acquisitions, or R&D</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            

# ----------- PEER COMPARISON TAB ----------- #
with tabs[2]:
    st.subheader(f"🏢 {ticker} Peer Comparison")
    
    # Function to get peer companies
    def get_peers(ticker):
        """Get peer companies for a given ticker"""
        try:
            stock = yf.Ticker(ticker)
            
            # First try to get peer group if available
            peers = stock.info.get('peerGroup', [])
            
            # If no peers found, check if there's a recommendedSymbols attribute
            if not peers or len(peers) == 0:
                try:
                    if hasattr(stock, 'recommendations') and stock.recommendations is not None:
                        peers = stock.recommendations.get('recommendedSymbols', [])
                except:
                    peers = []
            
            # If still no peers, use a default set based on sector
            if not peers or len(peers) == 0:
                sector = stock.info.get('sector', '')
                if sector == 'Technology':
                    peers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
                elif sector == 'Financial Services':
                    peers = ['JPM', 'BAC', 'WFC', 'C', 'GS']
                elif sector == 'Healthcare':
                    peers = ['JNJ', 'PFE', 'MRK', 'ABBV', 'UNH']
                elif sector == 'Consumer Cyclical':
                    peers = ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD']
                elif sector == 'Communication Services':
                    peers = ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA']
                elif sector == 'Energy':
                    peers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
                else:
                    peers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
            
            # Ensure current ticker is not in the peers list and limit to 5
            if ticker in peers:
                peers.remove(ticker)
            peers = [p for p in peers if p != ticker][:5]
            
            # Add the current ticker to the beginning
            peers.insert(0, ticker)
            return peers
        except Exception as e:
            st.error(f"Error getting peer companies: {str(e)}")
            return [ticker, 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Get peer companies
    peers = get_peers(ticker)
    
    with st.spinner("Loading peer comparison data..."):
        # Fetch data for all peers
        peer_data = {}
        for peer in peers:
            try:
                info = yf.Ticker(peer).info
                # Get basic info
                peer_data[peer] = {
                    'Name': info.get('shortName', peer),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Market Cap': info.get('marketCap', 0),
                    'Price': info.get('currentPrice', 0),
                    'P/E Ratio': info.get('trailingPE', 0),
                    'Forward P/E': info.get('forwardPE', 0),
                    'P/S Ratio': info.get('priceToSalesTrailing12Months', 0),
                    'P/B Ratio': info.get('priceToBook', 0),
                    'EV/EBITDA': info.get('enterpriseToEbitda', 0),
                    'Gross Margin': info.get('grossMargins', 0),
                    'Operating Margin': info.get('operatingMargins', 0),
                    'Profit Margin': info.get('profitMargins', 0),
                    'ROE': info.get('returnOnEquity', 0),
                    'ROA': info.get('returnOnAssets', 0),
                    'Revenue Growth': info.get('revenueGrowth', 0),
                    'Earnings Growth': info.get('earningsGrowth', 0),
                    'Dividend Yield': info.get('dividendYield', 0),
                    'Beta': info.get('beta', 0),
                    '52W High': info.get('fiftyTwoWeekHigh', 0),
                    '52W Low': info.get('fiftyTwoWeekLow', 0),
                    '52W Change': info.get('52WeekChange', 0),
                }
            except Exception as e:
                st.warning(f"Error loading data for {peer}: {str(e)}")
                if peer != ticker:  # Skip warning for main ticker
                    continue
        
        # Create comparison DataFrame
        if peer_data:
            # Set of metrics for comparison
            valuation_metrics = ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio', 'EV/EBITDA']
            profitability_metrics = ['Gross Margin', 'Operating Margin', 'Profit Margin', 'ROE', 'ROA']
            growth_metrics = ['Revenue Growth', 'Earnings Growth']
            other_metrics = ['Dividend Yield', 'Beta', '52W Change']
            
            # Create navigation tabs for different metric categories
            peer_tabs = st.tabs(["Overview", "Valuation", "Profitability", "Growth", "Other Metrics"])
            
            # OVERVIEW TAB
            with peer_tabs[0]:
                # Create a summary card for each peer
                st.markdown("### Peer Companies - Key Metrics Overview")
                
                # Function to format market cap
                def format_market_cap(val):
                    if val >= 1e12:
                        return f"${val/1e12:.2f}T"
                    elif val >= 1e9:
                        return f"${val/1e9:.2f}B"
                    elif val >= 1e6:
                        return f"${val/1e6:.2f}M"
                    else:
                        return f"${val:.2f}"
                
                # Create cards for each peer company
                cols = st.columns(len(peer_data))
                
                for i, (peer, data) in enumerate(peer_data.items()):
                    with cols[i]:
                        # Highlight the current ticker with a special border
                        border_color = "#4CAF50" if peer == ticker else "#e0e0e0"
                        border_width = "3px" if peer == ticker else "1px"
                        
                        st.markdown(f"""
                        <div style="border: {border_width} solid {border_color}; border-radius: 10px; padding: 15px; height: 100%;">
                            <h4 style="margin-top: 0; color: {'#2E7D32' if peer == ticker else '#333'}; text-align: center;">
                                {data['Name']} ({peer})
                            </h4>
                            <div style="margin: 15px 0; text-align: center;">
                                <div style="font-size: 14px; color: #666;">Market Cap</div>
                                <div style="font-size: 18px; font-weight: bold;">
                                    {format_market_cap(data['Market Cap']) if data['Market Cap'] else 'N/A'}
                                </div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #eee;">
                                <span>P/E Ratio</span>
                                <span style="font-weight: bold;">{f"{data['P/E Ratio']:.2f}" if data['P/E Ratio'] else 'N/A'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #eee;">
                                <span>Profit Margin</span>
                                <span style="font-weight: bold;">{f"{data['Profit Margin']:.2%}" if data['Profit Margin'] else 'N/A'}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #eee;">
                                <span>Rev. Growth</span>
                                <span style="font-weight: bold; color: {'green' if data['Revenue Growth'] > 0 else 'red'}">
                                    {f"{data['Revenue Growth']:.2%}" if isinstance(data['Revenue Growth'], (int, float)) and data['Revenue Growth'] != 0 else 'N/A'}
                                </span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #eee;">
                                <span>ROE</span>
                                <span style="font-weight: bold; color: {'green' if data['ROE'] > 0 else 'red'}">
                                    {f"{data['ROE']:.2%}" if isinstance(data['ROE'], (int, float)) and data['ROE'] != 0 else 'N/A'}
                                </span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span>Dividend Yield</span>
                                <span style="font-weight: bold;">
                                    {f"{data['Dividend Yield']:.2%}" if isinstance(data['Dividend Yield'], (int, float)) and data['Dividend Yield'] != 0 else 'N/A'}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add industry context
                main_sector = peer_data[ticker]['Sector']
                main_industry = peer_data[ticker]['Industry']
                
                st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="margin-top: 0;">{ticker} Industry Context</h4>
                    <p><strong>Sector:</strong> {main_sector}</p>
                    <p><strong>Industry:</strong> {main_industry}</p>
                    <p>The peer companies shown above are selected based on similar business models, 
                    market capitalization, or industry classification to provide relevant comparison points.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # VALUATION TAB
            with peer_tabs[1]:
                st.markdown("### Valuation Metrics Comparison")
                
                # Create a DataFrame for valuation metrics
                valuation_df = pd.DataFrame({
                    'Metric': valuation_metrics
                })
                
                # Add data for each peer
                for peer, data in peer_data.items():
                    valuation_df[peer] = [data.get(metric, 'N/A') for metric in valuation_metrics]
                
                # Calculate average for each metric (excluding the main ticker)
                average_vals = []
                for metric in valuation_metrics:
                    values = [data.get(metric, 0) for peer, data in peer_data.items() if peer != ticker and data.get(metric, 0) > 0]
                    avg = sum(values) / len(values) if values else 0
                    average_vals.append(avg)
                
                valuation_df['Peer Avg'] = average_vals
                
                # Format the DataFrame
                valuation_df_formatted = valuation_df.copy()
                for col in valuation_df.columns[1:]:
                    valuation_df_formatted[col] = valuation_df[col].apply(
                        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x > 0 else "N/A"
                    )
                
                # Display the DataFrame
                st.dataframe(valuation_df_formatted.set_index('Metric'), use_container_width=True)
                
                # Create visualizations for valuation metrics
                st.markdown("#### Visualization")
                
                # Select a metric to visualize
                selected_valuation = st.selectbox(
                    "Select a valuation metric to visualize:",
                    valuation_metrics
                )
                
                # Prepare data for plotting
                plot_data = {'Peer': [], 'Value': []}
                for peer, data in peer_data.items():
                    if data.get(selected_valuation, 0) > 0:
                        plot_data['Peer'].append(peer)
                        plot_data['Value'].append(data.get(selected_valuation, 0))
                
                plot_df = pd.DataFrame(plot_data)
                
                if not plot_df.empty:
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(
                        plot_df['Peer'], 
                        plot_df['Value'], 
                        color=['#4CAF50' if peer == ticker else '#90CAF9' for peer in plot_df['Peer']]
                    )
                    
                    # Add peer average line
                    peer_avg = sum([v for i, v in enumerate(plot_df['Value']) if plot_df['Peer'].iloc[i] != ticker]) / (len(plot_df) - 1) if len(plot_df) > 1 else 0
                    ax.axhline(y=peer_avg, color='#FF9800', linestyle='--', label=f'Peer Average: {peer_avg:.2f}')
                    
                    # Format the plot
                    ax.set_title(f"{selected_valuation} Comparison", fontsize=14)
                    ax.set_xlabel("Companies", fontsize=12)
                    ax.set_ylabel(selected_valuation, fontsize=12)
                    
                    # Add data labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.02 * max(plot_df['Value']),
                            f'{height:.2f}',
                            ha='center', va='bottom', fontweight='bold'
                        )
                    
                    ax.legend()
                    ax.grid(True, alpha=0.3, linestyle='--')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Add interpretation
                    if selected_valuation in ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio', 'EV/EBITDA']:
                        main_val = peer_data[ticker].get(selected_valuation, 0)
                        if main_val > 0 and peer_avg > 0:
                            premium_discount = (main_val / peer_avg - 1) * 100
                            
                            if premium_discount > 15:
                                interpretation = f"{ticker} is trading at a significant premium compared to peers."
                                potential_reasons = "This could be due to stronger growth prospects, better profitability, market leadership, or investor sentiment."
                            elif premium_discount > 5:
                                interpretation = f"{ticker} is trading at a moderate premium compared to peers."
                                potential_reasons = "This suggests investors may be willing to pay more due to perceived competitive advantages or growth potential."
                            elif premium_discount > -5:
                                interpretation = f"{ticker} is trading in line with its peer group."
                                potential_reasons = "This suggests the market values the company similarly to its competitors."
                            elif premium_discount > -15:
                                interpretation = f"{ticker} is trading at a moderate discount compared to peers."
                                potential_reasons = "This may represent a potential value opportunity if the discount is not justified by fundamentals."
                            else:
                                interpretation = f"{ticker} is trading at a significant discount compared to peers."
                                potential_reasons = "This could indicate underlying concerns about the company's prospects, or potentially an undervalued opportunity."
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
                                <h4 style="margin-top: 0;">Interpretation</h4>
                                <p>{interpretation} {ticker}'s {selected_valuation} of <strong>{main_val:.2f}</strong> is 
                                <strong style="color: {'green' if premium_discount < 0 else 'red'};">
                                    {abs(premium_discount):.1f}% {'below' if premium_discount < 0 else 'above'}
                                </strong> the peer average of <strong>{peer_avg:.2f}</strong>.</p>
                                <p>{potential_reasons}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("Insufficient data to create visualization for the selected metric.")
            
            # PROFITABILITY TAB
            with peer_tabs[2]:
                st.markdown("### Profitability Metrics Comparison")
                
                # Create a DataFrame for profitability metrics
                profitability_df = pd.DataFrame({
                    'Metric': profitability_metrics
                })
                
                # Add data for each peer
                for peer, data in peer_data.items():
                    profitability_df[peer] = [data.get(metric, 'N/A') for metric in profitability_metrics]
                
                # Calculate average for each metric (excluding the main ticker)
                average_prof_vals = []
                for metric in profitability_metrics:
                    values = [data.get(metric, 0) for peer, data in peer_data.items() if peer != ticker and data.get(metric, 0) > 0]
                    avg = sum(values) / len(values) if values else 0
                    average_prof_vals.append(avg)
                
                profitability_df['Peer Avg'] = average_prof_vals
                
                # Format the DataFrame (as percentages)
                profitability_df_formatted = profitability_df.copy()
                for col in profitability_df.columns[1:]:
                    profitability_df_formatted[col] = profitability_df[col].apply(
                        lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x != 'N/A' else "N/A"
                    )
                
                # Display the DataFrame
                st.dataframe(profitability_df_formatted.set_index('Metric'), use_container_width=True)
                
                # Create visualizations for profitability metrics
                st.markdown("#### Visualization")
                
                # Select a metric to visualize
                selected_profitability = st.selectbox(
                    "Select a profitability metric to visualize:",
                    profitability_metrics
                )
                
                # Prepare data for plotting
                plot_data = {'Peer': [], 'Value': []}
                for peer, data in peer_data.items():
                    if data.get(selected_profitability, 0) > 0:
                        plot_data['Peer'].append(peer)
                        plot_data['Value'].append(data.get(selected_profitability, 0))
                
                plot_df = pd.DataFrame(plot_data)
                
                if not plot_df.empty:
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(
                        plot_df['Peer'], 
                        plot_df['Value'], 
                        color=['#4CAF50' if peer == ticker else '#90CAF9' for peer in plot_df['Peer']]
                    )
                    
                    # Add peer average line
                    peer_avg = sum([v for i, v in enumerate(plot_df['Value']) if plot_df['Peer'].iloc[i] != ticker]) / (len(plot_df) - 1) if len(plot_df) > 1 else 0
                    ax.axhline(y=peer_avg, color='#FF9800', linestyle='--', label=f'Peer Average: {peer_avg:.2%}')
                    
                    # Format the plot
                    ax.set_title(f"{selected_profitability} Comparison", fontsize=14)
                    ax.set_xlabel("Companies", fontsize=12)
                    ax.set_ylabel(selected_profitability, fontsize=12)
                    
                    # Format y-axis as percentage
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                    
                    # Add data labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.02 * max(plot_df['Value']),
                            f'{height:.1%}',
                            ha='center', va='bottom', fontweight='bold'
                        )
                    
                    ax.legend()
                    ax.grid(True, alpha=0.3, linestyle='--')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Add interpretation
                    main_val = peer_data[ticker].get(selected_profitability, 0)
                    if main_val > 0 and peer_avg > 0:
                        relative_perf = (main_val / peer_avg - 1) * 100
                        
                        if relative_perf > 20:
                            interpretation = f"{ticker} significantly outperforms peers in {selected_profitability}."
                            implications = "This suggests strong competitive advantages, efficient operations, or pricing power relative to competitors."
                        elif relative_perf > 5:
                            interpretation = f"{ticker} outperforms peers in {selected_profitability}."
                            implications = "This indicates better-than-average operational efficiency or business model advantages."
                        elif relative_perf > -5:
                            interpretation = f"{ticker}'s {selected_profitability} is in line with peer average."
                            implications = "The company performs comparably to its competitors in this metric."
                        elif relative_perf > -20:
                            interpretation = f"{ticker} underperforms peers in {selected_profitability}."
                            implications = "This may indicate operational challenges, competitive pressures, or strategic investment priorities."
                        else:
                            interpretation = f"{ticker} significantly underperforms peers in {selected_profitability}."
                            implications = "This could signal fundamental business challenges that may require strategic attention."
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
                            <h4 style="margin-top: 0;">Interpretation</h4>
                            <p>{interpretation} {ticker}'s {selected_profitability} of <strong>{main_val:.2%}</strong> is 
                            <strong style="color: {'green' if relative_perf > 0 else 'red'};">
                                {abs(relative_perf):.1f}% {'above' if relative_perf > 0 else 'below'}
                            </strong> the peer average of <strong>{peer_avg:.2%}</strong>.</p>
                            <p>{implications}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Insufficient data to create visualization for the selected metric.")
            
            # GROWTH TAB
            with peer_tabs[3]:
                st.markdown("### Growth Metrics Comparison")
                
                # Create a DataFrame for growth metrics
                growth_df = pd.DataFrame({
                    'Metric': growth_metrics
                })
                
                # Add data for each peer
                for peer, data in peer_data.items():
                    growth_df[peer] = [data.get(metric, 'N/A') for metric in growth_metrics]
                
                # Calculate average for each metric (excluding the main ticker)
                average_growth_vals = []
                for metric in growth_metrics:
                    values = [data.get(metric, 0) for peer, data in peer_data.items() if peer != ticker and data.get(metric, 0) > 0]
                    avg = sum(values) / len(values) if values else 0
                    average_growth_vals.append(avg)
                
                growth_df['Peer Avg'] = average_growth_vals
                
                # Format the DataFrame (as percentages)
                growth_df_formatted = growth_df.copy()
                for col in growth_df.columns[1:]:
                    growth_df_formatted[col] = growth_df[col].apply(
                        lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x != 'N/A' else "N/A"
                    )
                
                # Display the DataFrame
                st.dataframe(growth_df_formatted.set_index('Metric'), use_container_width=True)
                
                # Create radar chart for growth metrics
                st.markdown("#### Growth Performance Radar Chart")
                
                # Prepare data for radar chart
                radar_metrics = growth_metrics
                
                # Get values for current ticker and calculate peer average
                ticker_values = []
                peer_avg_values = []
                
                for metric in radar_metrics:
                    ticker_val = peer_data[ticker].get(metric, 0)
                    ticker_values.append(max(0, ticker_val))  # Ensure no negative values for radar chart
                    
                    # Calculate peer average (excluding main ticker)
                    peer_vals = [data.get(metric, 0) for peer, data in peer_data.items() if peer != ticker and data.get(metric, 0) > 0]
                    peer_avg = sum(peer_vals) / len(peer_vals) if peer_vals else 0
                    peer_avg_values.append(max(0, peer_avg))  # Ensure no negative values for radar chart
                
                # Create radar chart
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, polar=True)
                
                # Number of metrics
                N = len(radar_metrics)
                
                # Angle of each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the loop
                
                # Add ticker values
                ticker_values += ticker_values[:1]  # Close the loop
                ax.plot(angles, ticker_values, 'g-', linewidth=2, label=ticker)
                ax.fill(angles, ticker_values, 'g', alpha=0.25)
                
                # Add peer average values
                peer_avg_values += peer_avg_values[:1]  # Close the loop
                ax.plot(angles, peer_avg_values, 'b-', linewidth=2, label='Peer Average')
                ax.fill(angles, peer_avg_values, 'b', alpha=0.1)
                
                # Set labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_metrics)
                
                # Add legend and title
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('Growth Metrics Comparison', size=15, y=1.1)
                
                # Display the chart
                st.pyplot(fig)
                
                # Add growth summary
                main_rev_growth = peer_data[ticker].get('Revenue Growth', 0)
                main_earn_growth = peer_data[ticker].get('Earnings Growth', 0)
                peer_rev_growth_avg = sum([data.get('Revenue Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Revenue Growth', 0) > 0]) / len([data.get('Revenue Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Revenue Growth', 0) > 0]) if len([data.get('Revenue Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Revenue Growth', 0) > 0]) > 0 else 0
                peer_earn_growth_avg = sum([data.get('Earnings Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Earnings Growth', 0) > 0]) / len([data.get('Earnings Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Earnings Growth', 0) > 0]) if len([data.get('Earnings Growth', 0) for peer, data in peer_data.items() if peer != ticker and data.get('Earnings Growth', 0) > 0]) > 0 else 0
                
                # Determine growth positioning
                if main_rev_growth > peer_rev_growth_avg * 1.2 and main_earn_growth > peer_earn_growth_avg * 1.2:
                    growth_position = "outperforming peers significantly"
                    growth_color = "green"
                elif main_rev_growth > peer_rev_growth_avg and main_earn_growth > peer_earn_growth_avg:
                    growth_position = "outperforming peers"
                    growth_color = "green"
                elif main_rev_growth < peer_rev_growth_avg * 0.8 and main_earn_growth < peer_earn_growth_avg * 0.8:
                    growth_position = "underperforming peers significantly"
                    growth_color = "red"
                elif main_rev_growth < peer_rev_growth_avg and main_earn_growth < peer_earn_growth_avg:
                    growth_position = "underperforming peers"
                    growth_color = "red"
                else:
                    growth_position = "performing mixed relative to peers"
                    growth_color = "orange"
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;">
                    <h4 style="margin-top: 0;">Growth Summary</h4>
                    <p>{ticker} is <strong style="color: {growth_color};">{growth_position}</strong> in terms of growth metrics.</p>
                    <ul>
                        <li>Revenue Growth: <strong>{main_rev_growth:.2%}</strong> (Peer Avg: {peer_rev_growth_avg:.2%})</li>
                        <li>Earnings Growth: <strong>{main_earn_growth:.2%}</strong> (Peer Avg: {peer_earn_growth_avg:.2%})</li>
                    </ul>
                    <p>A company's growth rates relative to peers can indicate competitive positioning and future market share potential.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # OTHER METRICS TAB
            with peer_tabs[4]:
                st.markdown("### Other Key Metrics Comparison")
                
                # Create a DataFrame for other metrics
                other_df = pd.DataFrame({
                    'Metric': other_metrics
                })
                
                # Add data for each peer
                for peer, data in peer_data.items():
                    other_df[peer] = [data.get(metric, 'N/A') for metric in other_metrics]
                
                # Calculate average for each metric (excluding the main ticker)
                average_other_vals = []
                for metric in other_metrics:
                    values = [data.get(metric, 0) for peer, data in peer_data.items() if peer != ticker and data.get(metric, 0) > 0]
                    avg = sum(values) / len(values) if values else 0
                    average_other_vals.append(avg)
                
                other_df['Peer Avg'] = average_other_vals
                
                # Format the DataFrame
                other_df_formatted = other_df.copy()
                for i, metric in enumerate(other_metrics):
                    for col in other_df.columns[1:]:
                        value = other_df.at[i, col]
                        if metric == 'Dividend Yield' or metric == '52W Change':
                            if isinstance(value, (int, float)) and value != 'N/A':
                                other_df_formatted.at[i, col] = f"{value:.2%}"
                            else:
                                other_df_formatted.at[i, col] = "N/A"
                        else:
                            if isinstance(value, (int, float)) and value != 'N/A':
                                other_df_formatted.at[i, col] = f"{value:.2f}"
                            else:
                                other_df_formatted.at[i, col] = "N/A"
                
                # Display the DataFrame
                st.dataframe(other_df_formatted.set_index('Metric'), use_container_width=True)
                
                # Create a 2x2 grid layout for visualizations
                st.markdown("### Visualizations")
                row1_col1, row1_col2 = st.columns(2)
                row2_col1, row2_col2 = st.columns(2)
                row3_col1, row3_col2 = st.columns([3, 2])
                
                # 1. Beta Comparison in top-left (row1_col1)
                with row1_col1:
                    st.markdown("#### Beta Comparison")
                    
                    # Prepare data for plotting
                    beta_data = {'Peer': [], 'Beta': []}
                    for peer, data in peer_data.items():
                        if data.get('Beta', 0) > 0:
                            beta_data['Peer'].append(peer)
                            beta_data['Beta'].append(data.get('Beta', 0))
                    
                    beta_df = pd.DataFrame(beta_data)
                    
                    if not beta_df.empty:
                        # Create a horizontal bar chart with smaller size
                        fig, ax = plt.subplots(figsize=(5, 4))
                        
                        # Sort by beta value
                        beta_df = beta_df.sort_values(by='Beta')
                        
                        # Define colors based on beta value
                        colors = []
                        for b in beta_df['Beta']:
                            if b < 0.8:
                                colors.append('#4CAF50')  # Green for low volatility
                            elif b < 1.2:
                                colors.append('#FFC107')  # Yellow for market-like volatility
                            else:
                                colors.append('#F44336')  # Red for high volatility
                        
                        # Create horizontal bars
                        bars = ax.barh(beta_df['Peer'], beta_df['Beta'], color=colors)
                        
                        # Add market line (beta = 1)
                        ax.axvline(x=1, color='black', linestyle='--', alpha=0.7, label='Market (β=1)')
                        
                        # Format the plot
                        ax.set_title("Beta (Volatility)", fontsize=12)
                        ax.set_xlabel("Beta Value", fontsize=10)
                        
                        # Add data labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(
                                width + 0.05,
                                bar.get_y() + bar.get_height()/2.,
                                f'{width:.2f}',
                                ha='left', va='center', fontweight='bold', fontsize=8
                            )
                        
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                        
                        # Add simplified beta interpretation
                        main_beta = peer_data[ticker].get('Beta', 0)
                        if main_beta > 0:
                            if main_beta < 0.8:
                                beta_desc = "low volatility"
                            elif main_beta < 1.2:
                                beta_desc = "market-like volatility"
                            else:
                                beta_desc = "high volatility"
                            
                            st.markdown(f"**{ticker}** (β={main_beta:.2f}) shows **{beta_desc}** vs. peer avg β={beta_df['Beta'].mean():.2f}")
                    else:
                        st.info("Insufficient beta data.")
                
                # 2. 52-Week Performance Comparison in top-right (row1_col2)
                with row1_col2:
                    st.markdown("#### 52-Week Performance")
                    
                    # Prepare data for plotting
                    perf_data = {'Peer': [], '52W Change': []}
                    for peer, data in peer_data.items():
                        if data.get('52W Change', 0) != 0:
                            perf_data['Peer'].append(peer)
                            perf_data['52W Change'].append(data.get('52W Change', 0))
                    
                    perf_df = pd.DataFrame(perf_data)
                    
                    if not perf_df.empty:
                        # Create a horizontal bar chart with smaller size
                        fig, ax = plt.subplots(figsize=(5, 4))
                        
                        # Sort by performance
                        perf_df = perf_df.sort_values(by='52W Change')
                        
                        # Define colors based on performance
                        colors = ['#4CAF50' if x >= 0 else '#F44336' for x in perf_df['52W Change']]
                        
                        # Create horizontal bars
                        bars = ax.barh(perf_df['Peer'], perf_df['52W Change'], color=colors)
                        
                        # Add zero line
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        
                        # Format the plot
                        ax.set_title("52-Week Price Performance", fontsize=12)
                        ax.set_xlabel("Price Change (%)", fontsize=10)
                        
                        # Format x-axis as percentage
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x*100)))
                        
                        # Add data labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(
                                width + 0.01 if width >= 0 else width - 0.03,
                                bar.get_y() + bar.get_height()/2.,
                                f'{width*100:.1f}%',
                                ha='left' if width >= 0 else 'right', 
                                va='center', 
                                fontweight='bold',
                                fontsize=8,
                                color='black'
                            )
                        
                        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                        
                        # Add simplified performance interpretation
                        main_perf = peer_data[ticker].get('52W Change', 0)
                        peer_avg_perf = perf_df['52W Change'].mean()
                        
                        if main_perf != 0:
                            perf_diff = main_perf - peer_avg_perf
                            perf_status = "outperformed" if perf_diff > 0 else "underperformed"
                            
                            st.markdown(f"**{ticker}** has **{perf_status}** peers by **{abs(perf_diff*100):.1f}%** over the past year.")
                    else:
                        st.info("Insufficient performance data.")
                
                # 3. Dividend Yield Comparison in bottom-left (row2_col1)
                with row2_col1:
                    st.markdown("#### Dividend Yield Comparison")
                    
                    # Prepare data for plotting
                    divYield_data = {'Peer': [], 'Dividend Yield': []}
                    for peer, data in peer_data.items():
                        if data.get('Dividend Yield', 0) > 0:
                            divYield_data['Peer'].append(peer)
                            divYield_data['Dividend Yield'].append(data.get('Dividend Yield', 0))
                    
                    divYield_df = pd.DataFrame(divYield_data)
                    
                    if not divYield_df.empty and len(divYield_df) > 1:
                        # Create a horizontal bar chart with smaller size
                        fig, ax = plt.subplots(figsize=(5, 4))
                        
                        # Sort by dividend yield
                        divYield_df = divYield_df.sort_values(by='Dividend Yield')
                        
                        # Create horizontal bars with consistent color
                        bars = ax.barh(divYield_df['Peer'], divYield_df['Dividend Yield'], color='#4CAF50')
                        
                        # Format the plot
                        ax.set_title("Dividend Yield Comparison", fontsize=12)
                        ax.set_xlabel("Dividend Yield", fontsize=10)
                        
                        # Format x-axis as percentage
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2f}%'.format(x*100)))
                        
                        # Add data labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(
                                width + 0.0005,
                                bar.get_y() + bar.get_height()/2.,
                                f'{width*100:.2f}%',
                                ha='left', va='center', fontweight='bold', fontsize=8
                            )
                        
                        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
                        plt.tight_layout()
                        
                        # Display the plot
                        st.pyplot(fig)
                        
                        # Add simplified dividend yield interpretation
                        main_yield = peer_data[ticker].get('Dividend Yield', 0)
                        peer_avg_yield = divYield_df['Dividend Yield'].mean()
                        
                        if main_yield > 0:
                            yield_diff = main_yield - peer_avg_yield
                            yield_status = "higher than" if yield_diff > 0 else "lower than"
                            
                            st.markdown(f"**{ticker}**'s yield (**{main_yield*100:.2f}%**) is {yield_status} peer avg (**{peer_avg_yield*100:.2f}%**).")
                    elif len(divYield_df) == 1 and ticker in divYield_df['Peer'].values:
                        yield_val = divYield_df[divYield_df['Peer'] == ticker]['Dividend Yield'].iloc[0]
                        st.info(f"{ticker} has a {yield_val*100:.2f}% yield, but no peers with dividends found.")
                    else:
                        st.info("No dividend data available.")
                
                # 4. Peer Comparison Summary in bottom-right (row2_col2)
            with row2_col2:
                st.markdown("#### Peer Comparison")
                
                # Calculate overall rankings as before
                ranking_metrics = {
                    'Valuation': ['P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'EV/EBITDA'],
                    'Profitability': ['Gross Margin', 'Operating Margin', 'Profit Margin', 'ROE', 'ROA'],
                    'Growth': ['Revenue Growth', 'Earnings Growth'],
                    'Stability': ['Beta', 'Dividend Yield']
                }
                
                # Function to determine if lower is better for a metric
                def lower_is_better(metric):
                    return metric in ['P/E Ratio', 'P/S Ratio', 'P/B Ratio', 'EV/EBITDA', 'Beta']
                
                # Calculate scores for each company in each category
                scores = {peer: {'Valuation': 0, 'Profitability': 0, 'Growth': 0, 'Stability': 0, 'Overall': 0} for peer in peer_data.keys()}
                valid_counts = {peer: {'Valuation': 0, 'Profitability': 0, 'Growth': 0, 'Stability': 0, 'Overall': 0} for peer in peer_data.keys()}
                
                for category, metrics_list in ranking_metrics.items():
                    for metric in metrics_list:
                        # Get all valid values for this metric
                        valid_values = [(peer, data.get(metric, 0)) for peer, data in peer_data.items() if data.get(metric, 0) != 0 and data.get(metric, 0) != 'N/A']
                        
                        if valid_values:
                            # Sort based on whether lower or higher is better
                            sorted_peers = sorted(valid_values, key=lambda x: x[1], reverse=not lower_is_better(metric))
                            
                            # Assign scores (higher score is better)
                            for i, (peer, _) in enumerate(sorted_peers):
                                scores[peer][category] += len(sorted_peers) - i
                                valid_counts[peer][category] += 1
                                scores[peer]['Overall'] += len(sorted_peers) - i
                                valid_counts[peer]['Overall'] += 1
                
                # Normalize scores by dividing by the number of valid metrics
                for peer in scores:
                    for category in scores[peer]:
                        if valid_counts[peer][category] > 0:
                            scores[peer][category] = scores[peer][category] / valid_counts[peer][category]
                        else:
                            scores[peer][category] = 0
                
                # Create a DataFrame for visualization
                summary_data = []
                for peer in scores:
                    row = {
                        'Peer': peer,
                        'Valuation': scores[peer]['Valuation'],
                        'Profitability': scores[peer]['Profitability'],
                        'Growth': scores[peer]['Growth'],
                        'Stability': scores[peer]['Stability'],
                        'Overall': scores[peer]['Overall']
                    }
                    summary_data.append(row)
                
                summary_df = pd.DataFrame(summary_data)
                
                # Create a radar chart showing all individual peers
                if len(summary_df) > 1:
                    # Categories for the radar chart
                    categories = ['Valuation', 'Profitability', 'Growth', 'Stability']
                    
                    # Create radar chart with smaller size
                    fig = plt.figure(figsize=(5, 4))
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Number of categories
                    N = len(categories)
                    
                    # Angle of each axis
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]  # Close the loop
                    
                    # Color map for different peers
                    colors = plt.cm.tab10(np.linspace(0, 1, len(summary_df)))
                    
                    # Maximum 5 peers to avoid crowding the chart
                    max_peers = min(5, len(summary_df))
                    
                    # Ensure the main ticker is included
                    included_peers = [ticker]
                    
                    # Add top N-1 peers by overall score (excluding the main ticker)
                    other_peers = summary_df[summary_df['Peer'] != ticker].sort_values(by='Overall', ascending=False)
                    for peer in other_peers['Peer'][:max_peers-1]:
                        included_peers.append(peer)
                    
                    # Plot each peer on the radar chart
                    for i, peer in enumerate(included_peers):
                        peer_data = summary_df[summary_df['Peer'] == peer]
                        if not peer_data.empty:
                            # Get values
                            values = [peer_data.iloc[0][cat] for cat in categories]
                            values += values[:1]  # Close the loop
                            
                            # Highlight the main ticker with a thicker line and different style
                            if peer == ticker:
                                ax.plot(angles, values, color='red', linewidth=2.5, label=peer)
                                ax.fill(angles, values, color='red', alpha=0.1)
                            else:
                                ax.plot(angles, values, color=colors[i], linewidth=1.5, label=peer)
                                ax.fill(angles, values, color=colors[i], alpha=0.05)
                    
                    # Set labels
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories, fontsize=8)
                    
                    # Add legend - handle potential overcrowding by adjusting font size
                    if max_peers <= 3:
                        legend_font = 8
                    else:
                        legend_font = 7
                    
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=legend_font)
                    plt.title(f'{ticker} vs. Peers', size=12, y=1.05)
                    
                    # Display the chart
                    st.pyplot(fig)
                    
                    # Add simplified strengths/weaknesses based on peer comparison
                    main_ticker_data = summary_df[summary_df['Peer'] == ticker].iloc[0] if ticker in summary_df['Peer'].values else None
                    
                    if main_ticker_data is not None:
                        # Calculate the peer average excluding the main ticker
                        peer_avg = {}
                        for cat in categories:
                            cat_values = [summary_df[summary_df['Peer'] == p].iloc[0][cat] for p in included_peers if p != ticker]
                            peer_avg[cat] = sum(cat_values) / len(cat_values) if cat_values else 0
                        
                        # Identify strengths and weaknesses
                        strengths = []
                        weaknesses = []
                        
                        for cat in categories:
                            if main_ticker_data[cat] > peer_avg[cat] * 1.1:
                                strengths.append(cat)
                            elif main_ticker_data[cat] < peer_avg[cat] * 0.9:
                                weaknesses.append(cat)
                        
                        # Display compact strengths/weaknesses
                        if strengths:
                            st.markdown(f"**Strengths:** {', '.join(strengths)}")
                        if weaknesses:
                            st.markdown(f"**Areas for improvement:** {', '.join(weaknesses)}")
                else:
                    st.info("Insufficient peer data for comparison.")
                
                # Third row with full-width rankings table
                row3_col = st.container()
                with row3_col:
                    st.markdown("### Overall Peer Rankings")
                    
                    # Create rankings DataFrame
                    rankings_df = summary_df[['Peer', 'Overall', 'Valuation', 'Profitability', 'Growth', 'Stability']].copy()
                    
                    # Convert scores to ranks (1 is best)
                    for col in ['Overall', 'Valuation', 'Profitability', 'Growth', 'Stability']:
                        rankings_df[f'{col} Rank'] = rankings_df[col].rank(ascending=False).astype(int)
                    
                    # Keep only the rank columns plus Peer
                    display_ranks = rankings_df[['Peer', 'Overall Rank', 'Valuation Rank', 'Profitability Rank', 'Growth Rank', 'Stability Rank']].copy()
                    
                    # Format for display
                    for peer in display_ranks['Peer']:
                        # Highlight the current ticker
                        if peer == ticker:
                            display_ranks.loc[display_ranks['Peer'] == peer, 'Peer'] = f"➤ {peer}"
                    
                    # Sort by Overall Rank
                    display_ranks = display_ranks.sort_values(by='Overall Rank')
                    
                    # Display rankings
                    st.dataframe(display_ranks.set_index('Peer'), use_container_width=True)

# ----------- NEWS & INSIGHTS TAB REDESIGN ----------- #
with tabs[3]:  # Positioned as the fifth tab (index 4)
    st.subheader(f"📰 {ticker} News & Social Insights")
    
    # Initialize session state variables for caching
    if 'news_cache' not in st.session_state:
        st.session_state.news_cache = {}
    if 'news_cache_timestamp' not in st.session_state:
        st.session_state.news_cache_timestamp = {}
    if 'social_cache' not in st.session_state:
        st.session_state.social_cache = {}
    if 'social_cache_timestamp' not in st.session_state:
        st.session_state.social_cache_timestamp = {}
    if 'sentiment_analysis_cache' not in st.session_state:
        st.session_state.sentiment_analysis_cache = {}
    if 'news_insights_cache' not in st.session_state:
        st.session_state.news_insights_cache = {}

    # Function to fetch news using Alpha Vantage News API with caching
    def get_stock_news(ticker, limit=10):
        """Fetch news articles for a given ticker using Marketaux API with efficient caching"""
        cache_key = f"{ticker}_{limit}"
        
        # Check if we have cached data that's not expired (less than 6 hours old to conserve API calls)
        current_time = datetime.now()
        if (cache_key in st.session_state.news_cache and cache_key in st.session_state.news_cache_timestamp and
            (current_time - st.session_state.news_cache_timestamp[cache_key]).total_seconds() < 21600):  # 6 hours
            return st.session_state.news_cache[cache_key]
        
        try:
            # Marketaux API key
            api_key = os.getenv('MARKETAUX_API_KEY')
            
            # Check if API key is available
            if not api_key:
                st.warning("Marketaux API key not found. News data may be limited.")
                return []  # Or return a fallback list of news items
            
            # Marketaux API endpoint - using efficient parameters to minimize request count
            url = f"https://api.marketaux.com/v1/news/all"
            
            # Parameters for the API request
            params = {
                "symbols": ticker,
                "filter_entities": "true",    # Get entity data for better analysis
                "language": "en",             # English articles only
                "api_token": api_key,
                "limit": str(min(limit, 100)) # Ensure we don't request more than needed
            }
            
            # Make the request
            response = requests.get(url, params=params)
            
            # Parse the response
            data = response.json()
            
            # Process the news data
            news_items = []
            
            if "data" in data and isinstance(data["data"], list):
                for item in data["data"][:limit]:
                    # Extract relevant information
                    title = item.get("title", "")
                    description = item.get("description", "")
                    url = item.get("url", "")
                    source = item.get("source", "")
                    published_at = item.get("published_at", "")
                    
                    # Format the time to match expected format
                    time_published = ""
                    if published_at:
                        try:
                            # Convert ISO 8601 format to our format
                            dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            time_published = dt.strftime("%Y%m%dT%H%M%S")
                        except:
                            time_published = datetime.now().strftime("%Y%m%dT%H%M%S")
                    
                    # Create news item dictionary
                    news_item = {
                        "title": title,
                        "summary": description,
                        "url": url,
                        "source": source if source else "Marketaux",
                        "time_published": time_published
                    }
                    
                    # Add sentiment if available
                    if "entities" in item and item["entities"]:
                        for entity in item["entities"]:
                            if entity.get("symbol") == ticker:
                                sentiment = entity.get("sentiment_score", 0)
                                # Add pre-calculated sentiment for efficiency
                                news_item["sentiment_analysis"] = {
                                    "sentiment_value": sentiment,
                                    "sentiment_label": "positive" if sentiment > 0.1 else ("negative" if sentiment < -0.1 else "neutral"),
                                    "sentiment_scores": {
                                        "positive": max(0, sentiment),
                                        "negative": max(0, -sentiment),
                                        "neutral": 1 - abs(sentiment)
                                    }
                                }
                                break
                    
                    news_items.append(news_item)
                
                # Store in cache with longer expiration to conserve API calls
                st.session_state.news_cache[cache_key] = news_items
                st.session_state.news_cache_timestamp[cache_key] = current_time
                return news_items
            else:
                # Handle error cases from the API
                if "error" in data:
                    st.warning(f"Marketaux API error: {data['error']['message']}")
                else:
                    st.warning(f"No news data returned from Marketaux for {ticker}")
                
                # Return empty list to indicate no news found
                return []
                
        except Exception as e:
            st.error(f"Error fetching news from Marketaux: {str(e)}")
            return []
    
    # Function to analyze sentiment using FinBERT
    def analyze_finbert_sentiment(texts):
        """
        Analyze sentiment of texts using FinBERT.
        Returns a dictionary with sentiment scores for each text.
        Uses caching to improve performance.
        """
        # Create a unique key for the texts
        cache_key = hashlib.md5(str(texts).encode()).hexdigest()
        
        # Check if we have cached results
        if cache_key in st.session_state.sentiment_analysis_cache:
            return st.session_state.sentiment_analysis_cache[cache_key]
        
        try:
            # Import transformers only when needed
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Initialize FinBERT model and tokenizer (first time only)
            if 'finbert_model' not in st.session_state:
                # Load FinBERT model and tokenizer
                st.session_state.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
                st.session_state.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            
            tokenizer = st.session_state.finbert_tokenizer
            model = st.session_state.finbert_model
            
            results = []
            
            # Process in batches to avoid memory issues
            batch_size = 8
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and prepare for model
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.softmax(outputs.logits, dim=1)
                
                # Process each text in the batch
                for j, text in enumerate(batch_texts):
                    # Get sentiment scores (Positive, Negative, Neutral)
                    scores = predictions[j].tolist()
                    
                    # FinBERT classes: [Positive, Negative, Neutral]
                    sentiment_scores = {
                        "positive": scores[0],
                        "negative": scores[1],
                        "neutral": scores[2]
                    }
                    
                    # Get the sentiment with the highest score
                    sentiment_label = max(sentiment_scores, key=sentiment_scores.get)
                    
                    # Calculate a single sentiment score (-1 to 1)
                    # Positive is positive, Negative is negative, Neutral is 0
                    sentiment_value = sentiment_scores["positive"] - sentiment_scores["negative"]
                    
                    results.append({
                        "text": text,
                        "sentiment_label": sentiment_label,
                        "sentiment_value": sentiment_value,
                        "sentiment_scores": sentiment_scores
                    })
            
            # Store in cache
            st.session_state.sentiment_analysis_cache[cache_key] = results
            
            return results
        except Exception as e:
            st.error(f"Error in FinBERT sentiment analysis: {str(e)}")
            
            # Fallback to TextBlob for sentiment analysis
            from textblob import TextBlob
            
            results = []
            for text in texts:
                blob = TextBlob(text)
                sentiment_value = blob.sentiment.polarity
                
                if sentiment_value > 0.1:
                    sentiment_label = "positive"
                elif sentiment_value < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                sentiment_scores = {
                    "positive": max(0, sentiment_value),
                    "negative": max(0, -sentiment_value),
                    "neutral": 1 - abs(sentiment_value)
                }
                
                results.append({
                    "text": text,
                    "sentiment_label": sentiment_label,
                    "sentiment_value": sentiment_value,
                    "sentiment_scores": sentiment_scores
                })
            
            # Store in cache
            st.session_state.sentiment_analysis_cache[cache_key] = results
            
            return results
    
    # Function to fetch social media data from Reddit
    def get_social_media_data(ticker):
        """
        Fetch social media data from Reddit using PRAW.
        Uses caching to reduce API calls.
        """
        cache_key = f"{ticker}_social"
        
        # Check if we have cached data that's not expired (less than 3 hours old)
        current_time = datetime.now()
        if (cache_key in st.session_state.social_cache and cache_key in st.session_state.social_cache_timestamp and
            (current_time - st.session_state.social_cache_timestamp[cache_key]).total_seconds() < 10800):
            return st.session_state.social_cache[cache_key]
        
        try:
            import praw
            
            # Initialize social media data structure
            social_data = {
                "platforms": {
                    "reddit": {"mentions": 0, "posts": [], "sentiment_scores": {"positive": 0, "negative": 0, "neutral": 0}}
                },
                "trending_terms": [],
                "total_mentions": 0
            }
            
            # Search terms tailored for stock discussion
            search_terms = [
                f"${ticker}",                # Stock symbol with $ prefix (common in financial discussions)
                ticker,                      # Plain stock symbol
                f"{ticker} stock",           # Stock symbol with "stock"
                f"{ticker} investor",        # Investor discussions
                f"{ticker} earnings",        # Earnings-related discussions
            ]
            
            # ---------- Reddit API (using PRAW) ----------
            try:
                with st.spinner(f"Fetching Reddit data for {ticker}..."):
                    # Check if Reddit credentials exist in session state
                    if 'reddit_api' not in st.session_state:
                        # Initialize Reddit API client with your credentials
                        reddit = praw.Reddit(
                            client_id="ePUV66LIS-2LHVcOy7yryg",
                            client_secret="jHI1dXl2yPYsUqdE-G3xZfov0AQcUA",
                            user_agent="Noor Ahmad Raza/1.0 by TryEasy6525"
                        )
                        st.session_state.reddit_api = reddit
                    else:
                        reddit = st.session_state.reddit_api
                    
                    # Subreddits to search
                    subreddits = ['stocks', 'investing', 'wallstreetbets', 'finance', 'SecurityAnalysis']
                    
                    # Try to add a ticker-specific subreddit if it exists
                    try:
                        ticker_subreddit = reddit.subreddit(ticker)
                        # Check if the subreddit exists and has posts
                        ticker_subreddit.hot(limit=1)  # This will raise exception if subreddit doesn't exist
                        subreddits.append(ticker)
                    except:
                        pass  # Ticker subreddit doesn't exist or is private
                    
                    reddit_posts = []
                    reddit_count = 0
                    
                    # Get company name for better searches
                    company_name = ""
                    try:
                        # Try to get company name from yfinance
                        stock = yf.Ticker(ticker)
                        company_info = stock.info
                        if 'shortName' in company_info:
                            company_name = company_info['shortName']
                            # Add to search terms
                            if company_name:
                                search_terms.append(company_name)
                    except:
                        pass
                    
                    # Search each subreddit for each term
                    for subreddit_name in subreddits:
                        subreddit = reddit.subreddit(subreddit_name)
                        
                        for term in search_terms:
                            try:
                                # Get posts from the past week
                                posts = subreddit.search(
                                    term, 
                                    sort='relevance',
                                    time_filter='week',
                                    limit=10  # Limit per search term/subreddit combination
                                )
                                
                                for post in posts:
                                    if hasattr(post, 'selftext'):
                                        reddit_posts.append({
                                            "title": post.title,
                                            "content": post.selftext,
                                            "subreddit": post.subreddit.display_name,
                                            "date": datetime.fromtimestamp(post.created_utc),
                                            "url": f"https://www.reddit.com{post.permalink}"
                                        })
                                        reddit_count += 1
                            except Exception as e:
                                st.warning(f"Error searching r/{subreddit_name} for {term}: {str(e)}")
                                continue
                    
                    # Remove duplicate posts
                    unique_posts = []
                    seen_titles = set()
                    for post in reddit_posts:
                        title = post["title"]
                        if title not in seen_titles:
                            seen_titles.add(title)
                            unique_posts.append(post)
                    
                    reddit_posts = unique_posts
                    
                    # Get text for sentiment analysis (combine title and content)
                    reddit_texts = [(post["title"] + " " + post["content"]) for post in reddit_posts]
                    
                    # Analyze sentiment with FinBERT or TextBlob
                    if reddit_texts:
                        try:
                            reddit_sentiment_results = analyze_finbert_sentiment(reddit_texts)
                            
                            # Calculate aggregated sentiment scores
                            reddit_sentiment = {"positive": 0, "negative": 0, "neutral": 0}
                            for result in reddit_sentiment_results:
                                reddit_sentiment["positive"] += result["sentiment_scores"]["positive"]
                                reddit_sentiment["negative"] += result["sentiment_scores"]["negative"]
                                reddit_sentiment["neutral"] += result["sentiment_scores"]["neutral"]
                            
                            # Normalize
                            total = len(reddit_sentiment_results)
                            if total > 0:
                                reddit_sentiment["positive"] /= total
                                reddit_sentiment["negative"] /= total
                                reddit_sentiment["neutral"] /= total
                            
                            social_data["platforms"]["reddit"]["sentiment_scores"] = reddit_sentiment
                        except Exception as e:
                            st.error(f"Error analyzing Reddit sentiment: {str(e)}")
                            # Fallback to default distribution
                            social_data["platforms"]["reddit"]["sentiment_scores"] = {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
                    
                    # Select top posts for display
                    selected_posts = []
                    if reddit_posts:
                        # Sort by date (newest first)
                        sorted_posts = sorted(reddit_posts, key=lambda x: x["date"], reverse=True)
                        selected_posts = [f"{post['title']} (r/{post['subreddit']})" for post in sorted_posts[:min(5, len(sorted_posts))]]
                    
                    # Update the social data
                    social_data["platforms"]["reddit"]["mentions"] = reddit_count
                    social_data["platforms"]["reddit"]["posts"] = selected_posts
                    social_data["platforms"]["reddit"]["full_posts"] = reddit_posts[:10]  # Store full post details for display
            except ImportError:
                st.warning("PRAW not installed. Run 'pip install praw' to enable Reddit data collection.")
                # Fallback to simulated data for Reddit
                social_data["platforms"]["reddit"]["mentions"] = random.randint(30, 400)
                
                example_reddit_posts = [
                    f"DD on {ticker}: Why I'm bullish for the next quarter",
                    f"Anyone else concerned about {ticker}'s debt levels?",
                    f"{ticker} Technical Analysis: Support and resistance levels for this week",
                    f"Institutional buying of {ticker} has increased this month",
                    f"Thoughts on {ticker}'s upcoming product release?"
                ]
                
                random.shuffle(example_reddit_posts)
                selected_reddit = example_reddit_posts[:min(5, len(example_reddit_posts))]
                social_data["platforms"]["reddit"]["posts"] = selected_reddit
                social_data["platforms"]["reddit"]["sentiment_scores"] = {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
                reddit_texts = []
            
            # ---------- Process Combined Data ----------
            # Get text for trending term extraction
            all_text = " ".join(reddit_texts)
            
            # If we have no real data, use simulated data for trending terms
            if not all_text:
                # Generate trending terms related to the ticker
                common_stock_terms = ["buy", "sell", "hold", "bullish", "bearish", "overvalued", "undervalued", 
                                    "earnings", "revenue", "guidance", "dividend", "split", "upgrade", "downgrade",
                                    "price target", "breakout", "resistance", "support", "CEO", "analyst", "quarter"]
                
                # Select random subset and assign random counts
                selected_terms = random.sample(common_stock_terms, min(10, len(common_stock_terms)))
                trending_terms = [{"term": term, "count": random.randint(5, 100)} for term in selected_terms]
                
                # Sort by count (highest first)
                trending_terms.sort(key=lambda x: x["count"], reverse=True)
                social_data["trending_terms"] = trending_terms
            else:
                # More comprehensive set of financial terms to track
                common_stock_terms = [
                    # Trading actions
                    "buy", "sell", "hold", "long", "short", "position", "calls", "puts", "options",
                    
                    # Sentiment
                    "bullish", "bearish", "overvalued", "undervalued", "mooning", "crashing",
                    
                    # Financials
                    "earnings", "revenue", "profit", "margin", "guidance", "EPS", "PE", "dividend", "yield",
                    
                    # Corporate events
                    "split", "buyback", "acquisition", "merger", "IPO", "offering", "debt",
                    
                    # Analysis
                    "upgrade", "downgrade", "price target", "breakout", "resistance", "support",
                    
                    # People & institutions
                    "CEO", "analyst", "institutional", "hedge", "shorts", "retail", "insider",
                    
                    # Time periods
                    "quarter", "fiscal", "forecast", "outlook"
                ]
                
                # Add ticker-specific terms
                company_name = ""
                try:
                    # Try to get company name from yfinance
                    stock = yf.Ticker(ticker)
                    company_info = stock.info
                    if 'shortName' in company_info:
                        company_name = company_info['shortName']
                        # Add company name and variations to search terms
                        name_terms = [company_name]
                        if " " in company_name:
                            name_terms.append(company_name.split(" ")[0])  # First word of company name
                        
                        common_stock_terms.extend(name_terms)
                except:
                    pass
                    
                # Add regex import if not already at the top of your file
                import re
                    
                trending_terms = []
                # Extract trending terms with more sophisticated counting
                for term in common_stock_terms:
                    # Case-insensitive word boundary search
                    count = len(re.findall(r'\b' + re.escape(term) + r'\b', all_text.lower()))
                    if count > 0:
                        trending_terms.append({"term": term, "count": count})
                    
                # Sort by count (highest first)
                trending_terms.sort(key=lambda x: x["count"], reverse=True)
                
                # Keep top 10
                social_data["trending_terms"] = trending_terms[:10]
            
            # Calculate total mentions
            social_data["total_mentions"] = sum(platform["mentions"] for platform in social_data["platforms"].values())
            
            # Store in cache
            st.session_state.social_cache[cache_key] = social_data
            st.session_state.social_cache_timestamp[cache_key] = current_time
            
            return social_data
        except Exception as e:
            st.error(f"Error fetching social media data: {str(e)}")
            # Fallback to completely simulated data
            social_data = {
                "platforms": {
                    "reddit": {
                        "mentions": random.randint(30, 400),
                        "posts": [
                            f"DD on {ticker}: Why I'm bullish for the next quarter",
                            f"Anyone else concerned about {ticker}'s debt levels?",
                            f"{ticker} Technical Analysis: Support and resistance levels for this week"
                        ],
                        "sentiment_scores": {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
                    }
                },
                "trending_terms": [
                    {"term": "buy", "count": random.randint(50, 100)},
                    {"term": "earnings", "count": random.randint(40, 90)},
                    {"term": "bullish", "count": random.randint(30, 80)},
                    {"term": "price target", "count": random.randint(20, 70)},
                    {"term": "upgrade", "count": random.randint(10, 60)},
                    {"term": "revenue", "count": random.randint(5, 50)},
                    {"term": "growth", "count": random.randint(5, 40)},
                    {"term": "dividend", "count": random.randint(5, 30)},
                    {"term": "CEO", "count": random.randint(5, 20)},
                    {"term": "forecast", "count": random.randint(5, 10)}
                ],
                "total_mentions": random.randint(80, 900)
            }
            
            return social_data
    
    # Function to generate news insights using Cohere API
    def generate_news_insights(ticker, news_items, reddit_posts=None):
        """Generate insights from news and Reddit data using Cohere API"""
        api_key = "TvrgUHF3GKzAB5sYBHK7UkHApcr2VZ0nJnBkNATD"  # Replace with your actual Cohere API key
        api_url = "https://api.cohere.ai/v1/generate"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Extract news titles and summaries
        news_digest = ""
        for i, news in enumerate(news_items[:5]):  # Process top 5 news items
            if i >= 5:
                break
                
            title = news.get("title", "")
            summary = news.get("summary", "")
            source = news.get("source", "")
            date = news.get("time_published", "")
            
            news_digest += f"Article {i+1}:\n"
            news_digest += f"Title: {title}\n"
            news_digest += f"Summary: {summary}\n"
            news_digest += f"Source: {source}\n"
            news_digest += f"Date: {date}\n\n"
        
        # Add Reddit posts if available
        reddit_digest = ""
        if reddit_posts and len(reddit_posts) > 0:
            reddit_posts = reddit_posts[:3]  # Process top 3 Reddit posts
            
            for i, post in enumerate(reddit_posts):
                title = post.get("title", "")
                subreddit = post.get("subreddit", "")
                
                reddit_digest += f"Reddit Post {i+1}:\n"
                reddit_digest += f"Title: {title}\n"
                reddit_digest += f"Subreddit: r/{subreddit}\n\n"
        
        # Create a prompt for Cohere
        prompt = f"""As a financial analyst, analyze the following information for {ticker} and provide insights:

        RECENT NEWS:
        {news_digest}
        
        REDDIT DISCUSSIONS:
        {reddit_digest}
        
        Based on this information, provide:
        1. Key Themes: Summarize the main themes across these sources (2-3 sentences)
        2. Market Sentiment: Describe the current sentiment on {ticker} (bullish, bearish, mixed) and why (2 sentences)
        3. Impact Factors: Identify potential factors that could impact {ticker}'s stock price (2-3 bullet points)
        4. Investment Considerations: Suggest what metrics or developments investors should monitor (1-2 bullet points)
        
        Format your response under clear headings for each section and be concise. Focus only on what can be directly inferred from the provided information.
        """
        
        payload = {
            "model": "command",
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            insights = result.get("generations", [{}])[0].get("text", "").strip()
            
            return insights
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return "Unable to generate insights at this time. Please try again later."
    
    # ===== DASHBOARD TOP SECTION =====
    # News & Social settings in a clean interface
    st.markdown("### 📊 Dashboard Settings")
    
    settings_col1, settings_col2, settings_col3 = st.columns([1, 1, 1])
    
    with settings_col1:
        news_limit_options = [5, 10, 20, 30, 50]
        news_limit = st.selectbox("Number of news articles:", news_limit_options, index=1)  # Default to 10
    
    with settings_col2:
        date_filter = st.selectbox("Time period:", ["Past 24 hours", "Past 3 days", "Past week", "Past month"], index=2)
    
    with settings_col3:
        refresh_button = st.button("🔄 Refresh Data")
        if refresh_button:
            # Clear caches to force refresh
            st.session_state.news_cache = {}
            st.session_state.news_cache_timestamp = {}
            st.session_state.social_cache = {}
            st.session_state.social_cache_timestamp = {}
            st.session_state.sentiment_analysis_cache = {}
            st.experimental_rerun()
    
    # Horizontal divider
    st.markdown("---")
    
    # ===== DATA FETCHING & ANALYSIS =====
    # Fetch news and social data with the selected filters
    with st.spinner("Fetching news, social discussions, and performing analysis..."):
        news_items = get_stock_news(ticker, limit=news_limit)
        news_items = news_items[:news_limit]
        
        # Perform sentiment analysis on news items
        if news_items:
            # Prepare texts for sentiment analysis
            news_texts = []
            for news in news_items:
                title = news.get("title", "")
                summary = news.get("summary", "")
                text = f"{title} {summary}"
                news_texts.append(text)
            
            # Analyze sentiment with FinBERT
            news_sentiment_results = analyze_finbert_sentiment(news_texts)
            
            # Add sentiment results to news items
            for i, result in enumerate(news_sentiment_results):
                if i < len(news_items):
                    news_items[i]["sentiment_analysis"] = result
        
        # Fetch social media data
        social_data = get_social_media_data(ticker)
        
        # Add debugging statements here
        if social_data and "platforms" in social_data and "reddit" in social_data["platforms"]:
            st.write("Reddit platform exists in social data")
            if "full_posts" in social_data["platforms"]["reddit"]:
                st.write(f"Found {len(social_data['platforms']['reddit']['full_posts'])} full Reddit posts")
            else:
                st.write("'full_posts' key not found in reddit data")

        # Get Reddit posts for display
        reddit_posts = []
        if social_data and "platforms" in social_data and "reddit" in social_data["platforms"]:
            if "full_posts" in social_data["platforms"]["reddit"]:
                reddit_posts = social_data["platforms"]["reddit"]["full_posts"]
    
    # ===== SENTIMENT OVERVIEW SECTION =====
    # Create a unified sentiment analysis section that combines news and social media
    
    # Calculate combined sentiment scores
    combined_sentiment = {
        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "score": 0,
        "count": 0,
        "sources": {
            "news": {"positive": 0, "negative": 0, "neutral": 0, "score": 0, "count": 0},
            "reddit": {"positive": 0, "negative": 0, "neutral": 0, "score": 0, "count": 0}
        }
    }
    
    # Add news sentiment
    if news_items:
        news_sentiment = {"positive": 0, "negative": 0, "neutral": 0, "count": 0, "score": 0}
        for news in news_items:
            if "sentiment_analysis" in news:
                sentiment = news["sentiment_analysis"]
                news_sentiment["positive"] += sentiment["sentiment_scores"]["positive"]
                news_sentiment["negative"] += sentiment["sentiment_scores"]["negative"]
                news_sentiment["neutral"] += sentiment["sentiment_scores"]["neutral"]
                news_sentiment["score"] += sentiment["sentiment_value"]
                news_sentiment["count"] += 1
        
        if news_sentiment["count"] > 0:
            news_sentiment["positive"] /= news_sentiment["count"]
            news_sentiment["negative"] /= news_sentiment["count"]
            news_sentiment["neutral"] /= news_sentiment["count"]
            news_sentiment["score"] /= news_sentiment["count"]
            
            combined_sentiment["sources"]["news"] = news_sentiment
            
            # Add to combined total
            combined_sentiment["positive"] += news_sentiment["positive"] * news_sentiment["count"]
            combined_sentiment["negative"] += news_sentiment["negative"] * news_sentiment["count"]
            combined_sentiment["neutral"] += news_sentiment["neutral"] * news_sentiment["count"]
            combined_sentiment["score"] += news_sentiment["score"] * news_sentiment["count"]
            combined_sentiment["count"] += news_sentiment["count"]
    
    # Add Reddit sentiment
    if social_data and "platforms" in social_data and "reddit" in social_data["platforms"]:
        reddit_data = social_data["platforms"]["reddit"]
        reddit_mentions = reddit_data["mentions"]
        reddit_sentiment = reddit_data["sentiment_scores"]
        
        if reddit_mentions > 0:
            # Store the Reddit sentiment in sources
            combined_sentiment["sources"]["reddit"] = {
                "positive": reddit_sentiment["positive"],
                "negative": reddit_sentiment["negative"],
                "neutral": reddit_sentiment["neutral"],
                "score": reddit_sentiment["positive"] - reddit_sentiment["negative"],
                "count": reddit_mentions
            }
            
            # Add to combined total with weight from mentions
            combined_sentiment["positive"] += reddit_sentiment["positive"] * reddit_mentions
            combined_sentiment["negative"] += reddit_sentiment["negative"] * reddit_mentions
            combined_sentiment["neutral"] += reddit_sentiment["neutral"] * reddit_mentions
            combined_sentiment["score"] += (reddit_sentiment["positive"] - reddit_sentiment["negative"]) * reddit_mentions
            combined_sentiment["count"] += reddit_mentions
    
    # Normalize combined sentiment
    if combined_sentiment["count"] > 0:
        combined_sentiment["positive"] /= combined_sentiment["count"]
        combined_sentiment["negative"] /= combined_sentiment["count"]
        combined_sentiment["neutral"] /= combined_sentiment["count"]
        combined_sentiment["score"] /= combined_sentiment["count"]
    
    # ===== STORY-TELLING DASHBOARD LAYOUT =====
    
    # 1. HEADLINE SECTION: Big Sentiment Gauge + Key Stats
    st.markdown("## Market Pulse: Sentiment Overview")
    
    # Display combined sentiment gauge
    if combined_sentiment["count"] > 0:
        # Determine overall sentiment label and color
        score = combined_sentiment["score"]
        
        if score > 0.3:
            overall_label = "Very Positive"
            overall_color = "#00C853"  # Dark green
        elif score > 0.1:
            overall_label = "Positive"
            overall_color = "#4CAF50"  # Green
        elif score > -0.1:
            overall_label = "Neutral"
            overall_color = "#FFC107"  # Yellow
        elif score > -0.3:
            overall_label = "Negative"
            overall_color = "#FF9800"  # Orange
        else:
            overall_label = "Very Negative"
            overall_color = "#F44336"  # Red
        
        # Calculate percentages for visualization
        positive_pct = combined_sentiment["positive"] * 100
        neutral_pct = combined_sentiment["neutral"] * 100
        negative_pct = combined_sentiment["negative"] * 100
        
        # Make sure they sum to 100%
        total = positive_pct + neutral_pct + negative_pct
        if total > 0:
            positive_pct = (positive_pct / total) * 100
            neutral_pct = (neutral_pct / total) * 100
            negative_pct = (negative_pct / total) * 100
        
        # Top row with big sentiment gauge and key stats
        top_col1, top_col2 = st.columns([2, 1])
        
        with top_col1:
            # Create the visualization
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 28px; font-weight: bold; margin-bottom: 10px;">
                    <span style="color: {overall_color};">{overall_label}</span>
                </div>
                <div style="position: relative; width: 100%; height: 60px; background-color: #f5f5f5; border-radius: 30px; overflow: hidden; margin-bottom: 10px;">
                    <div style="display: flex; height: 100%;">
                        <div style="width: {positive_pct}%; background-color: #4CAF50;"></div>
                        <div style="width: {neutral_pct}%; background-color: #FFC107;"></div>
                        <div style="width: {negative_pct}%; background-color: #F44336;"></div>
                    </div>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: black; font-weight: bold; font-size: 18px;">
                        Sentiment Score: {score:.2f}
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; width: 100%;">
                    <span style="color: #F44336;">Very Negative</span>
                    <span style="color: #FF9800;">Negative</span>
                    <span style="color: #FFC107;">Neutral</span>
                    <span style="color: #4CAF50;">Positive</span>
                    <span style="color: #00C853;">Very Positive</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with top_col2:
            # Key stats
            st.markdown("### Key Metrics")
            
            # Create a metric card with total analyzed items
            total_news = len(news_items) if news_items else 0
            total_reddit = social_data["platforms"]["reddit"]["mentions"] if social_data and "platforms" in social_data and "reddit" in social_data["platforms"] else 0
            
            st.metric(
                label="News Articles Analyzed", 
                value=total_news
            )
            
            st.metric(
                label="Reddit Posts Analyzed", 
                value=total_reddit
            )
            
            # Calculate sentiment trend
            sentiment_trend = None
            if news_items and any("sentiment_analysis" in news for news in news_items):
                # Get sentiment over time from news
                dated_sentiments = []
                for news in news_items:
                    time_published = news.get("time_published", "")
                    
                    try:
                        if time_published and "sentiment_analysis" in news:
                            date_obj = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                            score = news["sentiment_analysis"]["sentiment_value"]
                            dated_sentiments.append((date_obj, score))
                    except:
                        pass
                
                if dated_sentiments and len(dated_sentiments) >= 2:
                    # Sort by date
                    dated_sentiments.sort(key=lambda x: x[0])
                    
                    # Calculate simple linear trend
                    first_half = dated_sentiments[:len(dated_sentiments)//2]
                    second_half = dated_sentiments[len(dated_sentiments)//2:]
                    
                    first_avg = sum(s[1] for s in first_half) / len(first_half)
                    second_avg = sum(s[1] for s in second_half) / len(second_half)
                    
                    sentiment_trend = second_avg - first_avg
            
            if sentiment_trend is not None:
                trend_text = ""
                if sentiment_trend > 0.05:
                    trend_text = "Improving"
                    trend_delta_color = "normal"
                elif sentiment_trend < -0.05:
                    trend_text = "Declining"
                    trend_delta_color = "inverse"
                else:
                    trend_text = "Stable"
                    trend_delta_color = "off"
                
                st.metric(
                    label="Sentiment Trend", 
                    value=trend_text,
                    delta=f"{sentiment_trend:.2f}",
                    delta_color=trend_delta_color
                )
    else:
        st.info("No sentiment data available for analysis.")
    
    # 2. SOURCE BREAKDOWN: News vs Reddit sentiment comparison
    st.markdown("## Sentiment Analysis by Source")
    
    if combined_sentiment["count"] > 0:
        source_col1, source_col2 = st.columns(2)
        
        with source_col1:
            st.markdown("### News Sentiment")
            
            if "news" in combined_sentiment["sources"] and combined_sentiment["sources"]["news"]["count"] > 0:
                news_sentiment = combined_sentiment["sources"]["news"]
                
                # Create a pie chart
                fig, ax = plt.subplots(figsize=(4, 4))
                
                labels = ['Positive', 'Neutral', 'Negative']
                sizes = [news_sentiment["positive"], news_sentiment["neutral"], news_sentiment["negative"]]
                colors = ['#4CAF50', '#FFC107', '#F44336']
                
                # Only show wedges with non-zero values
                valid_labels = [label for i, label in enumerate(labels) if sizes[i] > 0]
                valid_sizes = [size for size in sizes if size > 0]
                valid_colors = [color for i, color in enumerate(colors) if sizes[i] > 0]
                
                if valid_sizes:
                    wedges, texts, autotexts = ax.pie(
                        valid_sizes, 
                        labels=valid_labels, 
                        colors=valid_colors, 
                        autopct='%1.1f%%', 
                        startangle=90,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                    
                    # Styling
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.axis('equal')
                    plt.title(f"News Sentiment Distribution\nScore: {news_sentiment['score']:.2f}", fontweight='bold')
                    
                    st.pyplot(fig)
                else:
                    st.info("No sentiment data available for news.")
                
                # Show most positive/negative headlines
                st.markdown("#### Most Positive News")
                if news_items:
                    # Get most positive headline
                    positive_news = sorted(
                        [n for n in news_items if "sentiment_analysis" in n], 
                        key=lambda x: x["sentiment_analysis"]["sentiment_value"], 
                        reverse=True
                    )
                    
                    if positive_news:
                        pos_news = positive_news[0]
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #E8F5E9; border-left: 5px solid #4CAF50; margin-bottom: 10px;">
                            <div style="font-weight: bold;">{pos_news.get('title', 'No title')}</div>
                            <div style="font-size: 0.8em; color: #555;">Source: {pos_news.get('source', 'Unknown')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("#### Most Negative News")
                if news_items:
                    # Get most negative headline
                    negative_news = sorted(
                        [n for n in news_items if "sentiment_analysis" in n], 
                        key=lambda x: x["sentiment_analysis"]["sentiment_value"]
                    )
                    
                    if negative_news:
                        neg_news = negative_news[0]
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #FFEBEE; border-left: 5px solid #F44336; margin-bottom: 10px;">
                            <div style="font-weight: bold;">{neg_news.get('title', 'No title')}</div>
                            <div style="font-size: 0.8em; color: #555;">Source: {neg_news.get('source', 'Unknown')}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No news sentiment data available.")
            
        with source_col2:
            st.markdown("### Reddit Sentiment")
            
            if "reddit" in combined_sentiment["sources"] and combined_sentiment["sources"]["reddit"]["count"] > 0:
                reddit_sentiment = combined_sentiment["sources"]["reddit"]
                
                # Create a pie chart for Reddit sentiment
                fig, ax = plt.subplots(figsize=(4, 4))
                
                labels = ['Positive', 'Neutral', 'Negative']
                sizes = [reddit_sentiment["positive"], reddit_sentiment["neutral"], reddit_sentiment["negative"]]
                colors = ['#4CAF50', '#FFC107', '#F44336']
                
                # Only show wedges with non-zero values
                valid_labels = [label for i, label in enumerate(labels) if sizes[i] > 0]
                valid_sizes = [size for size in sizes if size > 0]
                valid_colors = [color for i, color in enumerate(colors) if sizes[i] > 0]
                
                if valid_sizes:
                    wedges, texts, autotexts = ax.pie(
                        valid_sizes, 
                        labels=valid_labels, 
                        colors=valid_colors, 
                        autopct='%1.1f%%', 
                        startangle=90,
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
                    )
                    
                    # Styling
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.axis('equal')
                    plt.title(f"Reddit Sentiment Distribution\nScore: {reddit_sentiment['score']:.2f}", fontweight='bold')
                    
                    st.pyplot(fig)
                else:
                    st.info("No sentiment data available for Reddit.")
                
                # Show most discussed subreddits
                st.markdown("#### Most Active Subreddits")
                
                if reddit_posts:
                    # Count posts by subreddit
                    subreddit_counts = {}
                    for post in reddit_posts:
                        subreddit = post.get("subreddit", "Unknown")
                        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1
                    
                    # Convert to list and sort
                    subreddit_list = [(sr, count) for sr, count in subreddit_counts.items()]
                    subreddit_list.sort(key=lambda x: x[1], reverse=True)
                    
                    # Display top subreddits
                    for sr, count in subreddit_list[:3]:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; padding: 5px 10px; background-color: #E3F2FD; margin-bottom: 5px; border-radius: 3px;">
                            <span style="font-weight: bold;">r/{sr}</span>
                            <span>{count} posts</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No Reddit sentiment data available.")
    else:
        st.info("No sentiment data available by source.")
    
    # 3. TRENDING TERMS: Visual representation of top terms across sources
    st.markdown("## Trending Topics & Terms")
    
    if social_data and "trending_terms" in social_data and social_data["trending_terms"]:
        trending_terms = social_data["trending_terms"]
        
        # Create a horizontal bar chart for trending terms
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Get data for top 10 terms
        terms = [term["term"] for term in trending_terms[:10]]
        counts = [term["count"] for term in trending_terms[:10]]
        
        # Reverse the lists to display highest at top
        terms.reverse()
        counts.reverse()
        
        # Create horizontal bars
        bars = ax.barh(terms, counts, color='#1976D2')
        
        # Add counts at the end of each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width}", 
                   ha='left', va='center', fontsize=9)
        
        # Styling
        ax.set_title("Most Frequently Mentioned Terms", fontsize=14, fontweight='bold')
        ax.set_xlabel("Mention Count", fontsize=10)
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        
        # Word cloud-like visualization
        st.markdown("### Topic Cloud")
        
        # Generate HTML for a word-cloud like display
        html_content = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">'
        
        # Color palette
        colors = ['#1976D2', '#2196F3', '#29B6F6', '#4CAF50', '#66BB6A', '#81C784', 
                 '#FFC107', '#FFD54F', '#FFE082', '#FF9800', '#FFA726', '#FFB74D']
        
        # Find max count for scaling
        max_count = max(term["count"] for term in trending_terms) if trending_terms else 1
        
        for term in trending_terms:
            # Scale the font size based on count
            scale_factor = term["count"] / max_count
            font_size = 14 + (scale_factor * 20)
            
            # Randomly select color
            color = random.choice(colors)
            
            html_content += f'<div style="padding: 5px 15px; background-color: {color}; border-radius: 20px; font-size: {font_size}px; color: white; font-weight: bold;">#{term["term"]}</div>'
        
        html_content += '</div>'
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.info("No trending terms data available.")
    
    # 4. AI-GENERATED INSIGHTS: Combined insights from news and Reddit
    st.markdown("## 🧠 AI-Generated Market Insights")
    
    if news_items or reddit_posts:
        with st.spinner("Generating insights from news and social data..."):
            insights = generate_news_insights(ticker, news_items, reddit_posts)
        
        with st.expander("View Comprehensive Analysis", expanded=True):
            st.markdown(insights)
    else:
        st.info("No data available to generate insights.")
    
    # 5. NEWS & REDDIT TABS: Side-by-side view of sources
    st.markdown("## Latest News & Discussions")
    
    news_reddit_tab1, news_reddit_tab2 = st.tabs(["📰 News Articles", "💬 Reddit Discussions"])
    
    with news_reddit_tab1:
        if news_items:
            # Display all news items
            for i, news in enumerate(news_items):
                # Extract news details
                title = news.get("title", "No title available")
                summary = news.get("summary", "No summary available")
                url = news.get("url", "#")
                source = news.get("source", "Unknown")
                time_published = news.get("time_published", "")
                
                # Format publication date nicely
                try:
                    if time_published:
                        date_obj = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                        formatted_date = date_obj.strftime("%b %d, %Y %H:%M")
                    else:
                        formatted_date = "Unknown date"
                except:
                    formatted_date = time_published
                
                # Get sentiment if available
                sentiment_label = "Neutral"
                sentiment_color = "orange"  # Yellow for neutral
                sentiment_score = 0
                
                if "sentiment_analysis" in news:
                    sentiment = news["sentiment_analysis"]
                    sentiment_label = sentiment["sentiment_label"].capitalize()
                    sentiment_score = sentiment["sentiment_value"]
                    
                    if sentiment_label == "Positive":
                        sentiment_color = "green"
                    elif sentiment_label == "Negative":
                        sentiment_color = "red"
                
                # Display the news item
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"Source: {source}")
                    with col2:
                        st.caption(f"Published: {formatted_date}")
                    
                    st.markdown(f"**[{title}]({url})**")
                    st.markdown(summary[:200] + ("..." if len(summary) > 200 else ""))
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"[Read full article]({url})")
                    with col2:
                        st.markdown(f"<span style='background-color: {sentiment_color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 0.8em;'>{sentiment_label} ({sentiment_score:.2f})</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # Show summary of displayed items
            st.caption(f"Showing {len(news_items)} news items")
        else:
            st.info("No recent news found for this ticker.")
    
    with news_reddit_tab2:
        if reddit_posts:
            # Display Reddit posts
            for i, post in enumerate(reddit_posts):
                title = post.get("title", "No title available")
                content = post.get("content", "")
                subreddit = post.get("subreddit", "Unknown")
                url = post.get("url", "#")
                
                # Try to get date
                date_str = "Unknown date"
                if "date" in post:
                    try:
                        date_str = post["date"].strftime("%b %d, %Y")
                    except:
                        pass
                
                # Display the post
                with st.container():
                    st.markdown(f"**[{title}]({url})**")
                    st.caption(f"Posted in r/{subreddit} on {date_str}")
                    
                    # Show content preview if available
                    if content:
                        preview_length = min(300, len(content))
                        st.markdown(content[:preview_length] + ("..." if len(content) > preview_length else ""))
                    
                    st.markdown(f"[View on Reddit]({url})")
                    st.markdown("---")
            
            # Show summary of displayed items
            st.caption(f"Showing {len(reddit_posts)} Reddit posts")
        else:
            st.info("No Reddit posts found for this ticker.")
    
    # 6. KEY EVENTS TIMELINE
    # Simple function to display key events timeline
    def display_key_events_timeline(ticker):
        st.markdown("## 📅 Key Events Timeline")
        
        # Use a simplified approach to get upcoming events
        def get_simple_events(ticker):
            """
            Get a simple list of upcoming events for a ticker
            Returns top 3 upcoming events with basic details
            """
            try:
                # Get basic company info
                stock = yf.Ticker(ticker)
                
                events = []
                
                # 1. Try to get next earnings date
                try:
                    earnings_calendar = stock.calendar
                    if earnings_calendar is not None:
                        earnings_date = earnings_calendar.get('Earnings Date')
                        if earnings_date is not None:
                            if isinstance(earnings_date, (list, np.ndarray)):
                                earnings_date = earnings_date[0]  # Take the first one if multiple
                            
                            earnings_date = pd.Timestamp(earnings_date).to_pydatetime()
                            
                            events.append({
                                'date': earnings_date,
                                'event': f"Earnings Release",
                                'description': f"{ticker} quarterly earnings announcement",
                                'type': 'earnings',
                                'source': 'Yahoo Finance',
                                'url': f"https://finance.yahoo.com/quote/{ticker}/analysis"
                            })
                except Exception as e:
                    pass
                
                # 2. Try to get ex-dividend date if available
                try:
                    if hasattr(stock, 'dividends') and len(stock.dividends) > 0:
                        # Get the most recent dividend information
                        last_div_date = stock.dividends.index[-1].to_pydatetime()
                        
                        # Estimate next dividend date (assuming quarterly)
                        # This is a simple estimate - not always accurate
                        next_div_date = last_div_date + timedelta(days=90)
                        
                        # Only add if it's in the future
                        if next_div_date > datetime.now():
                            events.append({
                                'date': next_div_date,
                                'event': "Estimated Dividend Date",
                                'description': f"Estimated based on previous dividend pattern",
                                'type': 'dividend',
                                'source': 'Yahoo Finance (estimated)',
                                'url': f"https://finance.yahoo.com/quote/{ticker}/history"
                            })
                except Exception as e:
                    pass
                
                # 3. Add a general shareholder meeting date (if within next 6 months)
                # This is a placeholder - actual dates would require specific data
                try:
                    company_name = stock.info.get('shortName', ticker)
                    
                    # Create an estimated date (just as a placeholder - not accurate)
                    today = datetime.now()
                    meeting_date = datetime(today.year, 4 if today.month < 4 else (today.month + 2), 15)
                    
                    # Only add if it's within the next 6 months
                    six_months_later = today + timedelta(days=180)
                    if meeting_date <= six_months_later and meeting_date >= today:
                        events.append({
                            'date': meeting_date,
                            'event': "Annual Shareholder Meeting",
                            'description': f"Estimated date for {company_name} annual meeting",
                            'type': 'meeting',
                            'source': 'Estimated',
                            'url': f"https://finance.yahoo.com/quote/{ticker}"
                        })
                except Exception as e:
                    pass
                
                # Sort events by date
                events.sort(key=lambda x: x['date'])
                
                # Get just the next 3 upcoming events
                upcoming_events = [e for e in events if e['date'] >= datetime.now()][:3]
                
                return upcoming_events
                
            except Exception as e:
                st.error(f"Error getting simple events: {str(e)}")
                return []
        
        # Get simple events
        events = get_simple_events(ticker)
        
        if events:
            # Display timeline
            for event in events:
                # Get event details
                event_type = event.get('type', 'other')
                event_title = event.get('event', 'Unknown Event')
                event_desc = event.get('description', '')
                event_date = event.get('date', datetime.now())
                event_source = event.get('source', 'Unknown Source')
                event_url = event.get('url', '#')
                
                # Determine color based on event type
                if event_type == 'earnings':
                    event_color = "#9C27B0"  # Purple
                    event_icon = "📊"
                elif event_type == 'financial':
                    event_color = "#2196F3"  # Blue
                    event_icon = "💰"
                elif event_type == 'leadership':
                    event_color = "#FF9800"  # Orange
                    event_icon = "👔"
                elif event_type == 'ma':
                    event_color = "#4CAF50"  # Green
                    event_icon = "🤝"
                elif event_type == 'legal':
                    event_color = "#F44336"  # Red
                    event_icon = "⚖️"
                elif event_type == 'product':
                    event_color = "#00BCD4"  # Cyan
                    event_icon = "🚀"
                elif event_type == 'dividend':
                    event_color = "#8BC34A"  # Light green
                    event_icon = "💵"
                elif event_type == 'meeting':
                    event_color = "#673AB7"  # Deep purple
                    event_icon = "🗓️"
                elif event_type == 'partnership':
                    event_color = "#3F51B5"  # Indigo
                    event_icon = "🔗"
                else:
                    event_color = "#607D8B"  # Gray
                    event_icon = "📋"
                
                # Display date in the format: Aug 15, 2023
                formatted_date = event_date.strftime('%b %d, %Y')
                
                # Create a timeline entry with improved styling
                st.markdown(f"""
                <div style="display: flex; margin-bottom: 25px; background-color: #f9f9f9; border-radius: 8px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="min-width: 100px; text-align: right; padding-right: 15px; font-size: 14px; color: #666; font-weight: bold;">
                        {formatted_date}
                    </div>
                    <div style="margin-left: 10px; width: 30px; display: flex; flex-direction: column; align-items: center;">
                        <div style="width: 30px; height: 30px; border-radius: 50%; background-color: {event_color}; display: flex; justify-content: center; align-items: center; color: white; font-size: 14px;">
                            {event_icon}
                        </div>
                        <div style="width: 2px; height: 100%; background-color: #e0e0e0; margin-top: 5px;"></div>
                    </div>
                    <div style="margin-left: 15px; flex-grow: 1;">
                        <div style="font-weight: bold; font-size: 16px; color: {event_color}; display: flex; justify-content: space-between;">
                            <span>{event_title}</span>
                            <a href="{event_url}" target="_blank" style="color: #2196F3; text-decoration: none; font-size: 14px;">View Details</a>
                        </div>
                        <div style="margin-top: 5px; color: #555;">
                            {event_desc}
                        </div>
                        <div style="font-size: 12px; color: #666; margin-top: 10px;">
                            Source: {event_source}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Add helper text
            st.caption(f"Showing upcoming events for {ticker}.")
        else:
            # Fallback to display if no events are found
            st.info(f"No upcoming key events found for {ticker}.")
            
            # Suggestions for the user
            st.markdown("""
            Check the company's investor relations page for the most up-to-date event information.
            """)

    # Usage in the News & Insights tab
    # Replace the original timeline section with this call:
    display_key_events_timeline(ticker)
    
    # Add a full-width disclaimer at the bottom
    st.markdown("""
    <div style="margin-top: 30px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666;">
        <strong>Disclaimer:</strong> News data and AI-generated insights are provided for informational purposes only and should not be considered as financial advice. 
        Always conduct your own research before making investment decisions.
    </div>
    """, unsafe_allow_html=True)   

# ----------- SUMMARY & ACTION TAB ----------- #
with tabs[4]:
    st.subheader(f"🎯 {ticker} Investment Summary & Action Plan")
    
    # Store the financial health score if it was calculated in Financials tab
    final_health_score = None
    try:
        # This assumes the code in the Financials tab calculated and stored this value
        if 'final_health_score' in locals():
            final_health_score = final_health_score
    except:
        pass
    
    # Create a spinner while generating the summary
    with st.spinner("Generating comprehensive investment summary..."):
        # Generate the executive summary
        executive_summary = generate_executive_summary(
            ticker, 
            stock_data, 
            stock_info, 
            metrics, 
            returns, 
            final_health_score,
            peer_data if 'peer_data' in locals() else None
        )
    
    # Display executive summary in an attractive container
    with st.container(border=True):
        st.markdown(executive_summary)
    
    # Create a visual recommendation indicator
    if "Buy" in executive_summary:
        rec_color = "#4CAF50"  # Green
        recommendation = "Buy"
    elif "Sell" in executive_summary:
        rec_color = "#F44336"  # Red
        recommendation = "Sell"
    else:
        rec_color = "#FFC107"  # Yellow/Amber
        recommendation = "Hold"
    
    # Display a prominent recommendation indicator
    st.markdown(f"""
    <div style="text-align: center; margin: 30px 0;">
        <div style="display: inline-block; background-color: {rec_color}; color: white; padding: 15px 30px; 
                border-radius: 50px; font-size: 24px; font-weight: bold;">
            Recommendation: {recommendation}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a simple timeline visualization for investment horizon
    st.markdown("### Investment Timeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f1f8e9; padding: 15px; border-radius: 10px; text-align: center;">
            <h4 style="margin-top: 0;">Short Term (1-3 months)</h4>
            <p>Monitor quarterly earnings reports and technical indicators for potential entry/exit points.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; text-align: center;">
            <h4 style="margin-top: 0;">Medium Term (3-12 months)</h4>
            <p>Evaluate business performance and competitive positioning in the industry.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #d0f8ce; padding: 15px; border-radius: 10px; text-align: center;">
            <h4 style="margin-top: 0;">Long Term (1+ years)</h4>
            <p>Focus on fundamental growth drivers and market expansion opportunities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a disclaimer for compliance
    st.markdown("""
    <div style="font-size: 12px; color: #777; text-align: center; margin-top: 30px;">
        <p><strong>Disclaimer:</strong> This summary is generated automatically based on available data and should not be 
        considered as personalized investment advice. Always conduct your own research and consult with a qualified 
        financial advisor before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# Add footer for consistency
display_footer()