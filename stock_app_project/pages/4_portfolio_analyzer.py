import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import os
import sys
from datetime import datetime, timedelta
import cohere
import io
import base64
from PIL import Image
import re

# Add the parent directory to system path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_theme import (
    apply_custom_theme, display_header, display_footer, enhance_sidebar,
    display_metric, display_card, display_alert, display_stock_card
)

from utils import (
    initialize_session_state, update_stock_context, get_active_stock,
    add_to_watchlist, remove_from_watchlist, get_stock_data, get_stock_info,
    calculate_portfolio_metrics, save_portfolio_data, get_portfolio_data,
    format_large_number, parse_financial_value
)

# Initialize Cohere client for AI-generated content
co = cohere.Client("TvrgUHF3GKzAB5sYBHK7UkHApcr2VZ0nJnBkNATD")

# ---------------- Page Configuration ---------------- #
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
apply_custom_theme()

# Initialize session state
initialize_session_state()

# Enhance sidebar
enhance_sidebar()

# ---------------- Page Header ---------------- #
display_header("💼 Portfolio Analyzer", 
              "Advanced portfolio management tools for tracking, analyzing, and optimizing your investments")

# ---------------- Helper Functions ---------------- #
def extract_tickers_from_text(text):
    """Extract potential stock tickers from text"""
    pattern = r'\b[A-Z]{1,5}\b'
    matches = re.findall(pattern, text)
    
    # Filter out common words that might be mistaken for tickers
    common_words = {'A', 'I', 'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME', 
                   'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE'}
    
    return [m for m in matches if m not in common_words]

def process_portfolio_screenshot(image_bytes):
    """Process a portfolio screenshot to extract data (placeholder for OCR)"""
    # In a real app, this would use OCR to extract portfolio data
    # For now, we'll just return a message
    
    st.info("Image uploaded! OCR processing would extract your portfolio holdings here.")
    st.image(Image.open(io.BytesIO(image_bytes)), caption="Uploaded Portfolio Image", width=400)
    
    # Return placeholder data
    return pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'quantity': [10, 5, 2],
        'price': [180.0, 350.0, 2800.0]
    })

def retrieve_stock_data(tickers):
    """Retrieve price and sector data for tickers"""
    result = []
    
    for ticker in tickers:
        try:
            # Get stock info
            info = get_stock_info(ticker)
            
            # Extract current price
            if 'regularMarketPrice' in info:
                price = info['regularMarketPrice']
            else:
                price_data = get_stock_data(ticker, period="1d")
                price = float(price_data['Close'].iloc[-1]) if not price_data.empty else 0
            
            # Extract sector
            sector = info.get('sector', 'Unknown')
            
            # Add to results
            result.append({
                'ticker': ticker,
                'price': price,
                'sector': sector,
                'name': info.get('longName', ticker)
            })
        except Exception as e:
            st.warning(f"Could not retrieve data for {ticker}: {e}")
    
    return pd.DataFrame(result)

def generate_portfolio_recommendations(portfolio_df, metrics):
    """Generate AI recommendations for portfolio optimization"""
    if portfolio_df.empty:
        return "Please upload or enter your portfolio to receive recommendations."
    
    try:
        # Extract metrics for the prompt
        total_value = metrics.get('total_value', 0)
        num_holdings = metrics.get('num_holdings', 0)
        top_holding = metrics.get('top_holding', 'N/A')
        top_holding_weight = metrics.get('top_holding_weight', 0)
        top_3_weight = metrics.get('top_3_weight', 0)
        
        # Get sector allocation data
        sector_allocation = metrics.get('sector_allocation', pd.DataFrame())
        if not sector_allocation.empty:
            sector_text = "\n".join([f"- {row['sector']}: {row['percentage']:.1f}%" for _, row in sector_allocation.iterrows()])
        else:
            sector_text = "- No sector data available"
        
        # Create portfolio holdings text
        holdings_text = "\n".join([
            f"- {row['ticker']}: {row['quantity']} shares at ${row['price']:.2f} (${row['value']:.2f}, {row['weight']*100:.1f}%)"
            for _, row in portfolio_df.iterrows()
        ])
        
        # Create prompt for the Cohere API
        prompt = f"""
        As a professional financial advisor, provide 3-5 specific recommendations to optimize this stock portfolio:
        
        Portfolio Summary:
        - Total value: ${total_value:.2f}
        - Number of holdings: {num_holdings}
        - Top holding: {top_holding} ({top_holding_weight:.1f}%)
        - Top 3 holdings concentration: {top_3_weight:.1f}%
        
        Sector Allocation:
        {sector_text}
        
        Holdings:
        {holdings_text}
        
        Provide actionable recommendations focused on:
        1. Diversification improvements
        2. Sector balance
        3. Concentration risk management
        4. Any other relevant optimization suggestions
        
        Keep each recommendation concise and specific. Don't use platitudes or generic advice.
        Format as a bulleted list with 3-5 actionable items.
        """
        
        # Get recommendations from Cohere
        response = co.generate(
            model='command-light',
            prompt=prompt,
            max_tokens=400,
            temperature=0.7
        )
        
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Unable to generate recommendations: {str(e)}"

def calculate_risk_metrics(portfolio_df):
    """Calculate various risk metrics for the portfolio"""
    if portfolio_df.empty:
        return {}
    
    try:
        # Get historical data for each stock
        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df['weight'].tolist()
        
        if len(tickers) == 0:
            return {
                'Expected Annual Return': 0,
                'Annual Volatility': 0,
                'Sharpe Ratio': 0,
                'Maximum Drawdown': 0
            }
        
        # Fetch historical prices for the last year
        hist_data = yf.download(tickers, period="1y")
        
        # Check if we have any data and if 'Adj Close' is in the columns
        if hist_data.empty:
            st.warning("No historical price data available. Using default risk metrics.")
            return {
                'Expected Annual Return': 0,
                'Annual Volatility': 0,
                'Sharpe Ratio': 0,
                'Maximum Drawdown': 0
            }
        
        # Handle different column structures based on number of tickers
        if isinstance(hist_data.columns, pd.MultiIndex):
            if 'Adj Close' in hist_data.columns.levels[0]:
                price_data = hist_data['Adj Close']
            else:
                # Fallback to 'Close' if 'Adj Close' is not available
                price_data = hist_data['Close'] if 'Close' in hist_data.columns.levels[0] else None
        else:
            # Single ticker case
            price_data = hist_data['Adj Close'] if 'Adj Close' in hist_data.columns else hist_data['Close']
        
        if price_data is None or price_data.empty:
            st.warning("Price data could not be retrieved. Using default risk metrics.")
            return {
                'Expected Annual Return': 0,
                'Annual Volatility': 0,
                'Sharpe Ratio': 0,
                'Maximum Drawdown': 0
            }
        
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        if returns.empty or returns.shape[0] < 5:
            st.warning("Insufficient historical data to calculate reliable risk metrics.")
            return {
                'Expected Annual Return': 0,
                'Annual Volatility': 0,
                'Sharpe Ratio': 0,
                'Maximum Drawdown': 0
            }
        
        # Ensure weights are properly aligned with return columns
        # For a single ticker, ensure weights is a single value
        if len(returns.columns) == 1 and len(weights) > 1:
            weights = [1.0]
        
        # For multiple tickers, ensure we have the correct number of weights
        if len(returns.columns) > 1 and len(returns.columns) != len(weights):
            # Adjust weights to match the available data
            available_tickers = list(returns.columns)
            ticker_to_weight = dict(zip(tickers, weights))
            weights = [ticker_to_weight.get(ticker, 0) for ticker in available_tickers]
            
            # Normalize weights to sum to 1
            if sum(weights) > 0:
                weights = [w/sum(weights) for w in weights]
            else:
                weights = [1/len(available_tickers) for _ in available_tickers]
        
        # Calculate portfolio return
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
        
        # Calculate portfolio volatility (risk)
        cov_matrix = returns.cov() * 252  # Annualized
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max) - 1
        max_drawdown = drawdown.min().min()
        
        return {
            'Expected Annual Return': portfolio_return * 100,  # as percentage
            'Annual Volatility': portfolio_vol * 100,  # as percentage
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown * 100  # as percentage
        }
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")
        return {
            'Expected Annual Return': 0,
            'Annual Volatility': 0,
            'Sharpe Ratio': 0,
            'Maximum Drawdown': 0
        }

def generate_portfolio_summary(portfolio_df, metrics, risk_metrics):
    """Generate a natural language summary of the portfolio"""
    if portfolio_df.empty:
        return "Please upload or enter your portfolio to see a summary."
    
    try:
        # Extract metrics for the prompt
        total_value = metrics.get('total_value', 0)
        num_holdings = metrics.get('num_holdings', 0)
        top_holding = metrics.get('top_holding', 'N/A')
        top_holding_weight = metrics.get('top_holding_weight', 0)
        
        # Extract risk metrics
        exp_return = risk_metrics.get('Expected Annual Return', 0)
        volatility = risk_metrics.get('Annual Volatility', 0)
        sharpe = risk_metrics.get('Sharpe Ratio', 0)
        max_dd = risk_metrics.get('Maximum Drawdown', 0)
        
        # Get sector allocation
        sector_allocation = metrics.get('sector_allocation', pd.DataFrame())
        if not sector_allocation.empty:
            top_sectors = sector_allocation.nlargest(3, 'percentage')
            sector_text = ", ".join([f"{row['sector']} ({row['percentage']:.1f}%)" for _, row in top_sectors.iterrows()])
        else:
            sector_text = "No sector data available"
        
        # Create prompt for the Cohere API
        prompt = f"""
        As a portfolio analyst, provide a concise 3-4 sentence summary of this stock portfolio:
        
        Portfolio Summary:
        - Total value: ${total_value:.2f}
        - Number of holdings: {num_holdings}
        - Top holding: {top_holding} ({top_holding_weight:.1f}%)
        - Top sectors: {sector_text}
        - Expected annual return: {exp_return:.2f}%
        - Annual volatility: {volatility:.2f}%
        - Sharpe ratio: {sharpe:.2f}
        - Maximum drawdown: {max_dd:.2f}%
        
        Provide a balanced assessment that includes portfolio size, diversification, sector concentration, and risk/return profile.
        Keep it factual and objective.
        """
        
        # Get summary from Cohere
        response = co.generate(
            model='command-light',
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Unable to generate portfolio summary: {str(e)}"

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.subheader("💼 Portfolio Tools")
    
    # Input method selection
    input_method = st.radio(
        "Select input method:",
        options=["Upload Screenshot", "Manual Entry", "Load Sample Portfolio"]
    )
    
    st.markdown("---")
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    
    # Risk profile selection
    risk_profile = st.select_slider(
        "Risk Profile:",
        options=["Conservative", "Moderate", "Balanced", "Growth", "Aggressive"],
        value="Balanced"
    )
    
    # Time horizon
    time_horizon = st.select_slider(
        "Investment Time Horizon:",
        options=["Short-term", "Medium-term", "Long-term"],
        value="Medium-term"
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        show_all_metrics = st.checkbox("Show advanced risk metrics", value=False)
        include_crypto = st.checkbox("Include cryptocurrency analysis", value=False)
        benchmark_index = st.selectbox(
            "Benchmark Index:",
            options=["S&P 500", "Nasdaq", "Dow Jones", "Russell 2000"],
            index=0
        )
    
    # Information
    st.markdown("---")
    st.info("""
    **About Portfolio Analysis**
    
    This tool helps you analyze your investment portfolio for diversification, risk, and potential optimization opportunities.
    
    All data is processed locally and not stored permanently.
    """)

# ---------------- Main Content ---------------- #
# Display different input methods based on selection
if input_method == "Upload Screenshot":
    st.subheader("📸 Upload Portfolio Screenshot")
    
    st.markdown("""
    Upload a screenshot of your portfolio from your brokerage account. 
    Our system will attempt to extract holding information using OCR.
    
    **Supported formats:** JPG, PNG
    """)
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Get file content and process
        image_bytes = uploaded_file.getvalue()
        portfolio_df = process_portfolio_screenshot(image_bytes)
        
        # Add column for value
        portfolio_df['value'] = portfolio_df['quantity'] * portfolio_df['price']
        
        # Calculate total value for weights
        total_value = portfolio_df['value'].sum()
        portfolio_df['weight'] = portfolio_df['value'] / total_value
        
        # Get additional data from yfinance
        stock_data = retrieve_stock_data(portfolio_df['ticker'])
        
        # Merge with stock data to get sectors
        portfolio_df = portfolio_df.merge(stock_data[['ticker', 'sector', 'name']], on='ticker', how='left')
        
        # Save portfolio data to session state
        metrics = calculate_portfolio_metrics(portfolio_df)
        recommendations = generate_portfolio_recommendations(portfolio_df, metrics)
        save_portfolio_data(portfolio_df, metrics, recommendations)

elif input_method == "Manual Entry":
    st.subheader("✏️ Manual Portfolio Entry")
    
    st.markdown("""
    Enter your stock holdings manually. Add as many positions as needed.
    """)
    
    # Create empty DataFrame if none exists in session state
    if 'manual_portfolio' not in st.session_state:
        st.session_state.manual_portfolio = pd.DataFrame(columns=['ticker', 'quantity', 'price', 'sector'])
        st.session_state.manual_portfolio_count = 0
    
    # Form for adding new position
    with st.form("add_position_form"):
        st.subheader("Add Position")
        position_col1, position_col2, position_col3 = st.columns(3)
        
        with position_col1:
            ticker = st.text_input("Ticker Symbol").upper()
        
        with position_col2:
            quantity = st.number_input("Quantity", min_value=0.0, step=0.01)
        
        with position_col3:
            price = st.number_input("Current Price ($)", min_value=0.0, step=0.01)
        
        submit_button = st.form_submit_button(label="Add Position")
        
        if submit_button and ticker and quantity > 0 and price > 0:
            # Try to get sector information
            try:
                info = get_stock_info(ticker)
                sector = info.get('sector', 'Unknown')
                name = info.get('longName', ticker)
            except:
                sector = "Unknown"
                name = ticker
            
            # Add to existing portfolio
            new_position = pd.DataFrame({
                'ticker': [ticker],
                'quantity': [quantity],
                'price': [price],
                'sector': [sector],
                'name': [name],
                'value': [quantity * price]
            })
            
            st.session_state.manual_portfolio = pd.concat([st.session_state.manual_portfolio, new_position], ignore_index=True)
            st.session_state.manual_portfolio_count += 1
            
            st.success(f"Added {ticker} to your portfolio!")
            st.rerun()
    
    # Display the current portfolio
    if not st.session_state.manual_portfolio.empty:
        st.subheader("Your Portfolio")
        
        # Calculate values and weights
        portfolio_df = st.session_state.manual_portfolio.copy()
        portfolio_df['value'] = portfolio_df['quantity'] * portfolio_df['price']
        total_value = portfolio_df['value'].sum()
        portfolio_df['weight'] = portfolio_df['value'] / total_value
        
        # Display as table
        display_df = portfolio_df[['ticker', 'name', 'quantity', 'price', 'value', 'weight', 'sector']].copy()
        display_df['price'] = display_df['price'].map('${:,.2f}'.format)
        display_df['value'] = display_df['value'].map('${:,.2f}'.format)
        display_df['weight'] = display_df['weight'].map('{:.2%}'.format)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio_df)
        recommendations = generate_portfolio_recommendations(portfolio_df, metrics)
        save_portfolio_data(portfolio_df, metrics, recommendations)
        
        # Button to clear portfolio
        if st.button("Clear Portfolio"):
            st.session_state.manual_portfolio = pd.DataFrame(columns=['ticker', 'quantity', 'price', 'sector'])
            st.session_state.manual_portfolio_count = 0
            st.rerun()
    else:
        st.info("Add positions to your portfolio using the form above.")

elif input_method == "Load Sample Portfolio":
    st.subheader("📊 Sample Portfolio")
    
    # Sample portfolio data
    sample_portfolio = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'DIS'],
        'quantity': [15, 10, 5, 8, 12, 20, 30, 18, 25, 22],
        'price': [180.0, 350.0, 2800.0, 145.0, 330.0, 220.0, 480.0, 145.0, 240.0, 110.0]
    })
    
    # Get additional data from yfinance
    stock_data = retrieve_stock_data(sample_portfolio['ticker'])
    
    # Add values and merge with stock data
    sample_portfolio['value'] = sample_portfolio['quantity'] * sample_portfolio['price']
    total_value = sample_portfolio['value'].sum()
    sample_portfolio['weight'] = sample_portfolio['value'] / total_value
    
    # Merge with stock data to get sectors and names
    portfolio_df = sample_portfolio.merge(stock_data[['ticker', 'sector', 'name']], on='ticker', how='left')
    
    # Display sample portfolio
    st.markdown("This is a sample diversified portfolio for demonstration purposes.")
    
    # Display as table
    display_df = portfolio_df[['ticker', 'name', 'quantity', 'price', 'value', 'weight', 'sector']].copy()
    display_df['price'] = display_df['price'].map('${:,.2f}'.format)
    display_df['value'] = display_df['value'].map('${:,.2f}'.format)
    display_df['weight'] = display_df['weight'].map('{:.2%}'.format)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(portfolio_df)
    recommendations = generate_portfolio_recommendations(portfolio_df, metrics)
    save_portfolio_data(portfolio_df, metrics, recommendations)
    
    st.info("This sample portfolio is provided for demonstration purposes only and does not constitute financial advice.")

# ---------------- Portfolio Analysis ---------------- #
# Check if we have portfolio data to analyze
portfolio_data = get_portfolio_data()

if portfolio_data:
    st.markdown("---")
    st.subheader("📊 Portfolio Analysis")
    
    # Extract data from session state
    portfolio_df = portfolio_data.get('df')
    metrics = portfolio_data.get('metrics')
    recommendations = portfolio_data.get('recommendations')
    
    # Display portfolio overview
    st.markdown("### Portfolio Overview")
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(portfolio_df)
    
    # Generate portfolio summary
    portfolio_summary = generate_portfolio_summary(portfolio_df, metrics, risk_metrics)
    
    # Display summary and key metrics
    st.markdown(f"""
    <div style="background-color:#f5f5f5;padding:20px;border-radius:10px;margin-bottom:20px;">
        <p style="font-style:italic;">{portfolio_summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric(
            "Total Value",
            f"{metrics.get('total_value', 0):,.2f}",
            prefix="$"
        )
    
    with col2:
        display_metric(
            "Number of Holdings",
            f"{metrics.get('num_holdings', 0)}"
        )
    
    with col3:
        display_metric(
            "Top Holding Weight",
            f"{metrics.get('top_holding_weight', 0):.2f}",
            suffix="%"
        )
    
    with col4:
        display_metric(
            "Expected Return",
            f"{risk_metrics.get('Expected Annual Return', 0):.2f}",
            suffix="%"
        )
    
    # Allocation charts
    st.markdown("### Portfolio Allocation")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Create holdings chart
        holdings_df = portfolio_df[['ticker', 'name', 'weight']].copy()
        holdings_df['weight'] = holdings_df['weight'] * 100  # Convert to percentage
        
        fig = px.pie(
            holdings_df,
            values='weight',
            names='ticker',
            title='Portfolio Allocation by Stock',
            hover_data=['name'],
            labels={'weight': 'Allocation (%)'},
            hole=0.4
        )
        
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Create sector allocation chart
        sector_allocation = metrics.get('sector_allocation')
        if sector_allocation is not None and not sector_allocation.empty:
            fig = px.bar(
                sector_allocation,
                x='sector',
                y='percentage',
                title='Sector Allocation',
                labels={'percentage': 'Allocation (%)', 'sector': 'Sector'},
                color='sector',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                yaxis=dict(ticksuffix="%")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sector allocation data not available.")
    
    # Risk analysis
    st.markdown("### Risk Analysis")
    
    # Display risk metrics in columns
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    
    with risk_col1:
        display_metric(
            "Expected Return",
            f"{risk_metrics.get('Expected Annual Return', 0):.2f}",
            suffix="%"
        )
    
    with risk_col2:
        display_metric(
            "Volatility",
            f"{risk_metrics.get('Annual Volatility', 0):.2f}",
            suffix="%"
        )
    
    with risk_col3:
        sharpe = risk_metrics.get('Sharpe Ratio', 0)
        sharpe_color = "green" if sharpe > 1 else "red" if sharpe < 0 else "orange"
        display_metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}"
        )
    
    with risk_col4:
        display_metric(
            "Max Drawdown",
            f"{risk_metrics.get('Maximum Drawdown', 0):.2f}",
            suffix="%"
        )
    
    # Show risk interpretation
    sharpe = risk_metrics.get('Sharpe Ratio', 0)
    
    if sharpe > 1.5:
        risk_assessment = "Excellent risk-adjusted returns"
    elif sharpe > 1:
        risk_assessment = "Good risk-adjusted returns"
    elif sharpe > 0.5:
        risk_assessment = "Average risk-adjusted returns"
    elif sharpe > 0:
        risk_assessment = "Below-average risk-adjusted returns"
    else:
        risk_assessment = "Poor risk-adjusted returns"
    
    st.markdown(f"""
    <div class="insights-panel">
        <p><strong>Risk Assessment:</strong> This portfolio has {risk_assessment}, with a Sharpe ratio of {sharpe:.2f}. 
        The expected annual return is {risk_metrics.get('Expected Annual Return', 0):.2f}% with 
        {risk_metrics.get('Annual Volatility', 0):.2f}% annual volatility.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show advanced risk visualization if requested
    if show_all_metrics:
        st.markdown("### Advanced Risk Visualization")
        
        # Try to create efficient frontier chart
        try:
            # Simulate different portfolio weights
            num_portfolios = 1000
            
            # Get tickers from portfolio
            tickers = portfolio_df['ticker'].tolist()
            
            if len(tickers) < 2:
                st.warning("Efficient frontier visualization requires at least two stocks.")
            else:
                # Download historical data
                hist_data = yf.download(tickers, period="1y")
                
                # Handle different column structures
                if isinstance(hist_data.columns, pd.MultiIndex):
                    if 'Adj Close' in hist_data.columns.levels[0]:
                        price_data = hist_data['Adj Close']
                    else:
                        # Fallback to 'Close' if 'Adj Close' is not available
                        price_data = hist_data['Close'] if 'Close' in hist_data.columns.levels[0] else None
                else:
                    # Single ticker case
                    price_data = hist_data['Adj Close'] if 'Adj Close' in hist_data.columns else hist_data['Close']
                
                if price_data is None or price_data.empty:
                    st.warning("Could not retrieve sufficient price data for efficient frontier analysis.")
                else:
                    # Calculate returns
                    returns_data = price_data.pct_change().dropna()
                    
                    if returns_data.empty or returns_data.shape[0] < 5:
                        st.warning("Insufficient historical data to generate an efficient frontier.")
                    else:
                        # Prepare results array
                        results = np.zeros((num_portfolios, 2))
                        
                        # Mean returns and covariance
                        mean_returns = returns_data.mean() * 252
                        cov_matrix = returns_data.cov() * 252
                        
                        # Get current weights aligned with available tickers
                        portfolio_tickers = portfolio_df['ticker'].tolist()
                        portfolio_weights = portfolio_df['weight'].values
                        
                        # Create a mapping of ticker to weight
                        ticker_to_weight = dict(zip(portfolio_tickers, portfolio_weights))
                        
                        # Check which tickers from the portfolio are in the returns data
                        available_tickers = list(returns_data.columns)
                        
                        # Align current weights with available return data
                        current_weights = np.array([ticker_to_weight.get(ticker, 0) for ticker in available_tickers])
                        
                        # Normalize weights if needed
                        if sum(current_weights) > 0:
                            current_weights = current_weights / sum(current_weights)
                        else:
                            current_weights = np.array([1/len(available_tickers) for _ in available_tickers])
                        
                        # Simulate random portfolios
                        for i in range(num_portfolios):
                            weights = np.random.random(len(available_tickers))
                            weights /= np.sum(weights)
                            
                            # Calculate portfolio return and volatility
                            portfolio_return = np.sum(mean_returns * weights)
                            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                            
                            results[i, 0] = portfolio_std_dev
                            results[i, 1] = portfolio_return
                        
                        # Create efficient frontier plot
                        ef_df = pd.DataFrame(results, columns=['Volatility', 'Return'])
                        
                        # Calculate current portfolio metrics
                        current_return = np.sum(mean_returns * current_weights)
                        current_vol = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
                        
                        fig = px.scatter(
                            ef_df,
                            x='Volatility',
                            y='Return',
                            title='Efficient Frontier and Portfolio Performance',
                            opacity=0.5,
                            labels={'Volatility': 'Volatility (σ)', 'Return': 'Expected Return'},
                            color_discrete_sequence=['lightblue']
                        )
                        
                        # Add current portfolio point
                        fig.add_trace(
                            go.Scatter(
                                x=[current_vol],
                                y=[current_return],
                                mode='markers',
                                marker=dict(size=15, color='red'),
                                name='Current Portfolio'
                            )
                        )
                        
                        fig.update_layout(
                            xaxis=dict(tickformat=".0%"),
                            yaxis=dict(tickformat=".0%")
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation of chart
                        st.markdown("""
                        **Interpretation:**
                        - Each blue dot represents a possible portfolio with different stock weightings
                        - The red dot represents your current portfolio
                        - Portfolios closer to the top-left corner have better risk-return profiles
                        - If your portfolio is below the efficient frontier curve, portfolio optimization could improve returns without increasing risk
                        """)
        except Exception as e:
            st.error(f"Unable to generate efficient frontier: {str(e)}")
            st.info("Try checking your internet connection or using a different set of stocks.")
    
    # Correlation matrix
    st.markdown("### Stock Correlation Analysis")
    
    # Try to create correlation matrix
    try:
        tickers = portfolio_df['ticker'].tolist()
        
        if len(tickers) > 1:  # Only create correlation matrix if we have more than one ticker
            # Get historical price data - add error handling for missing data
            price_data = yf.download(tickers, period="1y")
            
            # Check if we have any data and if 'Adj Close' is in the columns
            if not price_data.empty:
                # If we have multi-level columns (which yfinance returns for multiple tickers)
                if isinstance(price_data.columns, pd.MultiIndex):
                    if 'Adj Close' in price_data.columns.levels[0]:
                        close_data = price_data['Adj Close']
                    else:
                        # Fallback to 'Close' if 'Adj Close' is not available
                        close_data = price_data['Close'] if 'Close' in price_data.columns.levels[0] else None
                else:
                    # Single ticker case
                    close_data = price_data['Adj Close'] if 'Adj Close' in price_data.columns else price_data['Close']
                
                if close_data is not None:
                    # Calculate returns
                    returns_data = close_data.pct_change().dropna()
                    
                    # Check if we have enough data
                    if not returns_data.empty and returns_data.shape[0] > 5:  # Ensure we have at least a few data points
                        # Calculate correlation matrix
                        corr_matrix = returns_data.corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Stock Correlation Matrix",
                            labels=dict(color="Correlation")
                        )
                        
                        fig.update_layout(
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation of correlation
                        st.markdown("""
                        **Interpretation:**
                        - Values closer to 1 (dark blue) indicate strong positive correlation (stocks tend to move together)
                        - Values closer to -1 (dark red) indicate strong negative correlation (stocks tend to move in opposite directions)
                        - Values close to 0 indicate little correlation between stocks
                        - A well-diversified portfolio typically includes stocks with low or negative correlations to each other
                        """)
                    else:
                        st.warning("Insufficient historical data to generate a correlation matrix.")
                else:
                    st.warning("Unable to retrieve price data for one or more stocks.")
            else:
                st.warning("No historical price data available for the selected stocks.")
        else:
            st.info("Correlation analysis requires at least two stocks in your portfolio.")
    except Exception as e:
        st.error(f"Unable to generate correlation matrix: {str(e)}")
        st.info("Try checking your internet connection or using different stock tickers.")
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    # Prepare export data
    export_df = portfolio_df[['ticker', 'name', 'quantity', 'price', 'value', 'weight', 'sector']].copy()
    export_df['weight'] = export_df['weight'] * 100  # Convert to percentage
    
    # Create CSV
    csv = export_df.to_csv(index=False).encode('utf-8')
    
    # Download button
    st.download_button(
        label="📥 Download Portfolio Analysis (CSV)",
        data=csv,
        file_name="portfolio_analysis.csv",
        mime="text/csv"
    )
    
    # Navigation buttons
    st.markdown("---")
    
    # Create a row of buttons for quick navigation
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    
    with nav_col1:
        st.markdown("""
        <div style="text-align:center;">
            <a href="/Research_Center" target="_self" style="text-decoration:none;">
                <div class="custom-button-secondary" style="width:100%;">
                    🔬 Research Center
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown("""
        <div style="text-align:center;">
            <a href="/News_Insights" target="_self" style="text-decoration:none;">
                <div class="custom-button-accent" style="width:100%;">
                    📰 News Analysis
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        st.markdown("""
        <div style="text-align:center;">
            <a href="/" target="_self" style="text-decoration:none;">
                <div class="custom-button" style="width:100%;background-color:#9E9E9E;">
                    🏠 Dashboard
                </div>
            </a>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Please enter or upload your portfolio data to see the analysis.")

# Display footer
display_footer()