import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import re

# ---------------- Session State Management ---------------- #
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "context" not in st.session_state:
        st.session_state.context = {
            "active_stock": None,
            "recent_stocks": [],
            "watchlist": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "analysis_results": {},
            "forecast_data": {},
            "news_data": {},
            "portfolio_data": None
        }

def update_stock_context(ticker):
    """Update context when viewing a stock"""
    if not ticker:
        return
        
    st.session_state.context["active_stock"] = ticker
    
    # Add to recent stocks if not already there
    if ticker not in st.session_state.context["recent_stocks"]:
        st.session_state.context["recent_stocks"].insert(0, ticker)
        # Keep only the 5 most recent
        st.session_state.context["recent_stocks"] = st.session_state.context["recent_stocks"][:5]

def get_active_stock():
    """Get active stock with fallback to default"""
    return st.session_state.context.get("active_stock", "AAPL")

def add_to_watchlist(ticker):
    """Add a stock to the watchlist"""
    if ticker and ticker not in st.session_state.context["watchlist"]:
        st.session_state.context["watchlist"].append(ticker)
        return True
    return False

def remove_from_watchlist(ticker):
    """Remove a stock from the watchlist"""
    if ticker in st.session_state.context["watchlist"]:
        st.session_state.context["watchlist"].remove(ticker)
        return True
    return False

def save_portfolio_data(portfolio_df, metrics, recommendations):
    """Save portfolio analysis results to session state"""
    st.session_state.context["portfolio_data"] = {
        "df": portfolio_df,
        "metrics": metrics,
        "recommendations": recommendations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def get_portfolio_data():
    """Retrieve saved portfolio data"""
    return st.session_state.context.get("portfolio_data", None)

# ---------------- Stock Data Utilities ---------------- #
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period="1y", interval="1d"):
    """
    Get historical stock data with explicit date handling to avoid future dates.
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    
    try:
        # Set end date to yesterday to avoid partial data for today
        end_date = datetime.now() - timedelta(days=1)
        
        # Calculate start date based on the period
        if period == "1mo" or period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo" or period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo" or period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        elif period == "max":
            start_date = None  # yfinance will fetch all available data
        else:
            # Default to 1 year
            start_date = end_date - timedelta(days=365)
        
        # Format dates to strings for yfinance
        end_date_str = end_date.strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
        
        # Download data with explicit date range - REMOVED DEBUG MESSAGES
        if start_date:
            data = yf.download(ticker, start=start_date_str, end=end_date_str, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period="max", interval=interval, progress=False)
        
        # Check if data is empty - REMOVED INFO MESSAGE
        if data.empty:
            return pd.DataFrame()
        
        # Ensure the data is sorted by date
        data = data.sort_index()
        
        return data
    except Exception as e:
        # Instead of showing error, just return empty DataFrame
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_info(ticker):
    """Get stock information"""
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception as e:
        st.error(f"Error fetching info for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_news(ticker, limit=5):
    """Get recent news for a stock - improved version"""
    try:
        ticker_obj = yf.Ticker(ticker)
        news = []
        
        # Try to get news in multiple ways
        try:
            # First try the news property
            if hasattr(ticker_obj, 'news'):
                news = ticker_obj.news
        except:
            pass
            
        # Filter valid news items
        valid_news = []
        for item in news[:limit] if news else []:
            if isinstance(item, dict) and 'title' in item:
                # Make sure all required fields exist
                if 'publisher' not in item or not item['publisher']:
                    item['publisher'] = 'Yahoo Finance'
                if 'link' not in item or not item['link']:
                    item['link'] = f"https://finance.yahoo.com/quote/{ticker}"
                if 'providerPublishTime' not in item:
                    item['providerPublishTime'] = int(datetime.now().timestamp())
                
                valid_news.append(item)
        
        # If no news found, create generic entries based on stock information
        if not valid_news:
            info = ticker_obj.info
            company_name = info.get('shortName', ticker)
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Create some generic news items
            current_time = int(datetime.now().timestamp())
            
            generic_news = [
                {
                    'title': f"{company_name} Latest Stock Performance and Analysis",
                    'publisher': "Market Analysis",
                    'link': f"https://finance.yahoo.com/quote/{ticker}",
                    'providerPublishTime': current_time - 86400
                },
                {
                    'title': f"Industry Outlook: {industry} Trends and {company_name}'s Position",
                    'publisher': "Industry Watch",
                    'link': f"https://finance.yahoo.com/quote/{ticker}",
                    'providerPublishTime': current_time - 172800
                },
                {
                    'title': f"{sector} Sector Analysis: Opportunities and Challenges",
                    'publisher': "Sector Insights",
                    'link': f"https://finance.yahoo.com/quote/{ticker}",
                    'providerPublishTime': current_time - 259200
                }
            ]
            
            valid_news = generic_news[:limit]
            
        return valid_news
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_market_indices():
    """Get current market indices data"""
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^RUT": "Russell 2000",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225"
    }
    
    result = []
    for symbol, name in indices.items():
        try:
            data = yf.Ticker(symbol).history(period="2d")
            if len(data) >= 2:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                result.append({
                    "Index": name,
                    "Value": f"{current:.2f}",
                    "Change": f"{change:+.2f}%"
                })
        except:
            # If there's an error, add with empty data
            result.append({
                "Index": name,
                "Value": "N/A",
                "Change": "N/A"
            })
    
    return pd.DataFrame(result)

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_market_movers():
    """Get top market gainers and losers"""
    # This is a simplified implementation - in a real app, you would use a financial API
    # For now we'll use a sample of stocks and simulate market movement
    
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ", 
              "PG", "WMT", "UNH", "HD", "BAC", "PFE", "MA", "DIS", "NFLX", "ADBE"]
    
    all_data = []
    for ticker in stocks:
        try:
            data = yf.Ticker(ticker).history(period="2d")
            if len(data) >= 2:
                current = data['Close'].iloc[-1]
                prev = data['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                info = yf.Ticker(ticker).info
                name = info.get('shortName', ticker)
                
                all_data.append({
                    "Symbol": ticker,
                    "Name": name,
                    "Price": f"${current:.2f}",
                    "Change": f"{change:+.2f}%",
                    "ChangeValue": change
                })
        except:
            pass
    
    df = pd.DataFrame(all_data)
    
    if len(df) > 0:
        gainers = df.nlargest(5, 'ChangeValue')
        losers = df.nsmallest(5, 'ChangeValue')
        return gainers, losers
    else:
        # Fallback with dummy data
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_economic_calendar():
    """Get upcoming economic events (simplified example)"""
    # In a real app, you would get this from an API
    # For now, we'll create dummy data
    
    today = datetime.now()
    
    events = [
        {"Date": (today + timedelta(days=2)).strftime("%b %d, %Y"), 
         "Event": "Unemployment Rate", "Previous": "3.8%", "Forecast": "3.7%"},
        {"Date": (today + timedelta(days=3)).strftime("%b %d, %Y"), 
         "Event": "Fed Chair Speech", "Previous": "N/A", "Forecast": "N/A"},
        {"Date": (today + timedelta(days=5)).strftime("%b %d, %Y"), 
         "Event": "GDP Growth", "Previous": "2.1%", "Forecast": "2.3%"},
        {"Date": (today + timedelta(days=7)).strftime("%b %d, %Y"), 
         "Event": "CPI Data", "Previous": "3.2%", "Forecast": "3.0%"}
    ]
    
    return pd.DataFrame(events)

# ---------------- Simplified Price Chart Functions ---------------- #
def create_price_chart(df, ticker, include_volume=True):
    """
    Create a price chart with lines connecting data points.
    This version avoids Series ambiguity errors completely.
    """
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    try:
        # Check if we have any data
        if df.empty:
            raise ValueError("No data available for this ticker and time period")
        
        # Create figure
        fig = go.Figure()
        
        # Add price line with markers to show both the line and the data points
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines+markers',  # Show both lines and markers
            name='Price',
            line=dict(color='green', width=2),
            marker=dict(size=6, color='green')
        ))
        
        # Ensure the y-axis ranges are scalars, not Series
        if len(df) > 0:
            min_price = float(df['Close'].min()) * 0.99  # 1% below minimum
            max_price = float(df['Close'].max()) * 1.01  # 1% above maximum
        else:
            min_price = 0
            max_price = 100
        
        # Calculate percentage change for the period - using explicit scalar conversion
        if len(df) > 1:
            # Convert to float explicitly to avoid Series
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            
            # Calculate change as scalar values
            pct_change = ((end_price - start_price) / start_price) * 100
            change_text = f"Change: {pct_change:.2f}%"
            
            # Add annotation showing percentage change
            fig.add_annotation(
                x=df.index[-1],
                y=end_price,
                text=change_text,
                showarrow=True,
                arrowhead=1,
                ax=50,
                ay=-40,
                font=dict(color="black", size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Price History',
            yaxis=dict(
                title='Price ($)',
                range=[min_price, max_price]  # Set y-axis range to focus on price data
            ),
            template='plotly_white',
            height=500,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified'  # Show all points for a given x-coordinate
        )
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                type="date"
            )
        )
        
        # Add volume as a separate subplot if requested - with safer checking
        if include_volume and 'Volume' in df.columns:
            # Check if Volume column has any non-null values before proceeding
            has_volume_data = not df['Volume'].isnull().all()
            
            if has_volume_data:
                # Create a secondary y-axis for volume
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker=dict(color='rgba(0, 0, 0, 0.2)'),
                    yaxis='y2'
                ))
                
                # Update layout to include secondary y-axis
                fig.update_layout(
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    )
                )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating price chart: {str(e)}")
        
        # Return a simple error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Unable to create chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15)
        )
        fig.update_layout(height=400)
        return fig

def create_comparison_chart(tickers, period="1y", names=None):
    """
    Create a chart comparing multiple stocks.
    """
    try:
        if not tickers:
            return None
            
        # Get data for all tickers
        data = {}
        for i, ticker in enumerate(tickers):
            df = get_stock_data(ticker, period=period)
            if not df.empty:
                # Normalize to percentage change from start
                start_price = df['Close'].iloc[0]
                df['pct_change'] = ((df['Close'] - start_price) / start_price) * 100
                
                # Use provided name or ticker symbol
                name = names[i] if names and i < len(names) else ticker
                data[name] = df['pct_change']
        
        if not data:
            return None
            
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add a line for each ticker
        colors = ['#1E88E5', '#FF5722', '#4CAF50', '#9C27B0', '#FFC107', '#795548', '#607D8B']
        
        for i, (name, values) in enumerate(comparison_df.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=values.index,
                y=values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title='Percentage Change Comparison',
            yaxis_title='% Change',
            xaxis_title='Date',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

# ---------------- Portfolio Analysis Functions ---------------- #
def calculate_portfolio_metrics(portfolio_df):
    """Calculate key portfolio metrics"""
    if portfolio_df.empty:
        return None
    
    try:
        # Ensure we have the necessary columns
        required_cols = ['ticker', 'quantity', 'price']
        if not all(col in portfolio_df.columns for col in required_cols):
            return None
        
        # Calculate value and weights
        portfolio_df['value'] = portfolio_df['quantity'] * portfolio_df['price']
        total_value = portfolio_df['value'].sum()
        portfolio_df['weight'] = portfolio_df['value'] / total_value
        
        # Calculate sector allocation if sector data exists
        if 'sector' in portfolio_df.columns:
            sector_allocation = portfolio_df.groupby('sector')['value'].sum().reset_index()
            sector_allocation['percentage'] = (sector_allocation['value'] / total_value) * 100
            sector_allocation = sector_allocation.sort_values('percentage', ascending=False)
        else:
            sector_allocation = pd.DataFrame(columns=['sector', 'value', 'percentage'])
        
        # Calculate concentration metrics
        top_holding_weight = portfolio_df['weight'].max() * 100
        top_3_weight = portfolio_df.nlargest(3, 'weight')['weight'].sum() * 100
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        hhi = (portfolio_df['weight'] ** 2).sum()
        
        # Return metrics
        metrics = {
            "total_value": total_value,
            "num_holdings": len(portfolio_df),
            "top_holding": portfolio_df.loc[portfolio_df['weight'].idxmax(), 'ticker'],
            "top_holding_weight": top_holding_weight,
            "top_3_weight": top_3_weight,
            "hhi": hhi,
            "sector_allocation": sector_allocation
        }
        
        return metrics
    
    except Exception as e:
        st.error(f"Error calculating portfolio metrics: {str(e)}")
        return None

# ---------------- Data Processing Functions ---------------- #
def extract_ticker_from_text(text):
    """Extract potential stock tickers from text"""
    # Pattern for stock symbols (typically 1-5 uppercase letters)
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    
    # Common words that shouldn't be considered tickers
    common_words = {
        'A', 'I', 'AM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IN', 'IS', 'IT', 'ME',
        'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'TO', 'UP', 'US', 'WE', 'CEO', 'CFO', 'CTO', 'COO',
        'IPO', 'GDP', 'CPI', 'FBI', 'SEC', 'FED', 'USA', 'CPU', 'RAM', 'ROM', 'API', 'Q1', 'Q2', 
        'Q3', 'Q4', 'AI', 'ML', 'USD', 'ETF'
    }
    
    # Find all potential tickers
    potential_tickers = re.findall(ticker_pattern, text)
    
    # Filter out common words
    tickers = [ticker for ticker in potential_tickers if ticker not in common_words]
    
    return tickers

def format_large_number(num):
    """Format large numbers with K, M, B, T suffixes"""
    if num is None:
        return "N/A"
    
    # Handle pandas Series
    if hasattr(num, 'iloc') and hasattr(num, 'values'):
        if len(num) > 0:
            num = num.iloc[0]  # Convert Series to scalar
        else:
            return "N/A"
    
    # Handle string inputs
    if isinstance(num, str):
        try:
            num = float(num.replace(',', ''))
        except:
            return num
    
    # Ensure we have a numeric value
    try:
        num_float = float(num)
    except (ValueError, TypeError):
        return "N/A"
    
    # Format based on magnitude
    if num_float >= 1_000_000_000_000:
        return f"{num_float / 1_000_000_000_000:.2f}T"
    elif num_float >= 1_000_000_000:
        return f"{num_float / 1_000_000_000:.2f}B"
    elif num_float >= 1_000_000:
        return f"{num_float / 1_000_000:.2f}M"
    elif num_float >= 1_000:
        return f"{num_float / 1_000:.2f}K"
    else:
        return f"{num_float:.2f}"

def parse_financial_value(value_str):
    """Parse a financial value from a string with K, M, B, T suffixes"""
    if not value_str or not isinstance(value_str, str):
        return None
    
    value_str = value_str.strip().upper()
    
    # Remove currency symbols and commas
    for char in ['$', '€', '£', '¥', ',']:
        value_str = value_str.replace(char, '')
    
    # Try to parse with suffix
    suffix_map = {
        'T': 1_000_000_000_000,
        'B': 1_000_000_000,
        'M': 1_000_000,
        'K': 1_000
    }
    
    for suffix, multiplier in suffix_map.items():
        if suffix in value_str:
            try:
                return float(value_str.replace(suffix, '')) * multiplier
            except ValueError:
                return None
    
    # Try to parse as a plain number
    try:
        return float(value_str)
    except ValueError:
        return None