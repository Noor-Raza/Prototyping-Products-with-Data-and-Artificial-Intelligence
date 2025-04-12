import streamlit as st
import datetime

def apply_custom_theme():
    """Apply custom theme and styling to the app"""
    # Custom CSS for professional look and feel
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #1565C0;
        --accent-color: #FFA000;
        --background-color: #FAFAFA;
        --text-color: #212121;
        --light-gray: #EEEEEE;
        --card-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* General styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Cards and containers */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: var(--card-shadow);
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .card-header {
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* Custom styled metrics */
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 16px;
        color: #666;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    /* Banner styling */
    .banner {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: var(--card-shadow);
    }
    
    /* Custom button styling */
    .custom-button {
        background-color: var(--primary-color);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
        text-align: center;
        display: inline-block;
        margin: 5px;
    }
    
    .custom-button:hover {
        background-color: #1B5E20;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .custom-button-secondary {
        background-color: var(--secondary-color);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
        text-align: center;
        display: inline-block;
        margin: 5px;
    }
    
    .custom-button-secondary:hover {
        background-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .custom-button-accent {
        background-color: var(--accent-color);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
        text-align: center;
        display: inline-block;
        margin: 5px;
    }
    
    .custom-button-accent:hover {
        background-color: #FF8F00;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F5F5F5;
    }
    
    /* Custom sidebar menu */
    .sidebar-menu {
        margin-top: 20px;
    }
    
    .sidebar-menu-item {
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
    }
    
    .sidebar-menu-item:hover {
        background-color: #E0E0E0;
    }
    
    .sidebar-menu-item.active {
        background-color: var(--primary-color);
        color: white;
    }
    
    .sidebar-menu-icon {
        margin-right: 10px;
        font-size: 20px;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        margin-bottom: 20px;
    }
    
    .dataframe th {
        background-color: var(--primary-color);
        color: white;
        padding: 12px !important;
        text-align: left;
        font-weight: 500;
    }
    
    .dataframe td {
        padding: 10px !important;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:hover {
        background-color: var(--light-gray);
    }
    
    /* Footer styling */
    footer {
        background-color: #333;
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-top: 30px;
    }
    
    /* Dashboard cards styling */
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        height: 100%;
        transition: transform 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
    }
    
    .dashboard-card-header {
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 18px;
        color: var(--primary-color);
    }
    
    /* Status indicators */
    .status-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .status-negative {
        color: #F44336;
        font-weight: bold;
    }
    
    .status-neutral {
        color: #9E9E9E;
        font-weight: bold;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Insights panel */
    .insights-panel {
        background-color: #F5F5F5;
        border-left: 5px solid var(--primary-color);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Alert boxes */
    .alert-info {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .alert-success {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .alert-warning {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .alert-danger {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* Portfolio cards */
    .portfolio-holding {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: var(--card-shadow);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .portfolio-ticker {
        font-weight: bold;
        font-size: 18px;
    }
    
    .portfolio-value {
        font-size: 16px;
    }
    
    .portfolio-change-positive {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .portfolio-change-negative {
        color: #F44336;
        font-weight: bold;
    }
    
    /* Interactive elements */
    .clickable {
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .clickable:hover {
        opacity: 0.8;
    }
    
    /* Watch list items */
    .watchlist-item {
        display: flex;
        justify-content: space-between;
        padding: 10px;
        border-bottom: 1px solid #eee;
        align-items: center;
    }
    
    .watchlist-ticker {
        font-weight: bold;
    }
    
    /* Analysis section styling */
    .analysis-section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: var(--card-shadow);
    }
    
    .analysis-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }
    
    .analysis-title {
        font-size: 20px;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    /* Search bar styling */
    .search-container {
        display: flex;
        margin-bottom: 20px;
    }
    
    .search-input {
        flex-grow: 1;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px 0 0 5px;
        font-size: 16px;
    }
    
    .search-button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 0 5px 5px 0;
        cursor: pointer;
    }
    
    /* News card styling */
    .news-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease;
    }
    
    .news-card:hover {
        transform: translateY(-5px);
    }
    
    .news-source {
        color: #666;
        font-size: 14px;
    }
    
    .news-date {
        color: #666;
        font-size: 14px;
    }
    
    .news-title {
        font-size: 18px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .news-summary {
        font-size: 16px;
        margin-bottom: 10px;
    }
    
    /* Sentiment indicators */
    .sentiment-positive {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
    }
    
    .sentiment-negative {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 10px;
        border-radius: 5px;
    }
    
    .sentiment-neutral {
        background-color: #ECEFF1;
        border-left: 5px solid #9E9E9E;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header(title, subtitle):
    """Display a well-styled header with title and subtitle"""
    st.markdown(f"""
    <div class="banner">
        <h1 style="margin:0;color:white;">{title}</h1>
        <p style="margin:0;font-size:18px;opacity:0.9;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def display_metric(title, value, change=None, prefix="", suffix=""):
    """Display a metric with optional change indicator"""
    change_html = ""
    if change is not None:
        try:
            change_value = float(change.replace("%", "").replace("+", ""))
            color = "status-positive" if change_value >= 0 else "status-negative"
            icon = "↑" if change_value >= 0 else "↓"
            change_html = f'<span class="{color}">{icon} {change}</span>'
        except:
            change_html = f'<span>{change}</span>'
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix} {change_html}</div>
    </div>
    """, unsafe_allow_html=True)

def display_card(title, content, icon=None):
    """Display content in a card with consistent styling"""
    icon_html = f'<span style="margin-right:10px;">{icon}</span>' if icon else ''
    
    st.markdown(f"""
    <div class="dashboard-card">
        <div class="dashboard-card-header">{icon_html}{title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def display_alert(message, alert_type="info"):
    """Display a colored alert box
    
    Parameters:
    - message: Text to display
    - alert_type: One of "info", "success", "warning", "danger"
    """
    st.markdown(f"""
    <div class="alert-{alert_type}">
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display a professional footer"""
    current_year = datetime.datetime.now().year
    
    st.markdown("""
    <footer>
        <p>Stock Analysis Dashboard © 2025</p>
        <p style="font-size:12px;">Powered by yFinance, Prophet, and Cohere AI</p>
        <p style="font-size:10px;">Disclaimer: This application is for educational purposes only. Not financial advice.</p>
    </footer>
    """, unsafe_allow_html=True)

def enhance_sidebar():
    """Add custom styling and elements to sidebar"""
    st.sidebar.markdown("""
    <div style="text-align:center;margin-bottom:20px;">
        <h2 style="color:#2E7D32;">Stock Analysis</h2>
        <p>Professional trading insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a datetime display
    now = datetime.datetime.now()
    date_str = now.strftime("%b %d, %Y")
    time_str = now.strftime("%H:%M:%S")
    
    st.sidebar.markdown(f"""
    <div style="background-color:#f0f0f0;padding:10px;border-radius:5px;margin-bottom:20px;text-align:center;">
        <div style="font-size:18px;font-weight:bold;">{date_str}</div>
        <div style="font-size:14px;">{time_str}</div>
    </div>
    """, unsafe_allow_html=True)

def display_stock_card(ticker, name, price, change, sector=None):
    """Display a stock card with key information"""
    change_value = float(change.replace("%", "").replace("+", ""))
    change_class = "portfolio-change-positive" if change_value >= 0 else "portfolio-change-negative"
    change_icon = "↑" if change_value >= 0 else "↓"
    
    sector_html = f'<div style="font-size:14px;color:#666;">{sector}</div>' if sector else ''
    
    st.markdown(f"""
    <div class="portfolio-holding">
        <div>
            <div class="portfolio-ticker">{ticker}</div>
            <div style="font-size:14px;">{name}</div>
            {sector_html}
        </div>
        <div style="text-align:right;">
            <div class="portfolio-value">${price}</div>
            <div class="{change_class}">{change_icon} {change}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_menu(items, active_item=None):
    """Display a custom menu in the sidebar
    
    Parameters:
    - items: List of dicts with keys 'icon', 'label', and 'url'
    - active_item: Currently active item label
    """
    st.sidebar.markdown("<div class='sidebar-menu'>", unsafe_allow_html=True)
    
    for item in items:
        icon = item.get('icon', '')
        label = item.get('label', '')
        url = item.get('url', '#')
        active_class = "active" if label == active_item else ""
        
        st.sidebar.markdown(f"""
        <a href="{url}" style="text-decoration:none;color:inherit;">
            <div class="sidebar-menu-item {active_class}">
                <span class="sidebar-menu-icon">{icon}</span>
                <span>{label}</span>
            </div>
        </a>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

def display_watchlist(stocks, prices, changes):
    """Display a watchlist of stocks with current prices and changes"""
    for ticker, price, change in zip(stocks, prices, changes):
        try:
            change_value = float(change.replace("%", "").replace("+", ""))
            change_class = "portfolio-change-positive" if change_value >= 0 else "portfolio-change-negative"
            change_icon = "↑" if change_value >= 0 else "↓"
        except ValueError:
            change_class = ""
            change_icon = "-"
        
        st.markdown(f"""
        <div class="watchlist-item">
            <span class="watchlist-ticker">{ticker}</span>
            <span>${price}</span>
            <span class="{change_class}">{change_icon} {change}</span>
        </div>
        """, unsafe_allow_html=True)

def display_news_card(title, source, date, summary, sentiment=None):
    """Display a news card with title, source, date, and summary"""
    sentiment_class = ""
    sentiment_html = ""
    
    if sentiment:
        if sentiment.lower() == "positive" or sentiment.lower() == "bullish":
            sentiment_class = "sentiment-positive"
            emoji = "📈"
            label = "Bullish"
        elif sentiment.lower() == "negative" or sentiment.lower() == "bearish":
            sentiment_class = "sentiment-negative"
            emoji = "📉"
            label = "Bearish"
        else:
            sentiment_class = "sentiment-neutral"
            emoji = "⚖️"
            label = "Neutral"
            
        sentiment_html = f"""
        <div class="{sentiment_class}" style="margin-top:10px;">
            <span>{emoji} {label} sentiment</span>
        </div>
        """
    
    st.markdown(f"""
    <div class="news-card">
        <div style="display:flex;justify-content:space-between;">
            <span class="news-source">{source}</span>
            <span class="news-date">{date}</span>
        </div>
        <div class="news-title">{title}</div>
        <div class="news-summary">{summary}</div>
        {sentiment_html}
    </div>
    """, unsafe_allow_html=True)