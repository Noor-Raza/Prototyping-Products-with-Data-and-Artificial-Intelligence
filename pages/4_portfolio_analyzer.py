import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import os
import json
import re
import io
from PIL import Image
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Interactive Portfolio Optimizer", layout="wide")

# Initialize OpenAI API
# Try different ways to import OpenAI based on package version
try:
    from openai import OpenAI
    use_new_openai = True
except ImportError:
    import openai
    use_new_openai = False

# Get API key
api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
os.environ["OPENAI_API_KEY"] = api_key

if use_new_openai:
    try:
        openai_client = OpenAI(api_key=api_key)
    except:
        openai_client = None
else:
    openai = openai
    openai.api_key = api_key
    openai_client = openai

st.title("💼 Interactive Portfolio Optimizer")
st.markdown("Analyze and optimize your portfolio using AI-powered recommendations")

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'name', 'quantity', 'price', 'sector', 'industry'])

if 'optimization_history' not in st.session_state:
    st.session_state.optimization_history = []

# Helper Functions

# Function to get stock data
@st.cache_data(ttl=3600) 
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get the current price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # Get sector info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        name = info.get('longName', ticker)
        
        return {
            'name': name,
            'price': current_price,
            'sector': sector,
            'industry': industry
        }
    except Exception as e:
        st.warning(f"Couldn't fetch data for {ticker}: {e}")
        return {
            'name': ticker,
            'price': 0,
            'sector': 'Unknown',
            'industry': 'Unknown'
        }

# Process portfolio data
def process_portfolio(portfolio_df):
    if portfolio_df.empty:
        return portfolio_df, {}
    
    # Calculate value and weights
    portfolio_df['value'] = portfolio_df['quantity'] * portfolio_df['price']
    total_value = portfolio_df['value'].sum()
    portfolio_df['weight'] = portfolio_df['value'] / total_value
    
    # Calculate sector allocation
    sector_data = portfolio_df.groupby('sector')['value'].sum().reset_index()
    sector_data['percentage'] = sector_data['value'] / total_value * 100
    
    # Calculate basic metrics
    metrics = {
        'total_value': total_value,
        'num_holdings': len(portfolio_df),
        'top_holding': portfolio_df.loc[portfolio_df['weight'].idxmax(), 'ticker'],
        'top_holding_weight': portfolio_df['weight'].max() * 100,
        'sector_allocation': sector_data
    }
    
    return portfolio_df, metrics

# Input method selection with tabs
tab1, tab2 = st.tabs(["Manual Entry", "Sample Portfolio"])

# Manual portfolio entry
with tab1:
    st.subheader("Enter Your Holdings")
    
    # Form for adding new positions
    with st.form("add_position_form"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            ticker = st.text_input("Ticker Symbol").upper()
        
        with col2:
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0, value=1.0)
        
        with col3:
            custom_price = st.checkbox("Custom Price?")
            
        if custom_price:
            price = st.number_input("Price ($)", min_value=0.0, step=0.01)
        else:
            price = 0  # Will be fetched from yfinance
            
        submit = st.form_submit_button("Add Position")
        
        if submit and ticker and quantity > 0:
            # Get stock data if no custom price
            if not custom_price:
                stock_data = get_stock_data(ticker)
                price = stock_data['price']
                name = stock_data['name']
                sector = stock_data['sector']
                industry = stock_data['industry']
            else:
                # Try to get name and sector, but use defaults if that fails
                try:
                    stock_data = get_stock_data(ticker)
                    name = stock_data['name']
                    sector = stock_data['sector'] 
                    industry = stock_data['industry']
                except:
                    name = ticker
                    sector = "Unknown"
                    industry = "Unknown"
            
            # Check if ticker already exists
            if ticker in st.session_state.portfolio['ticker'].values:
                # Update existing position
                idx = st.session_state.portfolio[st.session_state.portfolio['ticker'] == ticker].index[0]
                st.session_state.portfolio.at[idx, 'quantity'] = quantity
                st.session_state.portfolio.at[idx, 'price'] = price
            else:
                # Add new position
                new_position = pd.DataFrame({
                    'ticker': [ticker],
                    'name': [name],
                    'quantity': [quantity],
                    'price': [price],
                    'sector': [sector],
                    'industry': [industry]
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
                
            st.success(f"Added {quantity} shares of {ticker} at ${price:.2f}")
            st.rerun()

# Sample portfolio
with tab2:
    if st.button("Load Sample Diversified Portfolio"):
        sample_portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'PG', 'XOM', 'VZ', 'PFE'],
            'quantity': [10, 5, 2, 3, 15, 12, 20, 30, 25, 40]
        })
        
        # Get current prices and info
        for idx, row in sample_portfolio.iterrows():
            stock_data = get_stock_data(row['ticker'])
            sample_portfolio.at[idx, 'name'] = stock_data['name']
            sample_portfolio.at[idx, 'price'] = stock_data['price']
            sample_portfolio.at[idx, 'sector'] = stock_data['sector']
            sample_portfolio.at[idx, 'industry'] = stock_data['industry']
        
        st.session_state.portfolio = sample_portfolio
        st.rerun()
    
    if st.button("Load Tech-Heavy Portfolio"):
        tech_portfolio = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'AMD', 'TSLA', 'META', 'CRM', 'INTC'],
            'quantity': [15, 10, 5, 8, 20, 30, 10, 12, 15, 25]
        })
        
        # Get current prices and info
        for idx, row in tech_portfolio.iterrows():
            stock_data = get_stock_data(row['ticker'])
            tech_portfolio.at[idx, 'name'] = stock_data['name']
            tech_portfolio.at[idx, 'price'] = stock_data['price']
            tech_portfolio.at[idx, 'sector'] = stock_data['sector']
            tech_portfolio.at[idx, 'industry'] = stock_data['industry']
        
        st.session_state.portfolio = tech_portfolio
        st.rerun()

# Display current portfolio
if not st.session_state.portfolio.empty:
    st.subheader("Your Portfolio")
    
    # Process portfolio data
    portfolio_df, metrics = process_portfolio(st.session_state.portfolio)
    
    # Display portfolio table
    display_df = portfolio_df[['ticker', 'name', 'quantity', 'price', 'value', 'weight', 'sector']].copy()
    display_df['price'] = display_df['price'].map('${:,.2f}'.format)
    display_df['value'] = display_df['value'].map('${:,.2f}'.format)
    display_df['weight'] = display_df['weight'].map('{:.2%}'.format)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Portfolio metrics and visualizations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Value", f"${metrics['total_value']:,.2f}")
    
    with col2:
        st.metric("Number of Holdings", metrics['num_holdings'])
    
    with col3:
        st.metric("Top Holding", f"{metrics['top_holding']} ({metrics['top_holding_weight']:.2f}%)")
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Holdings allocation chart
        fig1 = px.pie(
            portfolio_df,
            values='value',
            names='ticker',
            title='Portfolio Allocation by Stock',
            hover_data=['name'],
            labels={'value': 'Value ($)'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with chart_col2:
        # Sector allocation chart
        if 'sector_allocation' in metrics and not metrics['sector_allocation'].empty:
            fig2 = px.bar(
                metrics['sector_allocation'],
                x='sector',
                y='percentage',
                title='Sector Allocation',
                labels={'percentage': 'Allocation (%)', 'sector': 'Sector'},
                color='sector'
            )
            st.plotly_chart(fig2, use_container_width=True)

# Interactive optimization section
if not st.session_state.portfolio.empty:
    st.markdown("---")
    st.subheader("💡 Interactive Portfolio Optimization")
    
    # Risk preference slider
    st.write("What is your investment risk tolerance?")
    risk_preference = st.slider(
        "Risk Tolerance",
        min_value=1,
        max_value=10,
        value=5,
        help="1 = Very Conservative, 10 = Very Aggressive"
    )
    
    # Investment goals
    st.write("What are your investment goals? (Select all that apply)")
    col1, col2 = st.columns(2)
    with col1:
        income = st.checkbox("Income (dividends, regular payments)")
        growth = st.checkbox("Growth (capital appreciation)")
        preservation = st.checkbox("Capital Preservation (protect principal)")
    
    with col2:
        tax_efficiency = st.checkbox("Tax Efficiency")
        esg = st.checkbox("ESG/Sustainable Investing")
        sector_focus = st.checkbox("Specific Sector Focus")
    
    # More specific goals if sector focus is checked
    if sector_focus:
        preferred_sectors = st.multiselect(
            "Which sectors do you want to focus on?",
            options=["Technology", "Healthcare", "Consumer", "Financials", "Energy", "Utilities", "Industrial", "Materials", "Real Estate", "Communication Services"],
            default=[]
        )
    else:
        preferred_sectors = []
    
    # Time horizon
    time_horizon = st.radio(
        "Investment Time Horizon",
        options=["Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (5+ years)"]
    )
    
    # Create a list of selected goals
    selected_goals = []
    if income:
        selected_goals.append("Income")
    if growth:
        selected_goals.append("Growth")
    if preservation:
        selected_goals.append("Capital Preservation")
    if tax_efficiency:
        selected_goals.append("Tax Efficiency")
    if esg:
        selected_goals.append("ESG/Sustainable Investing")
    if sector_focus:
        selected_goals.append("Sector Focus")

# Function to generate prompts with increasing complexity
def generate_prompt(portfolio_df, metrics, risk_preference, goals, sectors, time_horizon, complexity=1):
    """Generate a prompt for the OpenAI API with varying complexity"""
    
    # Convert portfolio data to a simplified format
    holdings = []
    for _, row in portfolio_df.iterrows():
        holdings.append({
            "ticker": row['ticker'],
            "name": row['name'],
            "value": float(row['value']),
            "weight": float(row['weight']),
            "sector": row['sector']
        })
    
    # Convert sector allocation to a simplified format
    sectors_data = []
    for _, row in metrics['sector_allocation'].iterrows():
        sectors_data.append({
            "sector": row['sector'],
            "percentage": float(row['percentage'])
        })
        
    # Basic prompt
    if complexity == 1:
        prompt = f"""
        Please analyze this investment portfolio and suggest optimization changes:
        
        Portfolio total value: ${metrics['total_value']:.2f}
        Number of holdings: {metrics['num_holdings']}
        Risk preference (1-10 scale): {risk_preference}
        Investment goals: {', '.join(goals)}
        Time horizon: {time_horizon}
        
        Holdings:
        {json.dumps(holdings, indent=2)}
        
        Sector allocation:
        {json.dumps(sectors_data, indent=2)}
        
        Please provide specific recommendations to optimize this portfolio.
        """
    
    # More detailed prompt with specific instructions on response format
    elif complexity == 2:
        prompt = f"""
        Please analyze this investment portfolio and suggest optimization changes:
        
        Portfolio total value: ${metrics['total_value']:.2f}
        Number of holdings: {metrics['num_holdings']}
        Risk preference (1-10 scale): {risk_preference} (1 = Very Conservative, 10 = Very Aggressive)
        Investment goals: {', '.join(goals)}
        Time horizon: {time_horizon}
        
        Holdings:
        {json.dumps(holdings, indent=2)}
        
        Sector allocation:
        {json.dumps(sectors_data, indent=2)}
        
        Please provide your recommendations in a structured format with these sections:
        1. Analysis Summary - Brief overview of the portfolio
        2. Recommendations - Specific actions to take
        3. Target Allocation - Suggested target asset allocation
        
        Be specific about which stocks to buy, sell, or hold and why.
        """
        
    # Highly detailed prompt with explicit JSON structure requirements
    else:
        preferred_sectors_text = ""
        if sectors:
            preferred_sectors_text = f"Preferred sectors: {', '.join(sectors)}"
            
        prompt = f"""
        Please analyze this investment portfolio and provide recommendations in JSON format:
        
        Portfolio total value: ${metrics['total_value']:.2f}
        Number of holdings: {metrics['num_holdings']}
        Risk preference (1-10 scale): {risk_preference} (1 = Very Conservative, 10 = Very Aggressive)
        Investment goals: {', '.join(goals)}
        Time horizon: {time_horizon}
        {preferred_sectors_text}
        
        Holdings:
        {json.dumps(holdings, indent=2)}
        
        Sector allocation:
        {json.dumps(sectors_data, indent=2)}
        
        Return your analysis in the following JSON format ONLY:
        {{
            "summary": "Brief overall assessment of the portfolio",
            "risk_assessment": {{
                "current_risk_level": "Current portfolio risk level (Conservative/Moderate/Aggressive)",
                "target_risk_level": "Target risk level based on preference of {risk_preference}/10",
                "alignment": "How well current portfolio aligns with target risk"
            }},
            "recommendations": [
                {{
                    "ticker": "Ticker symbol",
                    "action": "buy|sell|hold",
                    "percentage_change": "Suggested percentage to buy or sell",
                    "reasoning": "Brief explanation"
                }},
                ...
            ],
            "target_allocation": {{
                "sectors": [
                    {{
                        "sector": "Sector name",
                        "current_percentage": XX.X,
                        "target_percentage": XX.X,
                        "change": "increase|decrease|maintain"
                    }},
                    ...
                ]
            }}
        }}
        
        Ensure your response is valid JSON and properly structured - nothing else.
        """
        
    return prompt

# Function to analyze portfolio with OpenAI with refinement
def analyze_portfolio_with_refinement(portfolio_df, metrics, risk_preference, goals, sectors, time_horizon):
    """Analyze portfolio with progressive prompt refinement if needed"""
    
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar to use this feature.")
        return None
        
    try:
        # Start with basic prompt (complexity level 1)
        basic_prompt = generate_prompt(portfolio_df, metrics, risk_preference, goals, sectors, time_horizon, complexity=1)
        
        # First try with basic prompt
        if use_new_openai:
            # For newer OpenAI package
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis."},
                    {"role": "user", "content": basic_prompt}
                ],
                temperature=0.7,
            )
            initial_response = response.choices[0].message.content
        else:
            # For older OpenAI package
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis."},
                    {"role": "user", "content": basic_prompt}
                ],
                temperature=0.7,
            )
            initial_response = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Check if the response contains valid JSON
            json_match = re.search(r'{.*}', initial_response, re.DOTALL)
            if json_match:
                # Try to parse the JSON portion
                parsed_response = json.loads(json_match.group(0))
                if isinstance(parsed_response, dict) and any(key in parsed_response for key in ['recommendations', 'summary', 'risk_assessment']):
                    # If we got a good response, return it with the prompt used
                    return {
                        "response": parsed_response,
                        "prompt_used": basic_prompt,
                        "refinement_level": 1
                    }
            # If we get here, JSON extraction failed or was incomplete
            raise ValueError("Could not extract valid JSON")
        except:
            # First refinement attempt with more detailed prompt
            refined_prompt = generate_prompt(portfolio_df, metrics, risk_preference, goals, sectors, time_horizon, complexity=2)
            
            # Try with more detailed prompt
            if use_new_openai:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis."},
                        {"role": "user", "content": refined_prompt}
                    ],
                    temperature=0.7,
                )
                refined_response = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis."},
                        {"role": "user", "content": refined_prompt}
                    ],
                    temperature=0.7,
                )
                refined_response = response.choices[0].message.content
            
            # Try to extract JSON from the refined response
            try:
                # Check if the response contains valid JSON
                json_match = re.search(r'{.*}', refined_response, re.DOTALL)
                if json_match:
                    # Try to parse the JSON portion
                    parsed_response = json.loads(json_match.group(0))
                    if isinstance(parsed_response, dict):
                        # If we got a better response, return it with the prompt used
                        return {
                            "response": parsed_response,
                            "prompt_used": refined_prompt,
                            "refinement_level": 2,
                            "text_response": refined_response  # Keep the text response as fallback
                        }
                # If we get here, JSON extraction failed again
                raise ValueError("Could not extract valid JSON from refined prompt")
            except:
                # Final attempt with explicitly structured JSON prompt
                final_prompt = generate_prompt(portfolio_df, metrics, risk_preference, goals, sectors, time_horizon, complexity=3)
                
                # Try with structured JSON prompt
                if use_new_openai:
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis. You must return a valid JSON response."},
                            {"role": "user", "content": final_prompt}
                        ],
                        temperature=0.5,  # Lower temperature for more structured output
                    )
                    final_response = response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a professional financial advisor specializing in portfolio analysis. You must return a valid JSON response."},
                            {"role": "user", "content": final_prompt}
                        ],
                        temperature=0.5,  # Lower temperature for more structured output
                    )
                    final_response = response.choices[0].message.content
                
                # Final attempt to extract JSON
                try:
                    # Try different extraction methods
                    if "```json" in final_response:
                        # Extract from code block
                        json_match = re.search(r'```json\s*(.*?)\s*```', final_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            parsed_response = json.loads(json_str)
                            return {
                                "response": parsed_response,
                                "prompt_used": final_prompt,
                                "refinement_level": 3
                            }
                    else:
                        # Try to extract anything between curly braces
                        json_match = re.search(r'({.*})', final_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            parsed_response = json.loads(json_str)
                            return {
                                "response": parsed_response,
                                "prompt_used": final_prompt,
                                "refinement_level": 3
                            }
                            
                    # If all extraction attempts fail, return the text response
                    return {
                        "response": None,
                        "prompt_used": final_prompt,
                        "refinement_level": 3,
                        "text_response": final_response
                    }
                except Exception as e:
                    # Return the text as a fallback
                    return {
                        "response": None,
                        "error": str(e),
                        "prompt_used": final_prompt,
                        "refinement_level": 3,
                        "text_response": final_response
                    }
    
    except Exception as e:
        st.error(f"Error analyzing portfolio: {str(e)}")
        return None
    
# Button to analyze portfolio
if not st.session_state.portfolio.empty:
    analyze_btn = st.button("Analyze My Portfolio")
    
    if analyze_btn:
        if api_key:
            with st.spinner("Analyzing your portfolio using AI... This might take a minute."):
                result = analyze_portfolio_with_refinement(
                    portfolio_df, 
                    metrics, 
                    risk_preference,
                    selected_goals, 
                    preferred_sectors, 
                    time_horizon
                )
                
                # Add to history
                if result:
                    st.session_state.optimization_history.append({
                        "timestamp": pd.Timestamp.now(),
                        "result": result
                    })
        else:
            st.error("Please enter your OpenAI API key in the sidebar.")
    
    # Display latest optimization result
    if st.session_state.optimization_history:
        latest = st.session_state.optimization_history[-1]
        result = latest["result"]
        
        st.markdown("---")
        st.subheader("📊 Portfolio Analysis Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Risk Assessment", "Target Allocation", "Implementation Plan"])
        
        with tab1:
            # Check if we have a structured response
            if result and "response" in result and result["response"]:
                response = result["response"]
                
                # Display summary if available
                if "summary" in response:
                    st.markdown(f"""
                    <div style="background-color:#f5f5f5;padding:15px;border-radius:10px;margin-bottom:20px;">
                        <p style="font-style:italic;">{response['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display recommendations if available
                if "recommendations" in response and isinstance(response["recommendations"], list):
                    st.markdown("### Recommended Actions")
                    
                    # Create a DataFrame for the recommendations
                    recs = []
                    for item in response["recommendations"]:
                        if isinstance(item, dict):
                            ticker = item.get("ticker", "")
                            action = item.get("action", "").upper()
                            pct_change = item.get("percentage_change", "")
                            reasoning = item.get("reasoning", "")
                            
                            # Add to list for DataFrame
                            recs.append({
                                "ticker": ticker,
                                "action": action,
                                "change": pct_change,
                                "reasoning": reasoning
                            })
                    
                    if recs:
                        # Convert to DataFrame
                        recs_df = pd.DataFrame(recs)
                        
                        # Color-code actions
                        def color_action(val):
                            if val == "BUY":
                                return 'background-color: #d4edda; color: #155724'
                            elif val == "SELL":
                                return 'background-color: #f8d7da; color: #721c24'
                            else:
                                return 'background-color: #fff3cd; color: #856404'
                                
                        # Display the styled DataFrame
                        st.dataframe(recs_df.style.applymap(color_action, subset=['action']), use_container_width=True)
                    else:
                        st.info("No specific recommendations provided.")
                else:
                    # Display text response as fallback
                    if "text_response" in result:
                        st.markdown("### Recommendations")
                        st.write(result["text_response"])
            else:
                # Display text response as fallback
                if "text_response" in result:
                    st.markdown("### Recommendations")
                    st.write(result["text_response"])
        
        with tab2:
            # Display risk assessment if available
            if result and "response" in result and result["response"]:
                response = result["response"]
                
                if "risk_assessment" in response and isinstance(response["risk_assessment"], dict):
                    risk = response["risk_assessment"]
                    
                    current_risk = risk.get("current_risk_level", "Unknown")
                    target_risk = risk.get("target_risk_level", "Unknown")
                    alignment = risk.get("alignment", "Unknown")
                    
                    # Display risk information in columns
                    risk_col1, risk_col2, risk_col3 = st.columns(3)
                    
                    with risk_col1:
                        st.metric("Current Risk Level", current_risk)
                        
                    with risk_col2:
                        st.metric("Target Risk Level", target_risk)
                        
                    with risk_col3:
                        st.metric("Risk Alignment", alignment)
                        
                    # Create risk scale visualization
                    risk_levels = ["Very Conservative", "Conservative", "Moderately Conservative", 
                                "Moderate", "Moderately Aggressive", "Aggressive", "Very Aggressive"]
                    
                    # Try to map the text risk levels to numeric scale for visualization
                    def map_risk_to_scale(risk_text):
                        risk_text = risk_text.lower()
                        if "very conservative" in risk_text:
                            return 0
                        elif "conservative" in risk_text and "moderately" in risk_text:
                            return 2
                        elif "conservative" in risk_text:
                            return 1
                        elif "moderate" in risk_text and "aggressive" not in risk_text and "conservative" not in risk_text:
                            return 3
                        elif "moderately aggressive" in risk_text:
                            return 4
                        elif "aggressive" in risk_text and "very" not in risk_text and "moderately" not in risk_text:
                            return 5
                        elif "very aggressive" in risk_text:
                            return 6
                        else:
                            # Default to middle if we can't parse
                            return 3
                    
                    # Map the risk levels to the scale
                    current_risk_idx = map_risk_to_scale(current_risk)
                    target_risk_idx = map_risk_to_scale(target_risk)
                    
                    # Create visual risk scale
                    st.markdown("### Risk Scale")
                    st.markdown(f"""
                    <div style="position:relative;height:60px;background-color:#E0E0E0;border-radius:15px;margin:30px 0;">
                        <div style="display:flex;justify-content:space-between;position:absolute;width:100%;top:-25px;">
                            <span>Very Conservative</span>
                            <span>Moderate</span>
                            <span>Very Aggressive</span>
                        </div>
                        <div style="position:absolute;left:{current_risk_idx * 16.67}%;transform:translateX(-50%);top:-40px;">
                            <div style="background-color:#1E88E5;color:white;padding:5px 10px;border-radius:15px;font-size:12px;text-align:center;white-space:nowrap;">Current</div>
                            <div style="width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;border-top:10px solid #1E88E5;margin:0 auto;"></div>
                        </div>
                        <div style="position:absolute;left:{target_risk_idx * 16.67}%;transform:translateX(-50%);bottom:-40px;">
                            <div style="width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;border-bottom:10px solid #4CAF50;margin:0 auto;"></div>
                            <div style="background-color:#4CAF50;color:white;padding:5px 10px;border-radius:15px;font-size:12px;text-align:center;white-space:nowrap;">Target</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No structured risk assessment available.")
                    
                    # Show text response if available
                    if "text_response" in result:
                        st.markdown("### Risk Assessment")
                        st.write(result["text_response"])
        
        with tab3:
            # Display target allocation if available
            if result and "response" in result and result["response"]:
                response = result["response"]
                
                if "target_allocation" in response and isinstance(response["target_allocation"], dict):
                    target = response["target_allocation"]
                    
                    if "sectors" in target and isinstance(target["sectors"], list):
                        sectors = target["sectors"]
                        
                        # Create a DataFrame for target sectors
                        sector_data = []
                        for item in sectors:
                            if isinstance(item, dict):
                                sector = item.get("sector", "")
                                current = item.get("current_percentage", 0)
                                target = item.get("target_percentage", 0)
                                change = item.get("change", "")
                                
                                # Add to list for DataFrame
                                sector_data.append({
                                    "sector": sector,
                                    "current_percentage": current,
                                    "target_percentage": target,
                                    "change": change
                                })
                        
                        if sector_data:
                            # Convert to DataFrame
                            sectors_df = pd.DataFrame(sector_data)
                            
                            # Create a visualization comparing current vs target
                            try:
                                # Melt the DataFrame for easier plotting
                                sectors_long = pd.melt(
                                    sectors_df,
                                    id_vars=["sector", "change"],
                                    value_vars=["current_percentage", "target_percentage"],
                                    var_name="allocation_type",
                                    value_name="percentage"
                                )
                                
                                # Map allocation types to display names
                                sectors_long["allocation_type"] = sectors_long["allocation_type"].map({
                                    "current_percentage": "Current",
                                    "target_percentage": "Target"
                                })
                                
                                # Create the bar chart
                                fig = px.bar(
                                    sectors_long,
                                    x="sector",
                                    y="percentage",
                                    color="allocation_type",
                                    barmode="group",
                                    title="Current vs Target Sector Allocation",
                                    labels={"percentage": "Allocation (%)", "sector": "Sector", "allocation_type": ""},
                                    color_discrete_map={"Current": "#1E88E5", "Target": "#4CAF50"}
                                )
                                
                                fig.update_layout(
                                    xaxis_tickangle=-45,
                                    yaxis=dict(title="Allocation (%)"),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating sector chart: {str(e)}")
                                
                            # Display sector recommendations in a table
                            st.markdown("### Sector Allocation Recommendations")
                            
                            # Color code changes
                            def color_change(val):
                                if isinstance(val, str):
                                    if "increase" in val.lower():
                                        return 'background-color: #d4edda; color: #155724'
                                    elif "decrease" in val.lower():
                                        return 'background-color: #f8d7da; color: #721c24'
                                    else:
                                        return 'background-color: #fff3cd; color: #856404'
                                return ''
                            
                            # Format percentages
                            sectors_df["current_percentage"] = sectors_df["current_percentage"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
                            sectors_df["target_percentage"] = sectors_df["target_percentage"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
                            
                            # Display the styled DataFrame
                            st.dataframe(sectors_df.style.applymap(color_change, subset=['change']), use_container_width=True)
                        else:
                            st.info("No specific sector allocation recommendations provided.")
                    else:
                        st.info("No sector allocation data available.")
                        
                        # Show text response if available
                        if "text_response" in result:
                            st.markdown("### Target Allocation")
                            st.write(result["text_response"])
        
        # Show refinement process
        with st.expander("🔍 View Prompt Refinement Process"):
            st.markdown(f"**Refinement Level Used:** {result['refinement_level']}")
            st.markdown("**Description of Refinement Process:**")
            if result['refinement_level'] == 1:
                st.markdown("✅ The basic prompt was sufficient to get a well-structured response.")
            elif result['refinement_level'] == 2:
                st.markdown("""
                1. ❌ The initial basic prompt did not return a properly structured response.
                2. ✅ The second more detailed prompt was successful in getting a structured response.
                """)
            elif result['refinement_level'] == 3:
                st.markdown("""
                1. ❌ The initial basic prompt did not return a properly structured response.
                2. ❌ The second more detailed prompt also failed to produce a correctly structured response.
                3. ✅ The final explicit JSON-structured prompt was required to get a properly formatted response.
                """)
            
            st.markdown("**Final Prompt Used:**")
            st.code(result['prompt_used'], language="text")
            
            # If we have a JSON response, show it
            if result["response"]:
                st.markdown("**Raw JSON Response:**")
                st.code(json.dumps(result["response"], indent=2), language="json")
    
    # Button to clear portfolio
    if st.button("Clear Portfolio"):
        st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'name', 'quantity', 'price', 'sector', 'industry'])
        st.session_state.optimization_history = []
        st.rerun()

else:
    st.info("Add positions to your portfolio using the form above or load a sample portfolio.")

# Display instructions in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Enter your OpenAI API key
    2. Add portfolio holdings or load a sample
    3. Set your investment preferences
    4. Click "Analyze My Portfolio"
    5. View the recommendations
    
    The AI will adapt its analysis to your specific goals and risk tolerance.
    """)
    
    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    This tool uses OpenAI's language models with progressive prompt refinement:
    
    1. First tries a simple prompt
    2. If needed, refines with more structure
    3. Final attempt uses explicit JSON formatting
    
    Each refinement step makes the prompt more specific to get the most useful structured output.
    """)

# step 9
# Function to generate implementation plan based on recommendations
def generate_implementation_plan(portfolio_df, recommendations, risk_preference, time_horizon):
    """
    Generate a customized implementation plan based on the AI recommendations
    using a separate LLM call to provide more actionable guidance.
    """
    if not api_key or not recommendations:
        return None
    
    try:
        # Extract recommendations data to pass to the LLM
        recommendation_items = []
        if isinstance(recommendations, list):
            for item in recommendations:
                if isinstance(item, dict):
                    ticker = item.get("ticker", "")
                    action = item.get("action", "")
                    change = item.get("percentage_change", "")
                    reasoning = item.get("reasoning", "")
                    
                    recommendation_items.append({
                        "ticker": ticker,
                        "action": action,
                        "change": change,
                        "reasoning": reasoning
                    })
        
        # Create a prompt for the implementation plan
        prompt = f"""
        Based on the following portfolio and investment recommendations, 
        create a detailed implementation plan with specific steps for the investor.
        
        Portfolio Information:
        - Total holdings: {len(portfolio_df)} stocks
        - Risk tolerance: {risk_preference}/10
        - Time horizon: {time_horizon}
        
        Recommended Actions:
        {json.dumps(recommendation_items, indent=2)}
        
        Provide a step-by-step implementation plan that includes:
        1. Timeline: When to make each trade (immediately, over time, specific sequence)
        2. Method: Specific trading approaches (limit orders, dollar-cost averaging, etc.)
        3. Monitoring: Key metrics to track after implementation
        4. Next Review: When to review the portfolio again
        5. Tax Considerations: Any tax implications to be aware of
        
        Personalize the plan based on the risk tolerance and time horizon.
        Make the plan concise, practical, and actionable.
        """
        
        # Call the OpenAI API
        if use_new_openai:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor specializing in portfolio implementation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            plan_text = response.choices[0].message.content
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional financial advisor specializing in portfolio implementation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            plan_text = response.choices[0].message.content
        
        return plan_text
    
    except Exception as e:
        return f"Error generating implementation plan: {str(e)}"

# Add implementation plan tab to the analysis results
if not st.session_state.portfolio.empty and st.session_state.optimization_history:
    latest = st.session_state.optimization_history[-1]
    result = latest["result"]
    
    # Check if we need to add the implementation plan
    # Only add it if we have actual recommendations
    if (result and "response" in result and result["response"] and 
            "recommendations" in result["response"] and 
            isinstance(result["response"]["recommendations"], list)):
    
        
        with tab4:
            st.subheader("📋 Implementation Plan")
            
            # Check if the implementation plan already exists
            if "implementation_plan" not in result:
                with st.spinner("Generating personalized implementation plan..."):
                    plan = generate_implementation_plan(
                        portfolio_df, 
                        result["response"]["recommendations"],
                        risk_preference, 
                        time_horizon
                    )
                    
                    # Add to session state
                    result["implementation_plan"] = plan
                    st.session_state.optimization_history[-1]["result"] = result
            else:
                plan = result["implementation_plan"]
            
            # Display the implementation plan
            if plan:
                # Try to extract sections to create a better UI
                sections = {
                    "Timeline": "",
                    "Method": "",
                    "Monitoring": "",
                    "Next Review": "",
                    "Tax Considerations": ""
                }
                
                # Try to find section headers (not perfect, but works for many formats)
                for section in sections.keys():
                    pattern = f"{section}[:\s]*(.*?)(?=(?:{"|".join(sections.keys())})[:\s]|$)"
                    match = re.search(pattern, plan, re.DOTALL | re.IGNORECASE)
                    if match:
                        sections[section] = match.group(1).strip()
                
                # Check if we successfully extracted sections
                if any(sections.values()):
                    # Display in a nice UI with expandable sections
                    for section, content in sections.items():
                        if content:
                            with st.expander(f"{section}", expanded=True):
                                st.markdown(content)
                else:
                    # Fallback to showing the full text
                    st.markdown(plan)
                
                # Add a download button for the implementation plan
                plan_text = f"# Portfolio Implementation Plan\n\n{plan}"
                st.download_button(
                    label="Download Implementation Plan",
                    data=plan_text,
                    file_name="portfolio_implementation_plan.txt",
                    mime="text/plain"
                )
            else:
                st.info("Unable to generate implementation plan. Please check if you have provided a valid API key.")

# step 10
# Add interactive chatbot section for portfolio questions
if not st.session_state.portfolio.empty:
    # Create a clear separator
    st.markdown("---")
    st.subheader("💬 Ask About Your Portfolio")
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Function to handle chat with portfolio context
    def get_portfolio_chat_response(question, portfolio_df, metrics, recommendations=None):
        """
        Generate a response to a user question about their portfolio
        using the current portfolio data and analysis as context.
        """
        if not api_key:
            return "Please enter your OpenAI API key to use the chat feature."
        
        # Create a snapshot of the portfolio state for context
        holdings = []
        for _, row in portfolio_df.iterrows():
            holdings.append({
                "ticker": row['ticker'],
                "name": row['name'],
                "quantity": float(row['quantity']),
                "price": float(row['price']),
                "value": float(row['value']),
                "weight": float(row['weight']),
                "sector": row['sector']
            })
        
        # Format sector allocation
        sectors = []
        for _, row in metrics['sector_allocation'].iterrows():
            sectors.append({
                "sector": row['sector'],
                "percentage": float(row['percentage'])
            })
        
        # Include recommendations if available
        recs_text = ""
        if recommendations:
            recs = []
            for item in recommendations:
                if isinstance(item, dict):
                    recs.append(item)
            if recs:
                recs_text = f"Recommendations: {json.dumps(recs, indent=2)}"
        
        # Create a conversational yet informative system prompt
        system_prompt = f"""
        You are a helpful financial advisor assistant that answers questions about a user's investment portfolio.
        
        The user has the following portfolio:
        - Total Value: ${metrics['total_value']:,.2f}
        - Number of Holdings: {metrics['num_holdings']}
        - Holdings: {json.dumps(holdings, indent=2)}
        - Sector Allocation: {json.dumps(sectors, indent=2)}
        {recs_text}
        
        Answer the user's questions about their portfolio thoughtfully and accurately based on the data provided.
        Keep answers concise and focused on the portfolio data when relevant, but don't be constrained by templates.
        Feel free to give advice when appropriate, but make it clear when you're offering opinions versus stating facts.
        Indicate when a question cannot be answered with the available data.
        
        For numerical answers, include relevant calculations or reasoning that led to your answer.
        """
        
        # Create messages including the conversation history for context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add a limited conversation history (last 6 exchanges) to maintain context
        history_to_include = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
        
        for exchange in history_to_include:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})
        
        # Add the current question
        messages.append({"role": "user", "content": question})
        
        # Call the OpenAI API
        try:
            if use_new_openai:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content
            
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    # Display chat input and handle responses
    user_question = st.text_input("Ask a question about your portfolio:", key="portfolio_question")
    
    # Get recommendations if they exist
    recommendations = None
    if st.session_state.optimization_history:
        latest = st.session_state.optimization_history[-1]
        if "response" in latest["result"] and latest["result"]["response"]:
            if "recommendations" in latest["result"]["response"]:
                recommendations = latest["result"]["response"]["recommendations"]
    
    if user_question and user_question.strip():
        # Generate response and add to chat history
        with st.spinner("Thinking..."):
            response = get_portfolio_chat_response(
                user_question, 
                portfolio_df, 
                metrics,
                recommendations
            )
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": response
            })
    
    # Display chat history
    st.markdown("### Conversation")
    
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <p style="margin: 0; font-style: italic;">Ask me anything about your portfolio! For example:</p>
            <ul style="margin-bottom: 0;">
                <li>What is my most concentrated position?</li>
                <li>How diversified is my portfolio?</li>
                <li>What sectors am I overweight in?</li>
                <li>How can I reduce risk in my portfolio?</li>
                <li>What would happen if I sold half my tech stocks?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat messages in a more visually appealing way
    for i, exchange in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"""
        <div style="background-color: #e8f4f8; padding: 10px; border-radius: 10px 10px 0 10px; margin: 10px 0 5px 50px;">
            <p style="margin: 0;"><strong>You:</strong> {exchange["question"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI response
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px 10px 10px 0; margin: 5px 50px 15px 0;">
            <p style="margin: 0;"><strong>Portfolio Assistant:</strong> {exchange["answer"]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()