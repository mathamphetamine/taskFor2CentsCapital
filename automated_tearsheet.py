#!/usr/bin/env python3
"""
Automated Portfolio Risk Management System
=========================================

This script automates the daily generation of portfolio risk analysis tearsheets
as new market data becomes available. It fetches the latest price data for the
portfolio assets, recalculates risk metrics, and generates an updated PDF report.

Theoretical Framework
--------------------
This system implements modern portfolio risk management techniques including:

1. Monte Carlo simulation for Value at Risk (VaR) and Conditional VaR (CVaR)
2. Performance metrics calculation (Sharpe, Sortino, Treynor ratios)
3. Drawdown analysis and recovery tracking
4. Benchmark comparison with beta calculation

System Architecture
------------------
The system follows a modular design with these components:

1. Data Acquisition: Fetches and stores historical price data from Yahoo Finance
2. Risk Analytics: Calculates key risk and performance metrics
3. Reporting: Generates PDF tearsheets with visualizations
4. Scheduling: Runs analysis daily at configurable times
5. Notification: Optionally sends email notifications with reports

Key Features
-----------
- Daily automated risk calculations
- Persistent data storage
- Professional PDF report generation
- Email notification system
- Robust error handling and logging
- Configurable portfolio parameters

Usage Example
------------
Basic usage:

```python
# Run once with default settings
python automated_tearsheet.py

# Run as a background service
nohup python automated_tearsheet.py &
```

Configuration:
- Edit PORTFOLIO_CONFIG to change assets, weights, etc.
- Edit EMAIL_CONFIG to enable and configure email notifications
- Adjust schedule timing in main() function

Dependencies
-----------
- yfinance: Market data retrieval
- numpy/pandas: Data manipulation
- matplotlib/seaborn: Visualization
- schedule: Task scheduling
- email/smtplib: Email notification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('portfolio_automation')

# Portfolio configuration
# -----------------------
# This dictionary contains all parameters related to the portfolio composition,
# data handling, and file storage. Modify these values to customize the analysis.
PORTFOLIO_CONFIG = {
    # Asset tickers and allocation weights
    'tickers': ['AMZN', 'FDX', 'NVDA'],  # Yahoo Finance ticker symbols
    'weights': np.array([0.25, 0.30, 0.45]),  # Portfolio allocation (must sum to 1.0)
    
    # Benchmark for comparison (S&P 500 index by default)
    'benchmark': '^GSPC',  # S&P 500 index (Yahoo Finance symbol)
    
    # Historical data parameters
    'lookback_days': 365,  # Number of days of historical data to use (typically 1 year)
    
    # File storage locations
    'output_dir': 'reports',  # Directory for PDF tearsheet reports
    'data_dir': 'data',  # Directory for historical price data CSV files
    
    # Additional parameters can be added here for future enhancements:
    # 'confidence_level': 0.95,  # Confidence level for VaR/CVaR calculations
    # 'trading_days': 252,  # Trading days per year for annualization
    # 'mc_simulations': 10000,  # Number of Monte Carlo simulations
}

# Email configuration for sending reports
# --------------------------------------
# This dictionary contains parameters for the email notification system.
# To enable email notifications, set 'enabled' to True and provide valid credentials.
EMAIL_CONFIG = {
    'enabled': False,  # Set to True to enable email sending
    
    # SMTP server configuration
    'smtp_server': 'smtp.gmail.com',  # SMTP server address
    'smtp_port': 587,  # SMTP port (587 for TLS, 465 for SSL)
    
    # Authentication credentials
    'username': 'your-email@gmail.com',  # Sender email address
    'password': 'your-app-password',  # Use app-specific password for Gmail
    
    # Recipients and message details
    'recipients': ['recipient1@example.com', 'recipient2@example.com'],  # List of recipients
    'subject': 'Daily Portfolio Risk Report'  # Email subject line prefix
    
    # Additional parameters can be added here for future enhancements:
    # 'cc': [],  # Carbon copy recipients
    # 'bcc': [],  # Blind carbon copy recipients
    # 'reply_to': None,  # Reply-to email address
}

def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in [PORTFOLIO_CONFIG['output_dir'], PORTFOLIO_CONFIG['data_dir']]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def fetch_market_data():
    """
    Fetch latest market data for portfolio assets and benchmark.
    Stores data in CSV files and returns the latest data.
    """
    tickers = PORTFOLIO_CONFIG['tickers'] + [PORTFOLIO_CONFIG['benchmark']]
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=PORTFOLIO_CONFIG['lookback_days'])).strftime('%Y-%m-%d')
    
    try:
        # Download data from Yahoo Finance
        logger.info(f"Fetching market data from {start_date} to {end_date}")
        stock_data = yf.download(tickers, start=start_date, end=end_date)
        
        # Save the data
        data_file = os.path.join(PORTFOLIO_CONFIG['data_dir'], f"market_data_{end_date}.csv")
        stock_data['Adj Close'].to_csv(data_file)
        logger.info(f"Saved market data to {data_file}")
        
        return stock_data['Adj Close']
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        # If fetch fails, try to load the most recent data file
        try:
            data_files = sorted([f for f in os.listdir(PORTFOLIO_CONFIG['data_dir']) if f.startswith('market_data_')])
            if data_files:
                latest_file = os.path.join(PORTFOLIO_CONFIG['data_dir'], data_files[-1])
                logger.info(f"Loading most recent data file: {latest_file}")
                return pd.read_csv(latest_file, index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError("No existing market data files found")
        except Exception as inner_e:
            logger.error(f"Failed to load existing data: {str(inner_e)}")
            raise

def calculate_returns(prices):
    """Calculate daily returns from price data."""
    return prices.pct_change().dropna()

def calculate_portfolio_returns(returns, weights):
    """Calculate portfolio returns based on asset returns and weights."""
    return returns.dot(weights)

def calculate_var_cvar(returns, weights, num_simulations=10000, days=1, confidence_level=0.95):
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR) using Monte Carlo simulation.
    
    Mathematical Background:
    -----------------------
    VaR represents the potential loss at a given confidence level. Mathematically:
    
        VaR_α = inf{l : P(L > l) ≤ 1-α}
    
    where:
    - α is the confidence level (e.g., 0.95)
    - L is the loss random variable
    
    CVaR (also known as Expected Shortfall) measures the expected loss given that the 
    loss exceeds VaR. Mathematically:
    
        CVaR_α = E[L | L ≥ VaR_α]
    
    Monte Carlo Implementation:
    --------------------------
    1. Calculate mean returns vector (μ) and covariance matrix (Σ) from historical data
    2. Generate N random return scenarios using a multivariate normal distribution:
       R ~ N(μ*days, Σ*days)
    3. Calculate portfolio returns for each scenario: R_p = weights · R
    4. Compute VaR as the quantile at (1-confidence_level)
    5. Compute CVaR as the mean of returns beyond VaR
    
    Parameters:
    ----------
    returns : DataFrame
        Historical returns for portfolio assets
    weights : array-like
        Portfolio weights for each asset
    num_simulations : int, default=10000
        Number of Monte Carlo simulations to run
    days : int, default=1
        Forecast horizon in days
    confidence_level : float, default=0.95
        Confidence level for VaR/CVaR calculation (typically 0.95 or 0.99)
    
    Returns:
    -------
    dict
        Dictionary containing:
        - 'var': Value at Risk at the specified confidence level
        - 'cvar': Conditional VaR (Expected Shortfall)
        - 'simulated_returns': Array of all simulated returns
        
    Interpretation:
    -------------
    - VaR: With {confidence_level*100}% confidence, the portfolio will not lose more than 
      VaR over the next {days} day(s)
    - CVaR: If the loss exceeds VaR, the expected loss over the next {days} day(s) will be CVaR
    
    Notes:
    -----
    - This implementation assumes returns follow a multivariate normal distribution
    - For longer time horizons, we scale by sqrt(days) for volatility (standard practice)
    - A fixed random seed (42) ensures reproducibility of results
    """
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Generate random returns
    np.random.seed(42)
    simulated_returns = np.random.multivariate_normal(
        mean_returns * days, cov_matrix * days, num_simulations
    )
    
    # Calculate portfolio returns for each simulation
    portfolio_simulated_returns = simulated_returns.dot(weights)
    
    # Sort the simulated returns
    sorted_returns = np.sort(portfolio_simulated_returns)
    
    # Calculate VaR
    var_index = int(num_simulations * (1 - confidence_level))
    var = -sorted_returns[var_index]
    
    # Calculate CVaR (Expected Shortfall)
    cvar = -sorted_returns[:var_index].mean()
    
    return {
        'var': var,
        'cvar': cvar,
        'simulated_returns': sorted_returns
    }

def calculate_performance_metrics(portfolio_returns, benchmark_returns, risk_free_rate=None):
    """
    Calculate key performance metrics for the portfolio.
    
    Mathematical Background:
    -----------------------
    1. Sharpe Ratio: Measures excess return per unit of risk
       Sharpe = (R_p - R_f) / σ_p
       where R_p = portfolio return, R_f = risk-free rate, σ_p = portfolio standard deviation
    
    2. Sortino Ratio: Similar to Sharpe but only penalizes downside volatility
       Sortino = (R_p - R_f) / σ_down
       where σ_down = standard deviation of negative returns only
    
    3. Beta: Measures portfolio's sensitivity to market movements
       β = Cov(R_p, R_m) / Var(R_m)
       where R_m = market/benchmark return
    
    4. Treynor Ratio: Measures excess return per unit of systematic risk
       Treynor = (R_p - R_f) / β
    
    Parameters:
    ----------
    portfolio_returns : Series
        Series of portfolio returns
    benchmark_returns : Series
        Series of benchmark returns (e.g., S&P 500)
    risk_free_rate : float, optional
        Risk-free rate (daily). If None, uses 10-year Treasury yield average
    
    Returns:
    -------
    dict
        Dictionary containing various performance metrics:
        - 'annual_return_portfolio': Annualized portfolio return
        - 'annual_volatility_portfolio': Annualized portfolio volatility
        - 'annual_return_benchmark': Annualized benchmark return
        - 'annual_volatility_benchmark': Annualized benchmark volatility
        - 'sharpe_ratio_portfolio': Portfolio Sharpe ratio
        - 'sharpe_ratio_benchmark': Benchmark Sharpe ratio
        - 'sortino_ratio_portfolio': Portfolio Sortino ratio
        - 'sortino_ratio_benchmark': Benchmark Sortino ratio
        - 'beta': Portfolio beta relative to benchmark
        - 'treynor_ratio': Portfolio Treynor ratio
        - 'risk_free_rate': Annualized risk-free rate used
    
    Interpretation:
    -------------
    - Higher Sharpe/Sortino ratios indicate better risk-adjusted performance
    - Beta > 1: More volatile than the market; Beta < 1: Less volatile
    - Higher Treynor ratio indicates better return per unit of market risk
    
    Notes:
    -----
    - All metrics are annualized assuming 252 trading days per year
    - Treasury yield is used as a proxy for the risk-free rate
    - The risk-free rate is fetched for the most recent 30 days and averaged
    """
    # If risk_free_rate not provided, use 10-year Treasury yield
    if risk_free_rate is None:
        risk_free_data = yf.download('^TNX', 
                                     start=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                     end=datetime.now().strftime('%Y-%m-%d'))
        risk_free_rate = risk_free_data['Adj Close'].mean() / 100 / 252  # Daily rate
    
    # Calculate annualized metrics
    trading_days = 252
    annualized_return_portfolio = portfolio_returns.mean() * trading_days
    annualized_volatility_portfolio = portfolio_returns.std() * np.sqrt(trading_days)
    annualized_return_benchmark = benchmark_returns.mean() * trading_days
    annualized_volatility_benchmark = benchmark_returns.std() * np.sqrt(trading_days)
    
    # Sharpe ratio
    sharpe_ratio_portfolio = (annualized_return_portfolio - (risk_free_rate * trading_days)) / annualized_volatility_portfolio
    sharpe_ratio_benchmark = (annualized_return_benchmark - (risk_free_rate * trading_days)) / annualized_volatility_benchmark
    
    # Sortino ratio - measuring downside risk only
    negative_returns_portfolio = portfolio_returns[portfolio_returns < 0]
    negative_returns_benchmark = benchmark_returns[benchmark_returns < 0]
    downside_deviation_portfolio = negative_returns_portfolio.std() * np.sqrt(trading_days)
    downside_deviation_benchmark = negative_returns_benchmark.std() * np.sqrt(trading_days)
    sortino_ratio_portfolio = (annualized_return_portfolio - (risk_free_rate * trading_days)) / downside_deviation_portfolio
    sortino_ratio_benchmark = (annualized_return_benchmark - (risk_free_rate * trading_days)) / downside_deviation_benchmark
    
    # Calculate portfolio beta for Treynor Ratio
    covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance
    treynor_ratio = (annualized_return_portfolio - (risk_free_rate * trading_days)) / beta
    
    return {
        'annual_return_portfolio': annualized_return_portfolio,
        'annual_volatility_portfolio': annualized_volatility_portfolio,
        'annual_return_benchmark': annualized_return_benchmark, 
        'annual_volatility_benchmark': annualized_volatility_benchmark,
        'sharpe_ratio_portfolio': sharpe_ratio_portfolio,
        'sharpe_ratio_benchmark': sharpe_ratio_benchmark,
        'sortino_ratio_portfolio': sortino_ratio_portfolio,
        'sortino_ratio_benchmark': sortino_ratio_benchmark,
        'beta': beta,
        'treynor_ratio': treynor_ratio,
        'risk_free_rate': risk_free_rate * trading_days
    }

def calculate_drawdowns(returns):
    """
    Calculate drawdowns for a return series.
    
    Mathematical Background:
    -----------------------
    Drawdown measures the decline from a historical peak in wealth or cumulative returns.
    
    1. Wealth Index: W(t) = (1 + r_1) × (1 + r_2) × ... × (1 + r_t)
       where r_i is the return at time i
    
    2. Running Maximum: M(t) = max{W(1), W(2), ..., W(t)}
    
    3. Drawdown at time t: DD(t) = (W(t) / M(t)) - 1
       Note: Drawdowns are negative values representing percentage losses
    
    The Maximum Drawdown (MDD) is defined as:
    MDD = min{DD(1), DD(2), ..., DD(T)}
    
    Parameters:
    ----------
    returns : Series
        Series of returns (typically daily)
    
    Returns:
    -------
    tuple
        A tuple containing:
        1. drawdown_df : DataFrame
           DataFrame with details about each drawdown episode:
           - Start Date: When the drawdown began
           - Max Drawdown Date: When the maximum loss occurred
           - End Date: When the drawdown recovered or the series ended
           - Max Drawdown: The maximum percentage loss
           - Duration (days): Length of the drawdown period in days
        
        2. drawdown : Series
           Continuous drawdown series with the same index as the input returns
    
    Interpretation:
    -------------
    - Drawdowns measure the magnitude, duration, and frequency of losses
    - Lower (more negative) maximum drawdowns indicate higher risk
    - Longer recovery periods suggest slower recuperation from losses
    - Analysis of worst drawdowns helps understand tail risk events
    
    Notes:
    -----
    - Drawdowns start when the cumulative return falls below its previous peak
    - Drawdowns end when the cumulative return reaches a new peak
    - Ongoing drawdowns have their end date set to the last date in the series
    """
    # Calculate wealth index (cumulative returns)
    wealth_index = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = wealth_index.cummax()
    
    # Calculate drawdown
    drawdown = (wealth_index / running_max) - 1
    
    # Create drawdown episodes
    is_zero = drawdown == 0
    # Shifts to identify start of drawdown periods
    start_indicators = (~is_zero) & is_zero.shift(1).fillna(True)
    # Group by episode
    episode_starts = drawdown.index[start_indicators].tolist()
    
    drawdown_info = []
    
    for i, start in enumerate(episode_starts):
        # Find end of this episode (next start or end of series)
        end = episode_starts[i+1] if i < len(episode_starts)-1 else drawdown.index[-1]
        
        # Get drawdown in this period
        episode_drawdown = drawdown.loc[start:end]
        
        if episode_drawdown.min() < 0:  # Only record if there was a drawdown
            max_drawdown = episode_drawdown.min()
            max_drawdown_date = episode_drawdown.idxmin()
            end_date = episode_drawdown.index[episode_drawdown == 0].min()
            
            # If drawdown hasn't recovered yet, set end_date to last date
            if pd.isna(end_date):
                end_date = episode_drawdown.index[-1]
            
            # Calculate duration
            duration = (end_date - start).days
            
            drawdown_info.append({
                'Start Date': start,
                'Max Drawdown Date': max_drawdown_date,
                'End Date': end_date,
                'Max Drawdown': max_drawdown,
                'Duration (days)': duration
            })
    
    drawdown_df = pd.DataFrame(drawdown_info)
    if not drawdown_df.empty:
        drawdown_df = drawdown_df.sort_values('Max Drawdown')
    
    return drawdown_df, drawdown

def generate_tearsheet(portfolio_returns, benchmark_returns, tickers, weights, 
                       var_1d, cvar_1d, var_10d, cvar_10d, perf_metrics, 
                       drawdowns_df, portfolio_drawdown, output_file):
    """
    Generate a PDF tear sheet with all risk metrics and visualizations.
    
    Parameters:
    - portfolio_returns: Series of portfolio returns
    - benchmark_returns: Series of benchmark returns
    - tickers: List of ticker symbols
    - weights: Array of portfolio weights
    - var_1d, cvar_1d: 1-day VaR and CVaR values
    - var_10d, cvar_10d: 10-day VaR and CVaR values
    - perf_metrics: Dictionary of performance metrics
    - drawdowns_df: DataFrame with drawdown information
    - portfolio_drawdown: Series of continuous drawdown values
    - output_file: Path to save the PDF report
    """
    # Custom percentage formatter for plots
    def percentage_format(x, pos):
        return f'{100 * x:.1f}%'
    
    with PdfPages(output_file) as pdf:
        # Page 1: Portfolio Overview
        plt.figure(figsize=(11, 8.5))
        plt.suptitle(f'Portfolio Risk Management System - {datetime.now().strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        
        # Portfolio composition
        plt.subplot(221)
        plt.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Portfolio Composition')
        
        # Cumulative returns
        cumul_returns = (1 + pd.DataFrame({
            'Portfolio': portfolio_returns,
            'S&P 500': benchmark_returns
        })).cumprod() - 1
        
        plt.subplot(222)
        cumul_returns.plot(ax=plt.gca())
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_format))
        plt.grid(True, alpha=0.3)
        
        # Key metrics table
        plt.subplot(212)
        plt.axis('off')
        
        # Format the metrics text
        annual_return_port = perf_metrics['annual_return_portfolio']
        annual_return_bench = perf_metrics['annual_return_benchmark']
        annual_vol_port = perf_metrics['annual_volatility_portfolio']
        annual_vol_bench = perf_metrics['annual_volatility_benchmark']
        sharpe_port = perf_metrics['sharpe_ratio_portfolio']
        sharpe_bench = perf_metrics['sharpe_ratio_benchmark']
        sortino_port = perf_metrics['sortino_ratio_portfolio']
        sortino_bench = perf_metrics['sortino_ratio_benchmark']
        beta = perf_metrics['beta']
        treynor = perf_metrics['treynor_ratio']
        max_dd_port = portfolio_drawdown.min()
        
        metrics_text = f"""
        Performance Summary (Annualized)
        ------------------------------------------
                        Portfolio       S&P 500
        Return:         {annual_return_port:.2%}         {annual_return_bench:.2%}
        Volatility:     {annual_vol_port:.2%}         {annual_vol_bench:.2%}
        Sharpe Ratio:   {sharpe_port:.2f}           {sharpe_bench:.2f}
        Sortino Ratio:  {sortino_port:.2f}           {sortino_bench:.2f}
        Beta:           {beta:.2f}           1.00
        Treynor Ratio:  {treynor:.2f}           {annual_return_bench:.2f}
        Max Drawdown:   {max_dd_port:.2%}         N/A
        
        Analysis Period: {portfolio_returns.index[0].strftime('%Y-%m-%d')} - {portfolio_returns.index[-1].strftime('%Y-%m-%d')}
        """
        
        plt.text(0.1, 0.9, metrics_text, fontsize=10, family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Page 2: Risk Metrics
        plt.figure(figsize=(11, 8.5))
        plt.suptitle('Risk Metrics & Analysis', fontsize=16, fontweight='bold')
        
        # VaR and CVaR visualizations
        gs = gridspec.GridSpec(2, 2)
        
        ax1 = plt.subplot(gs[0, 0])
        ax1.barh(['1-Day', '10-Day'], [var_1d, var_10d], color='skyblue')
        ax1.set_title('Value at Risk (VaR 95%)')
        ax1.xaxis.set_major_formatter(FuncFormatter(percentage_format))
        for i, v in enumerate([var_1d, var_10d]):
            ax1.text(v/2, i, f'{v:.2%}', va='center')
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.barh(['1-Day', '10-Day'], [cvar_1d, cvar_10d], color='coral')
        ax2.set_title('Conditional VaR (CVaR 95%)')
        ax2.xaxis.set_major_formatter(FuncFormatter(percentage_format))
        for i, v in enumerate([cvar_1d, cvar_10d]):
            ax2.text(v/2, i, f'{v:.2%}', va='center')
        
        # Drawdown visualization
        ax3 = plt.subplot(gs[1, :])
        ax3.plot(portfolio_drawdown.index, portfolio_drawdown, color='red', linewidth=1.5)
        ax3.fill_between(portfolio_drawdown.index, portfolio_drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Portfolio Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.yaxis.set_major_formatter(FuncFormatter(percentage_format))
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Page 3: Worst Drawdowns (if any)
        if not drawdowns_df.empty:
            plt.figure(figsize=(11, 8.5))
            plt.suptitle('Drawdown Analysis', fontsize=16, fontweight='bold')
            
            # Table of worst drawdowns
            plt.subplot(111)
            plt.axis('off')
            
            drawdown_text = "Worst 5 Drawdowns\n" + "-" * 80 + "\n"
            drawdown_text += "   Start Date      End Date     Max Drawdown    Duration\n"
            drawdown_text += "-" * 80 + "\n"
            
            for i, row in drawdowns_df.head(5).iterrows():
                drawdown_text += f"{row['Start Date'].strftime('%Y-%m-%d')}   {row['End Date'].strftime('%Y-%m-%d')}   {row['Max Drawdown']:.2%}         {row['Duration (days)']} days\n"
                
            plt.text(0.1, 0.9, drawdown_text, fontsize=10, family='monospace', va='top')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig()
            plt.close()
    
    logger.info(f"Tearsheet generated and saved to {output_file}")
    return output_file

def send_email_report(pdf_path):
    """Send the generated report via email."""
    if not EMAIL_CONFIG['enabled']:
        logger.info("Email sending is disabled in configuration")
        return
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['username']
        msg['To'] = ', '.join(EMAIL_CONFIG['recipients'])
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = f"{EMAIL_CONFIG['subject']} - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Email body
        msg.attach(MIMEText(f"Please find attached the portfolio risk report for {datetime.now().strftime('%Y-%m-%d')}.\n\n"))
        
        # Attach the PDF
        with open(pdf_path, 'rb') as f:
            part = MIMEBase('application', 'pdf')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(pdf_path)}"')
            msg.attach(part)
        
        # Send the email
        server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        server.starttls()
        server.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent to {', '.join(EMAIL_CONFIG['recipients'])}")
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")

def run_daily_analysis():
    """
    Main function to run the daily portfolio analysis and generate the tearsheet.
    
    This function orchestrates the entire risk analysis process:
    1. Sets up storage directories
    2. Fetches the latest market data
    3. Calculates returns and portfolio performance
    4. Runs risk simulations (VaR/CVaR)
    5. Computes performance metrics and drawdowns
    6. Generates and distributes the PDF tearsheet
    
    The analysis follows a sequential pipeline where each step depends on the 
    outputs of previous steps. Error handling ensures that failures are logged
    and don't crash the automated process.
    
    Returns:
    -------
    bool
        True if analysis completed successfully, False otherwise
    """
    logger.info("Starting daily portfolio analysis")
    
    try:
        # Step 1: Ensure directories exist for data and reports storage
        # This prevents file I/O errors later in the process
        setup_directories()
        
        # Step 2: Fetch latest market data from Yahoo Finance
        # This downloads price data for portfolio assets and benchmark
        logger.info("Fetching market data...")
        price_data = fetch_market_data()
        
        # Step 3: Calculate returns from price data
        # Convert prices to percentage returns and remove any NaN values
        logger.info("Calculating returns...")
        returns = calculate_returns(price_data)
        
        # Step 4: Calculate portfolio returns using asset weights
        # Extract portfolio configuration parameters
        tickers = PORTFOLIO_CONFIG['tickers']
        weights = PORTFOLIO_CONFIG['weights']
        benchmark = PORTFOLIO_CONFIG['benchmark']
        
        # Compute weighted portfolio returns and extract benchmark returns
        logger.info("Computing portfolio performance...")
        portfolio_returns = calculate_portfolio_returns(returns[tickers], weights)
        benchmark_returns = returns[benchmark]
        
        # Step 5: Run Monte Carlo simulations for risk metrics
        # Calculate 1-day and 10-day VaR/CVaR at 95% confidence
        logger.info("Running Monte Carlo simulations for VaR/CVaR...")
        var_cvar_1d = calculate_var_cvar(returns[tickers], weights, days=1)
        var_cvar_10d = calculate_var_cvar(returns[tickers], weights, days=10)
        
        # Step 6: Calculate performance metrics and ratios
        # Compute Sharpe, Sortino, Beta, Treynor, etc.
        logger.info("Calculating performance metrics...")
        perf_metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
        
        # Step 7: Analyze drawdowns (peak-to-trough declines)
        # Identify and analyze all drawdown episodes
        logger.info("Analyzing drawdowns...")
        drawdowns_df, portfolio_drawdown = calculate_drawdowns(portfolio_returns)
        
        # Step 8: Generate the PDF tearsheet with all analysis results
        # Create a professional report with visualizations
        logger.info("Generating tearsheet...")
        today_str = datetime.now().strftime('%Y-%m-%d')
        output_file = os.path.join(PORTFOLIO_CONFIG['output_dir'], f"portfolio_tearsheet_{today_str}.pdf")
        
        pdf_path = generate_tearsheet(
            portfolio_returns,
            benchmark_returns,
            tickers,
            weights,
            var_cvar_1d['var'],
            var_cvar_1d['cvar'],
            var_cvar_10d['var'],
            var_cvar_10d['cvar'],
            perf_metrics,
            drawdowns_df,
            portfolio_drawdown,
            output_file
        )
        
        # Step 9: Distribute the report via email if enabled
        # Send notification with PDF attachment
        logger.info("Distributing report...")
        send_email_report(pdf_path)
        
        logger.info("Daily portfolio analysis completed successfully")
        return True
    except Exception as e:
        # Comprehensive error handling to prevent automation failures
        logger.error(f"Error in daily portfolio analysis: {str(e)}")
        return False

def main():
    """
    Main entry point for the automated portfolio risk management system.
    
    This function:
    1. Runs an initial analysis immediately upon startup
    2. Sets up a daily schedule for automated report generation
    3. Maintains a continuous execution loop to keep the scheduler running
    
    Production Deployment Notes:
    ---------------------------
    For robust production deployment:
    
    1. Service Management:
       - Use systemd on Linux: Create a service unit file for automatic startup/restart
       - On Windows: Use Task Scheduler with the "Run whether user is logged in or not" option
    
    2. Error Handling:
       - External monitoring is recommended to ensure the process stays running
       - Consider implementing a heartbeat mechanism to an external monitoring service
    
    3. Resource Considerations:
       - Memory usage may increase over time; consider periodic restarts
       - Market data is only available during trading days; scheduling should account for this
    
    4. Security:
       - For email functionality, use app-specific passwords or OAuth2 
       - Store credentials in environment variables rather than in the code
       - Implement access controls on the generated reports directory
    """
    # Run once at startup to generate an initial report
    # This ensures we have a report even if we start the service after the scheduled time
    initial_result = run_daily_analysis()
    if initial_result:
        logger.info("Initial analysis completed successfully")
    else:
        logger.warning("Initial analysis failed - will retry at next scheduled run")
    
    # Schedule to run every day at 16:30 (market close + 30 minutes)
    # Using US Eastern time is recommended since market closes at 16:00 ET
    # Adjust this time based on when you expect the day's final data to be available
    schedule.every().day.at("16:30").do(run_daily_analysis)
    
    logger.info("Automated portfolio risk management system started")
    logger.info("Scheduled to run daily at 16:30")
    
    # Keep the script running to execute scheduled jobs
    # This infinite loop checks pending tasks every minute
    while True:
        # Run any pending scheduled tasks
        schedule.run_pending()
        
        # Sleep for 60 seconds before checking again
        # This controls CPU usage while still being responsive
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
