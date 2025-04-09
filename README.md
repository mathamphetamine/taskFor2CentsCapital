# Portfolio Risk Management System

A comprehensive system for monitoring and analyzing the risk profile of a portfolio consisting of three equities: Amazon (AMZN, 25%), FedEx (FDX, 30%), and Nvidia (NVDA, 45%).

## Overview

This project implements a complete portfolio risk management solution that:

1. Analyzes historical price data for a multi-asset portfolio
2. Calculates key risk and performance metrics using industry-standard methods
3. Generates professional tear sheets with visualizations and interpretations
4. Automates the entire process for daily risk monitoring

Components:
1. A Jupyter Notebook (`portfolio_risk_analysis.ipynb`) that performs a detailed interactive risk analysis
2. An automation script (`automated_tearsheet.py`) that enables daily risk report generation

## Theoretical Background

### Risk Metrics

#### Value at Risk (VaR)
Value at Risk represents the maximum expected loss over a specific time horizon at a given confidence level. Mathematically:

VaR_α = inf{l : P(L > l) ≤ 1-α}

Where:
- α is the confidence level (e.g., 95%)
- L is the loss random variable

Interpretation: "With 95% confidence, the maximum loss over the next day will not exceed X%."

#### Conditional Value at Risk (CVaR)
Also known as Expected Shortfall, CVaR measures the expected loss given that the loss exceeds VaR. Mathematically:

CVaR_α = E[L | L ≥ VaR_α]

Interpretation: "If losses exceed VaR, the expected loss will be X%."

### Performance Metrics

#### Sharpe Ratio
Measures excess return per unit of risk:

Sharpe = (R_p - R_f) / σ_p

Where:
- R_p = portfolio return
- R_f = risk-free rate
- σ_p = portfolio standard deviation

Higher values indicate better risk-adjusted performance.

#### Sortino Ratio
Similar to Sharpe but only penalizes downside volatility:

Sortino = (R_p - R_f) / σ_down

Where σ_down is the standard deviation of negative returns only.

#### Treynor Ratio
Measures excess return per unit of systematic risk:

Treynor = (R_p - R_f) / β

Where β (beta) represents the portfolio's sensitivity to market movements.

### Drawdown Analysis

Drawdown measures the decline from a historical peak in wealth:

DD(t) = (W(t) / M(t)) - 1

Where:
- W(t) is the wealth index at time t
- M(t) is the running maximum of wealth up to time t

Maximum drawdown (the worst peak-to-trough decline) is a key risk indicator.

## Features

- **Monte Carlo Simulation**: Robust VaR and CVaR calculation using 10,000 simulations
- **Multiple Time Horizons**: Both 1-day and 10-day risk forecasts
- **Comprehensive Performance Metrics**: Sharpe, Sortino, Treynor ratios
- **Detailed Drawdown Analysis**: Maximum drawdown and worst 5 drawdown episodes
- **Benchmark Comparison**: Performance comparison against S&P 500
- **Professional Visualizations**: Publication-quality charts and tear sheets
- **Fully Automated Workflow**: Daily data updates, analysis, and reporting
- **Email Notifications**: Optional email delivery of PDF reports
- **Robust Error Handling**: Fallback mechanisms and comprehensive logging

## Requirements

The following Python packages are required:

```
yfinance >= 0.2.55
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.10.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
fpdf >= 1.7.2
schedule >= 1.1.0 (for automation)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-risk-management.git
   cd portfolio-risk-management
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Jupyter Notebook

The notebook provides an interactive analysis environment with detailed explanations and visualizations:

```bash
jupyter notebook portfolio_risk_analysis.ipynb
```

The notebook is organized into the following sections:

1. **Data Collection**: Fetching and cleaning price data from Yahoo Finance
2. **Returns Analysis**: Calculating and visualizing asset and portfolio returns
3. **Monte Carlo Simulation**: Implementing VaR and CVaR calculations
4. **Performance Metrics**: Computing Sharpe, Sortino, and Treynor ratios
5. **Drawdown Analysis**: Identifying and measuring drawdown episodes
6. **Benchmark Comparison**: Comparing performance to S&P 500
7. **Tear Sheet Generation**: Creating a professional PDF report

### Automated Daily Reports

To set up the automated tearsheet generation:

1. Configure portfolio parameters in `automated_tearsheet.py`:
   ```python
   PORTFOLIO_CONFIG = {
       'tickers': ['AMZN', 'FDX', 'NVDA'],  # Yahoo Finance ticker symbols
       'weights': np.array([0.25, 0.30, 0.45]),  # Must sum to 1.0
       'benchmark': '^GSPC',  # S&P 500
       'lookback_days': 365,  # Historical data period
       'output_dir': 'reports',  # Report storage location
       'data_dir': 'data',  # Data storage location
   }
   ```

2. (Optional) Configure email notifications:
   ```python
   EMAIL_CONFIG = {
       'enabled': True,
       'smtp_server': 'smtp.gmail.com',
       'smtp_port': 587,
       'username': 'your-email@gmail.com',
       'password': 'your-app-password',
       'recipients': ['recipient@example.com'],
       'subject': 'Daily Portfolio Risk Report'
   }
   ```

3. Run the automation script:
   ```bash
   python automated_tearsheet.py
   ```

By default, the script will:
- Generate a report immediately upon startup
- Schedule daily report generation at 16:30 (30 minutes after market close)
- Store reports in the `reports` directory and historical price data in the `data` directory
- Log all operations to `portfolio_automation.log`

## Automation System Design

The automation system implements a complete end-to-end workflow:

1. **Data fetching and storage**:
   - Daily price updates using the Yahoo Finance API
   - Persistent storage of historical data in CSV format
   - Fallback to most recent data if live fetch fails

2. **Metric recalculation**:
   - Daily recalculation of all risk metrics
   - Monte Carlo simulation for VaR/CVaR (10,000 simulations)
   - Performance metrics compared to benchmark
   - Comprehensive drawdown analysis

3. **Report generation**:
   - Automatic PDF tearsheet generation with key metrics and visualizations
   - Daily reports named with date stamps (`portfolio_tearsheet_YYYY-MM-DD.pdf`)
   - Professional-quality charts and tables

4. **Scheduling mechanism**:
   - Python `schedule` library for task scheduling
   - Configurable timing for report generation
   - Continuous execution loop with error handling

5. **Notification system**:
   - Email notifications with attached PDF reports
   - Comprehensive logging system for tracking execution
   - Error handling and reporting

## Production Deployment

For robust production deployment:

1. **Service Management**:
   - On Linux: Create a systemd service unit file:
     ```
     [Unit]
     Description=Portfolio Risk Management System
     After=network.target

     [Service]
     User=username
     WorkingDirectory=/path/to/portfolio-risk-management
     ExecStart=/path/to/python /path/to/portfolio-risk-management/automated_tearsheet.py
     Restart=always
     RestartSec=10

     [Install]
     WantedBy=multi-user.target
     ```

   - On Windows: Use Task Scheduler with the "Run whether user is logged in or not" option

2. **Security Considerations**:
   - Store email credentials in environment variables rather than in code
   - Implement access controls on the reports directory
   - Use app-specific passwords or OAuth2 for email authentication

3. **Monitoring**:
   - Implement external monitoring to ensure the process remains running
   - Consider adding a heartbeat mechanism to an external monitoring service
   - Review logs regularly for warnings or errors

## Customization

The system can be easily adapted for different portfolios:

1. **Different Assets**: Change the `tickers` and `weights` in the configuration
2. **Different Benchmark**: Change the `benchmark` ticker symbol
3. **Different Time Horizons**: Modify the VaR/CVaR calculation days parameter
4. **Custom Scheduling**: Change the scheduling time in the `main()` function

## Interpretation of Results

### Risk Metrics:
- **Higher VaR/CVaR**: Indicates greater potential losses and higher risk
- **Higher Volatility**: Indicates more price fluctuation and potentially higher risk

### Performance Metrics:
- **Higher Sharpe/Sortino Ratios**: Better risk-adjusted performance
- **Beta > 1**: More volatile than the market
- **Beta < 1**: Less volatile than the market
- **Higher Treynor Ratio**: Better return per unit of market risk

### Drawdowns:
- **Deeper Maximum Drawdown**: Indicates higher risk
- **Longer Duration Drawdowns**: Suggests slower recovery from losses

## License

MIT

## Author

[Your Name] 