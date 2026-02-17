# F500 Problem Set 1 - Stock Return Analysis

This project provides a Python environment for completing Problem Set 1 on financial econometrics and trading strategies.

## Setup Instructions

### 1. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
# Create a virtual environment and install all dependencies
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# Or with development dependencies (jupyter, black, pytest, etc.)
uv pip install -r pyproject.toml --extra dev
```

Or use `uv run` to execute scripts without manually activating the venv:

```bash
uv run --with -r pyproject.toml python main.py
```

### 2. Project Structure

```
f500-pset1/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── pset1/                  # Main package
│   ├── __init__.py
│   ├── data_loader.py      # Download stock data
│   ├── analysis.py         # Q1 & Q4: Statistical analysis
│   ├── trading.py          # Q2 & Q3: Moving averages & Bollinger bands
│   ├── roll_model.py       # Q6 & Q7: Roll model analysis
│   └── utils.py            # Helper functions
├── notebooks/              # Jupyter notebooks for exploration
│   ├── q1_statistics.ipynb
│   ├── q2_moving_averages.ipynb
│   ├── q3_bollinger_bands.ipynb
│   └── q4_rolling_window.ipynb
├── data/                   # Downloaded data
└── results/                # Output files and plots
```

## Problem Set Overview

### Question 1: Basic Statistics
- Download daily price data for 1 stock index + 2 individual stocks
- Compute: mean, std dev, skewness, kurtosis
- Calculate first 20 autocorrelations
- Compare log returns vs actual returns

### Question 2: Moving Average Filters
- Compute SMA (Simple Moving Average) with k = 5, 10, 20, 50, 100, 200
- Compute EWMA (Exponential Weighted Moving Average)
- Test contrarian vs momentum trading strategies

### Question 3: Bollinger Bands
- Compute upper/lower Bollinger bands
- Test trading strategies based on bands

### Question 4: Rolling Window Analysis
- Apply rolling 252-day window to statistics from Q1
- Analyze time-varying patterns

### Questions 5-8: Theoretical
- Q5: Aggregated returns analysis
- Q6-7: Roll model and extensions
- Q8: Event study methodology

## Quick Start Example

```python
from pset1.data_loader import download_data
from pset1.analysis import compute_basic_statistics
from pset1.trading import compute_sma, backtest_strategy

# Download data
data = download_data(
    tickers=['SPY', 'AAPL', 'MSFT'],  # Example: S&P 500 ETF + 2 stocks
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Q1: Compute statistics
stats = compute_basic_statistics(data)
print(stats)

# Q2: Moving average strategy
sma_20 = compute_sma(data['SPY'], k=20)
results = backtest_strategy(data['SPY'], sma_20, strategy='momentum')
print(results)
```

## Data Sources

This project uses:
- **yfinance**: For downloading stock price data from Yahoo Finance
- Alternatives: pandas_datareader, Alpha Vantage, Quandl

## Key Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scipy**: Statistical functions
- **statsmodels**: Time series analysis, autocorrelations
- **matplotlib/seaborn**: Visualization
- **yfinance**: Data download

## Tips

1. **Returns Calculation**: 
   - Log returns: `log(P_t / P_{t-1})`
   - Simple returns: `(P_t - P_{t-1}) / P_{t-1}`

2. **Autocorrelation Testing**:
   - Use Ljung-Box test for joint significance
   - Individual t-tests for each lag

3. **Trading Strategy Evaluation**:
   - Calculate cumulative returns
   - Compare with buy-and-hold
   - Consider transaction costs

4. **Rolling Window**:
   - Use `pandas.DataFrame.rolling(window=252)`
   - Pay attention to minimum periods

## Output

Results will be saved to:
- `results/statistics/`: Summary tables (CSV/Excel)
- `results/plots/`: Visualizations (PNG/PDF)
- `results/reports/`: Final report

## References

- Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation"
- Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- Campbell, Lo, MacKinlay (1997). "The Econometrics of Financial Markets"
