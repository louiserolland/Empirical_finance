"""
Main analysis script for F500 Problem Set 1.

This script runs all analyses for the problem set.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pset1.data_loader import download_data, compute_returns
from pset1.analysis import (
    compute_basic_statistics,
    compute_autocorrelations,
    compare_return_methods,
    rolling_window_analysis,
    plot_statistics_summary
)
from pset1.trading import (
    compute_sma,
    compute_ewma,
    compute_bollinger_bands,
    generate_signals,
    generate_bollinger_signals,
    backtest_strategy,
    compare_strategies,
    plot_strategy
)
from pset1.roll_model import (
    compute_roll_spread_constant,
    test_roll_implication,
    compute_roll_spread_timevarying
)
from pset1.utils import (
    setup_plotting_style,
    print_section_header,
    export_results
)


def main():
    """Run complete analysis for Problem Set 1."""
    
    # Create output directories
    os.makedirs('results/statistics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/trading', exist_ok=True)
    
    # Setup plotting
    setup_plotting_style()
    
    # ========================================================================
    # STEP 1: DEFINE TICKERS AND DOWNLOAD DATA
    # ========================================================================
    print_section_header("STEP 1: DATA DOWNLOAD", width=80)
    
    # TODO: Modify these tickers for your chosen market
    tickers = [
        'SPY',   # S&P 500 ETF (stock index)
        'AAPL',  # Apple Inc.
        'MSFT'   # Microsoft Corp.
    ]
    
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    
    print(f"Downloading data for: {', '.join(tickers)}")
    print(f"Period: {start_date} to {end_date}")
    
    data = download_data(tickers, start_date=start_date, end_date=end_date)
    
    # Compute returns
    returns = {}
    for ticker in tickers:
        if ticker in data:
            returns[ticker] = compute_returns(data[ticker], method='log')
            print(f"{ticker}: {len(returns[ticker])} observations")
    
    # ========================================================================
    # QUESTION 1: BASIC STATISTICS
    # ========================================================================
    print_section_header("QUESTION 1: BASIC STATISTICS", width=80)
    
    # 1(a): Basic statistics
    print("\nQ1(a): Computing basic statistics...")
    all_stats = {}
    for ticker, ret in returns.items():
        all_stats[ticker] = compute_basic_statistics(ret, name=ticker)
    
    stats_df = pd.concat(all_stats, axis=1)
    print(stats_df)
    
    # 1(b): Autocorrelations
    print("\n" + "-" * 80)
    print("Q1(b): Computing autocorrelations...")
    acf_results = {}
    for ticker, ret in returns.items():
        acf_df, lb_test = compute_autocorrelations(ret, nlags=20)
        acf_results[ticker] = acf_df
        print(f"\n{ticker} - Ljung-Box test p-value: {lb_test['P-value']:.4f}")
    
    # 1(c): Log vs Simple returns
    print("\n" + "-" * 80)
    print("Q1(c): Comparing log vs simple returns...")
    comparison_results = {}
    for ticker in tickers:
        if ticker in data:
            comparison = compare_return_methods(data[ticker]['Adj Close'])
            comparison_results[ticker] = comparison
            print(f"\n{ticker}:")
            print(comparison)
    
    # Export Q1 results
    q1_results = {
        'Statistics': stats_df,
        **{f'ACF_{ticker}': acf_results[ticker] 
           for ticker in acf_results},
        **{f'LogVsSimple_{ticker}': comparison_results[ticker] 
           for ticker in comparison_results}
    }
    export_results(q1_results, 'results/statistics/q1_results.xlsx')
    
    # ========================================================================
    # QUESTION 2: MOVING AVERAGE STRATEGIES
    # ========================================================================
    print_section_header("QUESTION 2: MOVING AVERAGE STRATEGIES", width=80)
    
    # Focus on the index for trading strategies
    index_ticker = tickers[0]
    prices = data[index_ticker]['Adj Close']
    
    # Test different SMA windows
    k_values = [5, 10, 20, 50, 100, 200]
    
    print(f"\nTesting SMA strategies on {index_ticker}...")
    momentum_results = compare_strategies(prices, k_values, strategy='momentum')
    contrarian_results = compare_strategies(prices, k_values, strategy='contrarian')
    
    print("\nMomentum Strategy Results:")
    print(momentum_results)
    
    print("\nContrarian Strategy Results:")
    print(contrarian_results)
    
    # Test EWMA
    print("\n" + "-" * 80)
    print("Testing EWMA strategies...")
    ewma_results = []
    for k in [20, 50]:
        ewma = compute_ewma(prices, k=k)
        signals = generate_signals(prices, ewma, strategy='momentum')
        backtest = backtest_strategy(prices, signals)
        
        ewma_results.append({
            'k': k,
            'alpha': 2/(k+1),
            **backtest['metrics']
        })
    
    ewma_df = pd.DataFrame(ewma_results)
    print(ewma_df)
    
    # Export Q2 results
    q2_results = {
        'SMA_Momentum': momentum_results,
        'SMA_Contrarian': contrarian_results,
        'EWMA': ewma_df
    }
    export_results(q2_results, 'results/trading/q2_moving_averages.xlsx')
    
    # Plot example strategy
    sma_20 = compute_sma(prices, k=20)
    signals_mom = generate_signals(prices, sma_20, strategy='momentum')
    backtest_mom = backtest_strategy(prices, signals_mom)
    
    plot_strategy(
        prices, signals_mom, indicator=sma_20,
        backtest_results=backtest_mom,
        title=f"{index_ticker} - SMA(20) Momentum Strategy",
        save_path='results/plots/q2_sma_strategy.png'
    )
    
    # ========================================================================
    # QUESTION 3: BOLLINGER BANDS
    # ========================================================================
    print_section_header("QUESTION 3: BOLLINGER BANDS", width=80)
    
    # Compute Bollinger Bands
    middle, upper, lower = compute_bollinger_bands(prices, k=20, num_std=2.0)
    
    # Test strategies
    print("\nTesting Bollinger Band strategies...")
    
    signals_bb_contrarian = generate_bollinger_signals(
        prices, upper, lower, strategy='contrarian'
    )
    backtest_bb_contrarian = backtest_strategy(prices, signals_bb_contrarian)
    
    signals_bb_momentum = generate_bollinger_signals(
        prices, upper, lower, strategy='momentum'
    )
    backtest_bb_momentum = backtest_strategy(prices, signals_bb_momentum)
    
    print("\nContrarian Bollinger Band Strategy:")
    for key, value in backtest_bb_contrarian['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMomentum Bollinger Band Strategy:")
    for key, value in backtest_bb_momentum['metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    # Plot Bollinger Bands
    plot_strategy(
        prices, signals_bb_contrarian,
        indicator=middle, upper_band=upper, lower_band=lower,
        backtest_results=backtest_bb_contrarian,
        title=f"{index_ticker} - Bollinger Bands Contrarian Strategy",
        save_path='results/plots/q3_bollinger_bands.png'
    )
    
    # Export Q3 results
    q3_results = {
        'Contrarian': pd.DataFrame([backtest_bb_contrarian['metrics']]),
        'Momentum': pd.DataFrame([backtest_bb_momentum['metrics']])
    }
    export_results(q3_results, 'results/trading/q3_bollinger_bands.xlsx')
    
    # ========================================================================
    # QUESTION 4: ROLLING WINDOW ANALYSIS
    # ========================================================================
    print_section_header("QUESTION 4: ROLLING WINDOW ANALYSIS", width=80)
    
    print("\nComputing rolling 252-day statistics...")
    rolling_stats = {}
    for ticker, ret in returns.items():
        rolling_stats[ticker] = rolling_window_analysis(ret, window=252)
        print(f"{ticker}: Done")
    
    # Plot rolling statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for ticker in returns.keys():
        rs = rolling_stats[ticker]
        
        axes[0, 0].plot(rs.index, rs['Annual Mean'], label=ticker)
        axes[0, 1].plot(rs.index, rs['Annual Std Dev'], label=ticker)
        axes[1, 0].plot(rs.index, rs['Sharpe Ratio'], label=ticker)
        axes[1, 1].plot(rs.index, rs['Skewness'], label=ticker)
    
    axes[0, 0].set_title('Rolling Annual Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Rolling Annual Volatility')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Rolling Sharpe Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Rolling Skewness')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/q4_rolling_statistics.png', dpi=300)
    print("Rolling statistics plot saved!")
    
    # ========================================================================
    # QUESTION 6: ROLL MODEL
    # ========================================================================
    print_section_header("QUESTION 6: ROLL MODEL", width=80)
    
    print("\nTesting Roll model implications...")
    roll_results = {}
    for ticker in tickers:
        if ticker in data:
            prices_series = data[ticker]['Adj Close']
            
            # Test with actual prices
            results_actual = compute_roll_spread_constant(prices_series, use_log=False)
            
            # Test with log prices
            results_log = compute_roll_spread_constant(prices_series, use_log=True)
            
            roll_results[f'{ticker}_actual'] = results_actual
            roll_results[f'{ticker}_log'] = results_log
            
            print(f"\n{ticker}:")
            print(f"  Estimated Spread (actual): {results_actual['Estimated Spread']:.6f}")
            print(f"  Estimated Spread (log): {results_log['Estimated Spread']:.6f}")
            print(f"  Q6 Implication holds (actual): {results_actual['Q6 Implication Holds']}")
            print(f"  Q6 Implication holds (log): {results_log['Q6 Implication Holds']}")
    
    # Test multiple lags
    print("\n" + "-" * 80)
    print("Testing Roll implication at multiple lags...")
    test_df = test_roll_implication(data[index_ticker]['Adj Close'], nlags=10)
    print(test_df)
    
    # ========================================================================
    # QUESTION 7: TIME-VARYING SPREAD
    # ========================================================================
    print_section_header("QUESTION 7: TIME-VARYING SPREAD", width=80)
    
    print("\nTesting extended Roll model with time-varying spread...")
    tv_results, rolling_spread = compute_roll_spread_timevarying(
        data[index_ticker]['Adj Close']
    )
    
    print("\nTime-Varying Spread Results:")
    for key, value in tv_results.items():
        print(f"  {key}: {value}")
    
    # Plot time-varying spread
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rolling_spread.index, rolling_spread, linewidth=1)
    ax.set_title(f'{index_ticker} - Estimated Time-Varying Spread')
    ax.set_xlabel('Date')
    ax.set_ylabel('Estimated Spread')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/q7_timevarying_spread.png', dpi=300)
    print("Time-varying spread plot saved!")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section_header("ANALYSIS COMPLETE!", width=80)
    print("\nResults saved to:")
    print("  - results/statistics/")
    print("  - results/trading/")
    print("  - results/plots/")
    print("\nPlease review the outputs and answer the theoretical questions:")
    print("  - Question 5: Aggregated returns (theoretical)")
    print("  - Question 8: Event study methodology")


if __name__ == '__main__':
    main()
