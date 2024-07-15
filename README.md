# Black-Scholes Model Comparison Tool

This project provides a comprehensive tool for comparing the Black-Scholes option pricing model with real market data. It offers various analyses and visualizations to help understand option pricing dynamics and the model's performance against market prices.

## Features

- **Black-Scholes Calculation**: Calculates option prices using the Black-Scholes formula for both call and put options.
- **Market Data Retrieval**: Fetches real-time market data for options using the yfinance library.
- **Implied Volatility Calculation**: Computes implied volatility from market prices using both original and fallback methods.
- **Risk-Free Rate Extraction**: Automatically extracts the current 10-year Treasury rate as a proxy for the risk-free rate.
- **Visualizations**:
  - Option value changes with volatility
  - Option value changes with time to maturity
  - Implied volatility smile/skew for both call and put options
  - Comparison of Black-Scholes prices with market prices across different maturities
- **Correlation Analysis**: Calculates the correlation between Black-Scholes implied volatilities and market implied volatilities for both call and put options.

## Requirements

- Python 3.x
- Libraries: numpy, pandas, scipy, matplotlib,
- yfinance, requests, beautifulsoup4

## Installation

1. Clone this repository: git clone https://github.com/yourusername/black-scholes-comparison.git
2. Install required packages: pip install -r requirements.txt

## Usage

Run the main script: python black_scholes_comparison.py
Follow the prompts to input:
- Stock ticker
- Option expiration date or days to maturity
- Strike price

The program will then:
1. Calculate Black-Scholes prices
2. Fetch market prices
3. Calculate implied volatilities
4. Generate various plots
5. Compute correlations between model and market implied volatilities

## Key Functions

- `BS_CALL` and `BS_PUT`: Calculate Black-Scholes prices for call and put options
- `implied_volatility`: Calculates implied volatility using both original and fallback methods
- `plot_sigma_change`: Visualizes how option values change with volatility
- `plot_time_change`: Shows how option values change with time to maturity
- `plot_implied_volatility`: Plots the implied volatility smile/skew
- `plot_compare_prices_with_maturities`: Compares Black-Scholes and market prices across different maturities
