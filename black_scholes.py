import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import requests
from bs4 import BeautifulSoup


def extract_treasury_rate():
    '''
    Parameters: None
    Returns: float - Risk-free rate
    Does: Extracts the risk-free rate from the 10-year Treasury rate webpage.
    '''
    # URL of the webpage
    url = 'https://fred.stlouisfed.org/series/DGS10'

    # Send a GET request to the webpage and parse the HTML content of the webpage
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the risk-free-rate
    # Locate the div with the class 'float-start meta-col col-sm-5 col-5'
    div = soup.find('div', class_='float-start meta-col col-sm-5 col-5')
    span = div.find('span', class_='series-meta-observation-value')

    # Extract the text content from the span
    risk_free_rate = float(span.text.strip()) / 100
    # Return the risk-free rate
    return risk_free_rate


def get_stock():
    '''
    Does: Get user input for stock information including ticker, maturity date or days to maturity,
          and strike price.
    Returns: (str, float, float) - Ticker symbol, time to maturity in years, and strike price.
    '''
    # Get user input for stock ticker
    ticker = input('Enter stock ticker: ')
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    print(f'\nAvailable option expiration dates for {ticker}:')
    print(expiration_dates)

    # Get user input for maturity date or days to maturity
    option = int(input('\nWould you like to input 1. Maturity Date or 2. Days to Maturity? (Enter 1 or 2): '))
    if option == 1:
        mat_date = input('Enter maturity date (mm/dd/yyyy): ')
        current_date = datetime.date.today()
        maturity_date = datetime.datetime.strptime(mat_date, '%m/%d/%Y').date()
        difference = (maturity_date - current_date).days
        days = difference/365
    elif option == 2:
        difference = int(input('Enter days to maturity: '))
        days = difference/365

    # Get strike prices
    maturity_date = datetime.date.today() + datetime.timedelta(days=difference)
    formatted_maturity_date = maturity_date.strftime('%Y-%m-%d')
    opts = stock.option_chain(formatted_maturity_date)
    call_strikes = opts.calls['strike']
    put_strikes = opts.puts['strike']
    strike_prices_df = pd.DataFrame({'Call': call_strikes, 'Put': put_strikes})

    print('\nCurrent Price: $', stock.info['previousClose'])
    original_max_rows = pd.get_option('display.max_rows')
    pd.set_option('display.max_rows', None)
    print('Market Strike Prices:')
    print(strike_prices_df)
    pd.set_option('display.max_rows', original_max_rows)

    strike_price = float(input('\nEnter strike price: $'))
    return ticker, days, strike_price

def get_market_option_price(ticker, days, strike):
    '''
    Parameters: ticker (str) - Ticker symbol of the stock
                days (float) - Time to maturity in years
                strike (float) - Strike price
    Returns: (DataFrame, DataFrame, str) - DataFrames for call options, put options, and maturity date in 'YYYY-MM-DD' format
    Does: Retrieves market option prices for the given stock, time to maturity, and strike price.
    '''
    
    # Get the option chain for the given stock and maturity date
    stock = yf.Ticker(ticker)
    days = days * 365
    current_date = datetime.date.today()
    maturity_date = current_date + datetime.timedelta(days=days)
    formatted_maturity_date = maturity_date.strftime('%Y-%m-%d')
    
    # Get the call and put options for the given strike price   
    option_chain = stock.option_chain(formatted_maturity_date)
    calls = option_chain.calls
    selected_calls = calls[calls['strike'] == strike]
    puts = option_chain.puts
    selected_puts = puts[puts['strike'] == strike]
    
    return selected_calls, selected_puts, formatted_maturity_date

def get_vals(ticker):
    '''
    Parameters: ticker (str) - Ticker symbol of the stock
    Returns: (float, float) - Current stock price and volatility (sigma)
    Does: Retrieves the current stock price and calculates the volatility (sigma) based on one year of historical data.
    '''
    
    # Get the current stock price and calculate the volatility (sigma)
    tick = yf.Ticker(ticker)
    current_price = tick.info['previousClose']
    hist = tick.history(period='1y')
    std_dev = hist['Close'].std()
    sigma = std_dev / 100
    return current_price, sigma

def BS_CALL(S, K, T, r, sigma):
    '''
    Parameters: S (float) - Current stock price
                K (float) - Strike price
                T (float) - Time to maturity in years
                r (float) - Risk-free rate
                sigma (float) - Volatility (sigma)
    Returns: (float) - Black-Scholes call option price
    Does: Calculates the Black-Scholes call option price.
    '''
    
    # Calculate the Black-Scholes call option price
    N = norm.cdf
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*T) / (sigma*(np.sqrt(T)))
    d2 = d1 - (sigma * np.sqrt(T))
    C = S * N(d1) - K * np.exp(-r*T)* N(d2)
    return C

def BS_PUT(S, K, T, r, sigma):
    '''
    Parameters: S (float) - Current stock price
                K (float) - Strike price
                T (float) - Time to maturity in years
                r (float) - Risk-free rate
                sigma (float) - Volatility (sigma)
    Returns: (float) - Black-Scholes put option price
    Does: Calculates the Black-Scholes put option price.
    '''
    
    # Calculate the Black-Scholes put option price
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    P = K*np.exp(-r*T)*N(-d2) - S*N(-d1)
    return P

def implied_volatility(m_call, m_put, S, K, T, r, initial_sigma):
    '''
    Parameters: m_call (float) - Market call option price
                m_put (float) - Market put option price
                S (float) - Current stock price
                K (float) - Strike price
                T (float) - Time to maturity in years
                r (float) - Risk-free rate
                initial_sigma (float) - Initial guess for volatility (sigma)
    Returns: (float, float) - Implied volatilities for call and put options
    Does: Calculates the implied volatilities for call and put options using the Black-Scholes formula.
          If it can't solve for a volatility, it uses a fallback method.
    '''
    
    def original_method(market_price, option_func, sigma):
        '''
        Parameters: market_price (float) - Market option price
                    option_func (function) - Black-Scholes call or put function
                    sigma (float) - Initial guess for volatility (sigma)
        Returns: float - Implied volatility
        Does: Calculates the implied volatility using the original method.
        '''
        
        tolerance = 0.01
        max_iterations = 100
        for _ in range(max_iterations):
            calculated_price = option_func(S, K, T, r, sigma)
            if abs(calculated_price - market_price) < tolerance:
                return round(sigma, 4)
            elif calculated_price > market_price:
                sigma -= sigma * 0.05
            else:
                sigma += sigma * 0.05
        return None

    def fallback_method(market_price, option_func):
        '''
        Parameters: market_price (float) - Market option price
                    option_func (function) - Black-Scholes call or put function
        Returns: float - Implied volatility
        Does: Calculates the implied volatility using a fallback method.
        '''
        lower_bound, upper_bound = 0.0001, 5.0
        for _ in range(50):  # Limit iterations for performance
            sigma = (lower_bound + upper_bound) / 2
            price = option_func(S, K, T, r, sigma)
            
            if abs(price - market_price) < 0.01:  # Check if price is accurate to the cent
                return round(sigma, 4)
            
            if price < market_price:
                lower_bound = sigma
            else:
                upper_bound = sigma
        
        # If we couldn't find an exact match, return the closest
        lower_price = abs(option_func(S, K, T, r, lower_bound) - market_price)
        upper_price = abs(option_func(S, K, T, r, upper_bound) - market_price)
        return round(lower_bound if lower_price < upper_price else upper_bound, 4)

    call_vol = original_method(m_call, BS_CALL, initial_sigma)
    if call_vol is None:
        call_vol = fallback_method(m_call, BS_CALL)

    put_vol = original_method(m_put, BS_PUT, initial_sigma)
    if put_vol is None:
        put_vol = fallback_method(m_put, BS_PUT)

    return call_vol, put_vol

def plot_implied_volatility(ticker, T, S, r, original_sigma, K):
    stock = yf.Ticker(ticker)
    formatted_maturity_date = (datetime.date.today() + datetime.timedelta(days=T * 365)).strftime('%Y-%m-%d')
    option_chain = stock.option_chain(formatted_maturity_date)
    calls = option_chain.calls
    puts = option_chain.puts
    current_price = S
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    def plot_option_volatilities(options, ax, color, label, option_type):
        strikes = []
        market_vols = []
        calculated_vols = []
        
        for _, row in options.iterrows():
            strike = row['strike']
            market_price = row['lastPrice']
            market_iv = row['impliedVolatility']
            
            if option_type == 'call':
                vol, _ = implied_volatility(market_price, 0, S, strike, T, r, original_sigma)
            else:
                _, vol = implied_volatility(0, market_price, S, strike, T, r, original_sigma)
            
            strikes.append(strike)
            market_vols.append(market_iv)
            calculated_vols.append(vol if vol is not None and vol > 0.0001 else np.nan)
            
            if vol is not None and vol > 0.0001:
                ax.scatter(strike, vol, color=color, label=label if strike == options['strike'].iloc[0] else "")
            
        ax.scatter(strikes, market_vols, color='red', marker='x', label=f'Market {label}')
        
        return pd.DataFrame({
            'Strike': strikes,
            'Market_IV': market_vols,
            'Calculated_IV': calculated_vols
        })

    put_vol = plot_option_volatilities(puts, ax2, 'green', 'Put Volatility', 'put')
    call_vol = plot_option_volatilities(calls, ax1, 'blue', 'Call Volatility', 'call')

    for ax in (ax1, ax2):
        ax.axhline(y=original_sigma, color='r', linestyle='-', label='Used Sigma')
        ax.axvline(x=current_price, color='green', linestyle='--', label='Current Price')
        ax.axvline(x=K, color='orange', linestyle='--', label='Chosen Strike Price')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Implied Volatility')
        ax.legend()

    ax1.set_title('Call Option Implied Volatility')
    ax2.set_title('Put Option Implied Volatility')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

    return put_vol, call_vol

def plot_compare_prices_with_maturities(ticker, strike, S, r, sigma):
    '''
    Parameters: ticker (str) - Ticker symbol of the stock
                strike (float) - Strike price
                S (float) - Current stock price
                r (float) - Risk-free rate
                sigma (float) - Volatility (sigma)
    Does: Plots market and Black-Scholes option prices for call and put options with different maturities.
    '''
    
    # Get options dates
    stock = yf.Ticker(ticker)
    options_dates = stock.options

    market_call_prices = []
    bs_call_prices = []
    market_put_prices = []
    bs_put_prices = []
    valid_dates = []

    # Get market and Black-Scholes option prices for call and put options with different maturities
    for date in options_dates:
        try:
            option_chain = stock.option_chain(date)
            calls = option_chain.calls
            puts = option_chain.puts

            filtered_call = calls[calls['strike'] == strike]
            filtered_put = puts[puts['strike'] == strike]

            if not filtered_call.empty and not filtered_put.empty:
                T = (datetime.datetime.strptime(date, '%Y-%m-%d').date() - datetime.date.today()).days / 365

                market_call_price = filtered_call['lastPrice'].iloc[0]
                market_call_prices.append(market_call_price)
                bs_call_prices.append(BS_CALL(S, strike, T, r, sigma))

                market_put_price = filtered_put['lastPrice'].iloc[0]
                market_put_prices.append(market_put_price)
                bs_put_prices.append(BS_PUT(S, strike, T, r, sigma))

                valid_dates.append(date)
        except:
            pass  # Handling cases where the strike price doesn't exist

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(valid_dates, market_call_prices, 'o-', label='Market Call Price')
    plt.plot(valid_dates, bs_call_prices, 'x-', label='BS Call Price')
    plt.xlabel('Maturity Date')
    plt.ylabel('Price')
    plt.title('Call Option Prices')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_dates, market_put_prices, 'o-', label='Market Put Price')
    plt.plot(valid_dates, bs_put_prices, 'x-', label='BS Put Price')
    plt.xlabel('Maturity Date')
    plt.ylabel('Price')
    plt.title('Put Option Prices')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
    plt.pause(0.001)

def plot_sigma_change(S, K, T, r):
    '''
    Parameters: S (float) - Current stock price
                K (float) - Strike price
                T (float) - Time to maturity in years
                r (float) - Risk-free rate
    Does: Plots the option values (call and put) as the volatility (sigma) changes.
    '''
    
    # Use a more realistic range for sigma
    Sigmas = np.arange(0.01, 1.01, 0.01)
    
    # Calculate option values
    calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
    puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(Sigmas, calls, label='Call Value', color='blue')
    plt.plot(Sigmas, puts, label='Put Value', color='red')
    
    # Add vertical line for at-the-money volatility
    atm_vol = 0.2  # You can adjust this or calculate it based on market data
    plt.axvline(x=atm_vol, color='green', linestyle='--', label='Typical ATM Volatility')
    
    # Improve labels and title
    plt.xlabel('Volatility (Sigma)')
    plt.ylabel('Option Value')
    plt.title('Option Values as Volatility Changes')
    
    # Add gridlines for better readability
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Improve legend
    plt.legend(loc='upper left')
    
    # Add text to explain the parameters
    info_text = f'S={S}, K={K}, T={T:.2f} years, r={r:.2%}'
    plt.text(0.95, 0.05, info_text, horizontalalignment='right', verticalalignment='bottom', 
             transform=plt.gca().transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)

def plot_time_change(S, K, r, sigma):
    '''
    Parameters: S (float) - Current stock price
                K (float) - Strike price
                r (float) - Risk-free rate
                sigma (float) - Volatility (sigma)
    Does: Plots the option values (call and put) as the time to maturity (T) changes.
    '''
    
    # Plot the option values (call and put) as the time to maturity (T) changes
    Ts = np.arange(0.0001, 1, 0.01)
    calls = [BS_CALL(S, K, T, r, sigma) for T in Ts]
    puts = [BS_PUT(S, K, T, r, sigma) for T in Ts]
    plt.plot(Ts, calls, label='Call Value')
    plt.plot(Ts, puts, label='Put Value')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Value')
    plt.title('Value of Option as Time to Maturity Changes')
    plt.gca().invert_xaxis()  # Invert the x-axis
    plt.legend()
    plt.show(block=False)
    plt.pause(0.001)


def main():
    '''
    Parameters: None
    Returns: None
    Does: Calculates the price of an option using the Black-Scholes model and compares it to the market price.
    '''

    # Get the stock data
    ticker, T, K = get_stock()
    s, sigma = get_vals(ticker)
    r = extract_treasury_rate()
    call_price = BS_CALL(s, K, T, r, sigma)
    put_price = BS_PUT(s, K, T, r, sigma)
    print('\nCalculated:')
    print(pd.DataFrame({'Call Price': [f'{call_price:.2f}'], ' Put Price': [f'{put_price:.2f}']}))

    # Get the market data
    call, put, mat_date = get_market_option_price(ticker, T, K)
    print('\nMarket Option Prices for', ticker, 'with strike price', K, 'and expiration on', mat_date)
    
    if not call.empty:
        print('Call:')
        print(call[['strike', 'lastPrice']])
        m_call_price = call['lastPrice'].values[0]
        call_vol, _ = implied_volatility(m_call_price, 0, s, K, T, r, sigma)
    else:
        print('No call options available for this strike price.')
        call_vol = None

    if not put.empty:
        print('Put:')
        print(put[['strike', 'lastPrice']])
        m_put_price = put['lastPrice'].values[0]
        _, put_vol = implied_volatility(0, m_put_price, s, K, T, r, sigma)
    else:
        print('No put options available for this strike price.')
        put_vol = None

    # Print the implied volatility
    print('\nImplied Volatility')
    print(pd.DataFrame({'Call': [call_vol], 'Put': [put_vol]}))


    # Plot the results
    plot_sigma_change(s, K, T, r)
    plot_time_change(s, K, r, sigma)
    put_vol, call_vol = plot_implied_volatility(ticker, T, s, r, sigma, K)
    put_correlation = put_vol['Market_IV'].corr(put_vol['Calculated_IV'])
    call_correlation = call_vol['Market_IV'].corr(call_vol['Calculated_IV'])
    
    # Calculate correlation for put and call options
    print('\n IV Correlation between Black-Scholes and Market:')
    print(f"Put Option IV Correlation: {put_correlation}")
    print(f"Call Option IV Correlation: {call_correlation}")
    
    plot_compare_prices_with_maturities(ticker, K, s, r, sigma)

if __name__ == '__main__':
    main()

