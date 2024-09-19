import time
import requests
import numpy as np
import pandas as pd
from gli_calculator import get_normalized_gli  # Import the GLI calculation

# Lists to store historical data for ratio calculations
price_history = []
sp500_history = []
dxy_history = []
volume_history = []
momentum_30d = []
momentum_90d = []

sharpe_ratios = []
sortino_ratios = []
omega_ratios = []
rsi_values = []
volatility_30d = []
volatility_90d = []
drawdown_30d = []
drawdown_90d = []


def fetch_realized_value():
    # Placeholder for fetching or calculating realized value
    # You might want to replace this with actual logic or a real data source
    return 1000000  # Example static value


def fetch_live_data():
    try:
        # Fetch live price, total supply, volume for Bitcoin
        price_url = 'https://api.coingecko.com/api/v3/simple/price'
        supply_url = 'https://api.coingecko.com/api/v3/coins/bitcoin'
        market_data_url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'

        price_params = {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        price_response = requests.get(price_url, params=price_params)
        price_data = price_response.json()

        supply_response = requests.get(supply_url)
        supply_data = supply_response.json()

        market_data_params = {'vs_currency': 'usd', 'days': '1', 'interval': 'minute'}
        market_data_response = requests.get(market_data_url, params=market_data_params)
        market_data = market_data_response.json()

        # Check if 'total_volumes' is in the market_data response
        if 'total_volumes' in market_data:
            volumes = [x[1] for x in market_data['total_volumes']]
            current_volume = np.mean(volumes) if volumes else None
        else:
            print("Warning: 'total_volumes' not found in market data response.")
            current_volume = None

        current_price = price_data['bitcoin']['usd']
        total_supply = supply_data['market_data']['total_supply']

        return current_price, total_supply, current_volume
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None, None, None


def fetch_sp500_data():
    # Example placeholder for S&P 500 data (replace with live data source)
    return 4450  # Example S&P 500 price (static)


def fetch_dxy_data():
    # Example placeholder for DXY data (replace with live data source)
    return 104  # Example DXY price (static)


def calculate_mvrv_ratio(market_value, realized_value):
    return market_value / realized_value


def sharpe_ratio(returns, risk_free_rate=0):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    return (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0


def sortino_ratio(returns, risk_free_rate=0):
    downside_returns = [r for r in returns if r < 0]
    mean_return = np.mean(returns)
    downside_deviation = np.std(downside_returns) if downside_returns else 0
    return (mean_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0


def omega_ratio(returns, threshold=0):
    gains = [r - threshold for r in returns if r > threshold]
    losses = [threshold - r for r in returns if r < threshold]
    return sum(gains) / sum(losses) if sum(losses) != 0 else float('inf')


def calculate_rsi(prices, period=14):
    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_moving_average(prices, window=365):
    if len(prices) >= window:
        df = pd.DataFrame(prices, columns=['Price'])
        df['MA'] = df['Price'].rolling(window=window).mean()
        return df['MA'].iloc[-1]  # Return the most recent moving average value
    else:
        return None


def calculate_volatility(prices, window):
    if len(prices) >= window:
        df = pd.DataFrame(prices, columns=['Price'])
        df['Return'] = df['Price'].pct_change()
        return df['Return'].rolling(window=window).std().iloc[-1]  # Standard deviation of returns
    else:
        return None


def calculate_max_drawdown(prices, window):
    if len(prices) >= window:
        df = pd.DataFrame(prices, columns=['Price'])
        df['Peak'] = df['Price'].cummax()
        df['Drawdown'] = (df['Price'] - df['Peak']) / df['Peak']
        return df['Drawdown'].rolling(window=window).min().iloc[-1]  # Maximum drawdown
    else:
        return None


def calculate_momentum(prices, window):
    if len(prices) >= window:
        df = pd.DataFrame(prices, columns=['Price'])
        df['Momentum'] = df['Price'].pct_change(window)
        return df['Momentum'].iloc[-1]  # Most recent momentum value
    else:
        return None


def normalize(value, values):
    min_val = np.min(values) if values else 0
    max_val = np.max(values) if values else 1
    if max_val > min_val:
        return 2 * (value - min_val) / (max_val - min_val) - 1
    else:
        return 0  # In case all values are the same


def calculate_correlation(asset_1, asset_2, window=30):
    if len(asset_1) >= window and len(asset_2) >= window:
        asset_1_series = pd.Series(asset_1[-window:])
        asset_2_series = pd.Series(asset_2[-window:])
        correlation = asset_1_series.corr(asset_2_series)
        return correlation
    else:
        return None


def calculate_tpi(current_volume, normalized_momentum_30d, normalized_volatility_30d, normalized_drawdown_30d,
                  normalized_sp500_corr, normalized_dxy_corr, normalized_gli):
    # Weights for different components (can be tuned based on preference)
    volume_weight = 0.2
    momentum_weight = 0.2
    volatility_weight = 0.2
    drawdown_weight = 0.2
    sp500_corr_weight = 0.1
    dxy_corr_weight = 0.1
    gli_weight = 0.1

    # Normalize current volume
    normalized_volume = normalize(current_volume, volume_history)

    # Calculate the weighted average of components
    tpi = (
            volume_weight * normalized_volume +
            momentum_weight * normalized_momentum_30d +
            volatility_weight * normalized_volatility_30d +
            drawdown_weight * normalized_drawdown_30d +
            sp500_corr_weight * normalized_sp500_corr +
            dxy_corr_weight * normalized_dxy_corr +
            gli_weight * normalized_gli
    )

    # Normalize the TPI between -1 (sell) and 1 (buy)
    return max(min(tpi, 1), -1)


def main():
    realized_value = fetch_realized_value()

    try:
        while True:
            current_price, total_supply, current_volume = fetch_live_data()
            if current_price is None or total_supply is None or current_volume is None:
                print("Skipping iteration due to data fetch error.")
                time.sleep(60)  # Wait a minute before retrying
                continue

            sp500_price = fetch_sp500_data()
            dxy_price = fetch_dxy_data()
            market_value = current_price * total_supply

            mvrv_ratio = calculate_mvrv_ratio(market_value, realized_value)
            price_history.append(current_price)
            sp500_history.append(sp500_price)
            dxy_history.append(dxy_price)
            volume_history.append(current_volume)

            # Calculate returns from historical prices
            returns = np.diff(price_history) / price_history[:-1] if len(price_history) > 1 else []

            if len(returns) > 1:
                current_sharpe = sharpe_ratio(returns)
                current_sortino = sortino_ratio(returns)
                current_omega = omega_ratio(returns)

                # Update historical ratios
                sharpe_ratios.append(current_sharpe)
                sortino_ratios.append(current_sortino)
                omega_ratios.append(current_omega)

                # Calculate and normalize RSI
                if len(price_history) > 14:
                    current_rsi = calculate_rsi(price_history)
                    rsi_values.append(current_rsi)
                    normalized_rsi = normalize(current_rsi, rsi_values)
                else:
                    normalized_rsi = None

                # Calculate moving averages, volatility, and drawdowns
                ma_365d = calculate_moving_average(price_history, window=365)
                current_volatility_30d = calculate_volatility(price_history, window=30)
                current_drawdown_30d = calculate_max_drawdown(price_history, window=30)
                current_momentum_30d = calculate_momentum(price_history, window=30)

                # Normalize values
                normalized_volatility_30d = normalize(current_volatility_30d, volatility_30d)
                normalized_drawdown_30d = normalize(current_drawdown_30d, drawdown_30d)
                normalized_momentum_30d = normalize(current_momentum_30d, momentum_30d)

                # Calculate correlations
                normalized_sp500_corr = normalize(calculate_correlation(price_history, sp500_history), sp500_history)
                normalized_dxy_corr = normalize(calculate_correlation(price_history, dxy_history), dxy_history)

                # Retrieve normalized GLI value
                normalized_gli = get_normalized_gli()

                # Calculate TPI
                tpi = calculate_tpi(
                    current_volume,
                    normalized_momentum_30d,
                    normalized_volatility_30d,
                    normalized_drawdown_30d,
                    normalized_sp500_corr,
                    normalized_dxy_corr,
                    normalized_gli
                )

                print(f"Trend Probability Indicator (TPI): {tpi}")

            time.sleep(60)  # Fetch new data every minute

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
