# gli_calculator.py
import numpy as np

# Sample placeholder values for central bank balance sheets (in trillions USD)
central_bank_data = {
    "FED": 8.1,  # Federal Reserve System balance sheet
    "TGA": 0.5,  # Treasury General Account (to be subtracted)
    "RRP": 1.7,  # Reverse Repurchase Agreements (to be subtracted)
    "ECB": 8.6,  # European Central Bank
    "PBC": 5.6,  # People's Bank of China
    "BOJ": 7.1,  # Bank of Japan
    "BOE": 1.3,  # Bank of England
    "BOC": 0.5,  # Bank of Canada
    "RBA": 0.3,  # Reserve Bank of Australia
    "RBI": 0.8,  # Reserve Bank of India
    "SNB": 1.2,  # Swiss National Bank
    "CBR": 0.6,  # Central Bank of the Russian Federation
    "BCB": 0.7,  # Central Bank of Brazil
    "BOK": 0.4,  # Bank of Korea
    "RBNZ": 0.1,  # Reserve Bank of New Zealand
    "Riksbank": 0.2,  # Sweden's Central Bank
    "BNM": 0.1  # Central Bank of Malaysia
}


# Function to calculate the Global Liquidity Index (GLI)
def calculate_global_liquidity_index(central_bank_data):
    # Adjust U.S. liquidity (subtract TGA and RRP from FED)
    fed_adjusted = central_bank_data["FED"] - central_bank_data["TGA"] - central_bank_data["RRP"]

    # Sum the balances of all included central banks
    total_liquidity = fed_adjusted + sum([v for k, v in central_bank_data.items() if k not in ["FED", "TGA", "RRP"]])

    return total_liquidity


# Function to normalize the GLI between -1 and 1
def normalize_gli(gli_value, historical_gli_values):
    min_gli = np.min(historical_gli_values)
    max_gli = np.max(historical_gli_values)

    if max_gli > min_gli:
        normalized_gli = 2 * (gli_value - min_gli) / (max_gli - min_gli) - 1
    else:
        normalized_gli = 0  # If historical values are constant or unavailable

    return normalized_gli


# Historical GLI values for normalization (can be fetched or updated regularly)
historical_gli_values = [45, 47, 48, 46, 49, 50]  # Example historical GLI values


# Function to fetch current GLI and normalize it
def get_normalized_gli():
    current_gli = calculate_global_liquidity_index(central_bank_data)
    normalized_gli = normalize_gli(current_gli, historical_gli_values)
    return normalized_gli


if __name__ == "__main__":
    normalized_gli = get_normalized_gli()
    print(f"Normalized GLI: {normalized_gli}")