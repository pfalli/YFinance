import pandas as pd
from sqlalchemy import create_engine
import sys # Import sys for error handling
import os # Import os for file check

# Define database path
db_path = 'stocks.db'

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found.")
    print("Please run 'fetch_apple_stock.py' first to create the database.")
    sys.exit(1)

try:
    # Load data
    engine = create_engine(f'sqlite:///{db_path}')
    df = pd.read_sql('SELECT * FROM apple_stock_prices', con=engine)

    # Check if DataFrame is empty
    if df.empty:
        print("Error: The 'apple_stock_prices' table is empty or does not exist.")
        sys.exit(1)

    # --- Fix Column Names ---
    # Rename columns from tuples like ('Date', '') to simple strings
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    # --- End Fix ---

    # Set Date as Index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Sort just in case
    df.sort_index(inplace=True)

    # === Feature Engineering ===

    # Daily Return (%)
    df['Daily_Return'] = df['Close'].pct_change()

    # Moving Averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()

    # Volatility (Rolling STD)
    df['Volatility_7'] = df['Close'].rolling(window=7).std()
    df['Volatility_21'] = df['Close'].rolling(window=21).std()

    # Relative Strength Index (RSI) - optional
    def compute_rsi(data, window=14):
        delta = data.diff()
        # Make sure delta is a Series
        if isinstance(delta, pd.DataFrame):
            delta = delta.iloc[:, 0] # Use first column if it's a DataFrame
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Use Exponential Moving Average for RSI calculation for stability
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1e-10) # Replace 0 with a small number
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = compute_rsi(df['Close'])

    # Drop rows with NaNs from rolling windows/calculations
    df.dropna(inplace=True)

    # Save to a new table
    # Note: The index (Date) will be saved as a column named 'Date' by default
    df.to_sql('apple_features', con=engine, if_exists='replace', index=True)

    print("✅ Feature engineering complete. Saved as 'apple_features' table.")
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")
    # If the error is KeyError, provide more context
    if isinstance(e, KeyError):
        try:
            print("Current DataFrame columns:", df.columns)
        except NameError:
            print("DataFrame 'df' might not have been loaded correctly.")
    sys.exit(1)
