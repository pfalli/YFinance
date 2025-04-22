import pandas as pd
from sqlalchemy import create_engine, exc
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import the os module
import sys # Import sys to exit if needed

# Define database path
db_path = 'stocks.db'

# Check if the database file exists
if not os.path.exists(db_path):
    print(f"Error: Database file '{db_path}' not found.")
    print("Please run 'fetch_apple_stock.py' first to create the database.")
    sys.exit(1) # Exit the script if the file doesn't exist

try:
    # Create DB engine
    engine = create_engine(f'sqlite:///{db_path}')

    # Load Apple stock data
    df = pd.read_sql('SELECT * FROM apple_stock_prices', con=engine)

    # Check if DataFrame is empty
    if df.empty:
        print("Error: The 'apple_stock_prices' table is empty or does not exist.")
        sys.exit(1)

    # --- Fix Column Names ---
    # The columns are loaded as tuples, e.g., ('Date', ''), ('Close', 'AAPL')
    # We rename them to simple strings for easier access.
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    # --- End Fix ---


    # Preview
    print(df.head())
    print(df.info())

    # Convert Date & Set as Index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)


    #Plot Closing Price Over Time
    plt.figure(figsize=(12, 6))
    df['Close'].plot()
    plt.title('Apple Closing Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    # plt.show()
    plt.savefig('apple_closing_price.png') # Save the plot
    plt.close() # Close the plot figure to free memory

    # Plot Volume Traded Over Time
    plt.figure(figsize=(12, 6))
    df['Volume'].plot(color='orange')
    plt.title('Apple Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid(True)
    # plt.show()
    plt.savefig('apple_trading_volume.png') # Save the plot
    plt.close() # Close the plot figure

    #  Correlation Heatmap
    plt.figure(figsize=(8, 6))
    # Ensure only numeric columns are used for correlation
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    # plt.show()
    plt.savefig('apple_correlation_matrix.png') # Save the plot
    plt.close() # Close the plot figure

except exc.SQLAlchemyError as e:
    print(f"Database error occurred: {e}")
    sys.exit(1)
except FileNotFoundError: # Although checked, good practice to keep
    print(f"Error: Database file '{db_path}' not found during operation.")
    sys.exit(1)
except KeyError as e: # Catch the specific KeyError
    print(f"An error occurred accessing column: {e}")
    print("This might be due to unexpected column names in the database.")
    print("Current columns:", df.columns) # Print current columns for debugging
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)