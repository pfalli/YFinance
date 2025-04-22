import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

# Step 1: Download Apple stock data
ticker = 'AAPL'
apple_data = yf.download(ticker, start='2010-01-01', end='2024-12-31')
apple_data.reset_index(inplace=True)

# Step 2: Preview
print(apple_data.head())

# Step 3: Create SQLite DB connection
engine = create_engine('sqlite:///stocks.db', echo=True)

# Step 4: Store data in the 'apple_stock_prices' table
apple_data.to_sql('apple_stock_prices', con=engine, if_exists='replace', index=False)

print("Data successfully stored in stocks.db!")
