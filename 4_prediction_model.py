import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# Connect to the database
engine = create_engine('sqlite:///stocks.db')

# Load the data
df = pd.read_sql('SELECT * FROM apple_features', con=engine)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Predict tomorrow's closing price
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)


features = ['MA_7', 'MA_21', 'Volatility_7', 'Volatility_21', 'RSI']
target = 'Target'

X = df[features]
y = df[target]


from sklearn.model_selection import train_test_split

# Split: last 20% for testing
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📉 MAE: {mae:.2f}")
print(f"📈 R² Score: {r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Random Forest: Actual vs Predicted Closing Prices')
plt.xlabel('Time Index')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_predicted_vs_actual.png')
plt.close()
