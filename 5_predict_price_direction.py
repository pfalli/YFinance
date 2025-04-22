import pandas as pd
from sqlalchemy import create_engine
import numpy as np # Make sure numpy is imported

# Load data
engine = create_engine('sqlite:///stocks.db')
df = pd.read_sql('SELECT * FROM apple_features', con=engine)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Predict whether price goes up the next day
df['Target_UpDown'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop last row (no label)
df.dropna(inplace=True)

features = ['MA_7', 'MA_21', 'Volatility_7', 'Volatility_21', 'RSI']
X = df[features]
y_class = df['Target_UpDown']  # for classification
y_reg = df['Close'].shift(-1)  # for regression comparison

# Remove last row for y_reg too
y_reg = y_reg.dropna()
X = X.iloc[:-1]
y_class = y_class.iloc[:-1]


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_class, shuffle=False, test_size=0.2)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, shuffle=False, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train_cls)
y_pred_cls = clf.predict(X_test_scaled)

print("🎯 Classification Report (Random Forest):")
print(classification_report(y_test_cls, y_pred_cls))
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_cls))


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

reg = LinearRegression()
reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = reg.predict(X_test_scaled)

mse = mean_squared_error(y_test_reg, y_pred_reg) # Calculate MSE
rmse = np.sqrt(mse) # Calculate RMSE by taking the square root
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n📈 Linear Regression RMSE: {rmse:.2f}")
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n📈 Linear Regression RMSE: {rmse:.2f}")
print(f"📊 Linear Regression R² Score: {r2:.4f}")
