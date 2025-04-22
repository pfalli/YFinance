import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os

# ========== CONFIG ==========
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
HIDDEN_SIZE = 64
FEATURES = ['MA_7', 'MA_21', 'Volatility_7', 'Volatility_21', 'RSI']
TARGET = 'Close'
DB_PATH = 'stocks.db'
TABLE = 'apple_features'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================

# === Load data ===
engine = create_engine(f'sqlite:///{DB_PATH}')
df = pd.read_sql(f'SELECT * FROM {TABLE}', con=engine)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# === Target: next day Close ===
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# === Scale features ===
X_raw = df[FEATURES].values
y_raw = df['Target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === Create sequences ===
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_raw, SEQ_LEN)

# === Train/test split ===
split_idx = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# === Torch datasets ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

# === LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take output at last time step
        return out.squeeze()

model = LSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training ===
print("🧠 Training LSTM model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\n📊 LSTM Evaluation:")
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.4f}")

# === Visualization ===
plt.figure(figsize=(14, 6))
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("LSTM: Actual vs Predicted Apple Close Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

if not os.path.exists("plots"):
    os.makedirs("plots")

plt.savefig("plots/lstm_predicted_vs_actual.png")
plt.close()
print("📈 Plot saved to 'plots/lstm_predicted_vs_actual.png'")
