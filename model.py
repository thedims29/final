# Import Library
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = pd.read_csv('DataPupuk.csv', sep=';', encoding='latin-1')
X = data['Luas Tanah (m²)'].values.reshape(-1, 1)
y = data[['Banyak Pupuk (kg)', 'Air (liter)', 'Waktu (hari)']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Membuat dataset untuk LSTM
X_lstm = []
y_lstm = []

for i in range(1, len(X_scaled)):
    X_lstm.append(X_scaled[i-1])
    y_lstm.append(y_scaled[i])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Konversi ke tensor PyTorch
X_lstm = torch.FloatTensor(X_lstm).view(-1, 1, 1)  # [samples, time steps, features]
y_lstm = torch.FloatTensor(y_lstm)

# Membuat DataLoader
dataset = TensorDataset(X_lstm, y_lstm)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Membangun model LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 3)  # Output untuk 3 variabel

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Ambil output dari langkah waktu terakhir
        return out

# Inisialisasi model, loss function, dan optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Melatih model
num_epochs = 200
losses = []
maes = []

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    losses.append(loss.item())
    # Hitung MAE
    mae = torch.mean(torch.abs(outputs - targets)).item()
    maes.append(mae)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, MAE: {mae:.4f}')

# Plot loss
plt.plot(losses)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Plot MAE
plt.plot(maes)
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.show()

# Menyimpan model
torch.save(model.state_dict(), 'model_pupuk.pth')