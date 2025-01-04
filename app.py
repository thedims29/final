import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Definisikan model LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 3)  # Output untuk 3 variabel

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Ambil output dari langkah waktu terakhir
        return out

# Load model
model = LSTMModel()
model.load_state_dict(torch.load('model_pupuk.pth'))
model.eval()  # Set model ke evaluasi mode

# Pastikan direktori 'tmp/' ada
os.makedirs('/tmp', exist_ok=True)

# Load data untuk scaler
data = pd.read_csv('DataPupuk.csv', sep=';', encoding='latin-1')
X = data['Luas Tanah (m²)'].values.reshape(-1, 1)
y = data[['Banyak Pupuk (kg)', 'Air (liter)', 'Waktu (hari)']].values

# Normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X)
scaler_y.fit(y)

# Judul Aplikasi
st.title('Prediksi Kebutuhan Pupuk Menggunakan Streamlit')

# Input Luas Tanah
tanah_input = st.number_input('Masukkan Luas Tanah (m²):', min_value=0.0, step=0.1)

if st.button('Prediksi'):
    # Normalisasi input
    luas_tanah_scaled = scaler_X.transform([[tanah_input]])
    luas_tanah_scaled = torch.FloatTensor(luas_tanah_scaled).view(1, 1, 1)  # [batch_size, time_steps, features]

    # Prediksi
    with torch.no_grad():  # Nonaktifkan gradien untuk prediksi
        prediksi_scaled = model(luas_tanah_scaled)
        prediksi = scaler_y.inverse_transform(prediksi_scaled.numpy())

    banyak_pupuk = prediksi[0][0]
    air = prediksi[0][1]
    waktu = prediksi[0][2]

    # Tampilkan Hasil Prediksi
    st.success(f'Prediksi untuk Luas Tanah {tanah_input} m²:')
    st.write(f'Banyak Pupuk: {banyak_pupuk:.2f} kg')
    st.write(f'Air: {air:.2f} liter')
    st.write(f'Waktu: {waktu:.2f} hari')