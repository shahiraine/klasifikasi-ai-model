import streamlit as st
import joblib
import numpy as np

# 1. Load model yang sudah kamu download dari Colab
# Pastikan nama file 'model_ai.pkl' sama dengan yang kamu download
model = joblib.load('model_ai.pkl')

st.title("Aplikasi Prediksi AI")
st.write("Silakan masukkan data input di bawah ini:")

# Contoh Input (Sesuaikan dengan jumlah fitur/kolom dataset kamu)
# Jika modelmu menerima 4 input, buatlah 4 baris input seperti ini
val1 = st.number_input("Input Fitur 1", value=0.0)
val2 = st.number_input("Input Fitur 2", value=0.0)

if st.button("Prediksi Sekarang"):
    # Menyusun data input menjadi array
    data_input = np.array([[val1, val2]])
    
    # Melakukan prediksi
    prediction = model.predict(data_input)
    
    st.success(f"Hasil Prediksi adalah: {prediction[0]}")