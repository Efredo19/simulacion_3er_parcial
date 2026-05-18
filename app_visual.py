import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
import wfdb

# Configuración de la página web
st.set_page_config(page_title="Analizador ECG", page_icon="🩺", layout="wide")

st.title("🩺 Analizador Interactivo de Señales ECG")
st.write("Procesamiento del registro **test05_45s** (Motion Artifact Contaminated ECG Database).")

# 1. Carga de datos
@st.cache_data # Esto es para que no recargue el archivo pesado cada vez que muevas un botón
def cargar_datos():
    try:
        record = wfdb.rdrecord('test05_45s')
    except FileNotFoundError:
        record = wfdb.rdrecord('test05_45s', pn_dir='motion-artifact-ecg/1.0.0')
    return record.p_signal[:, 0], record.fs

ecg_raw, fs = cargar_datos()

# 2. Análisis Temporal en la barra lateral
st.sidebar.header("📋 Información del Registro")
duracion = len(ecg_raw) / fs
st.sidebar.write(f"**Frecuencia de muestreo:** {fs} Hz")
st.sidebar.write(f"**Duración:** {duracion:.2f} segundos")

# Control interactivo: Elegir el método de filtrado
metodo_filtro = st.sidebar.selectbox(
    "Selecciona el método de filtrado:",
    ["neurokit", "biosppy", "pantompkins"]
)

# 3. Procesamiento
ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=fs, method=metodo_filtro)
signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="neurokit")
picos_r = info["ECG_R_Peaks"]

# 4. Gráficos en la Web
tab1, tab2, tab3 = st.tabs(["📈 Señal Temporal", "📊 Análisis de Frecuencia (FFT)", "❤️ Latido Promedio"])

with tab1:
    st.subheader("Filtrado y Detección de Picos QRS")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    tiempo = np.linspace(0, duracion, len(ecg_raw))
    ax1.plot(tiempo, ecg_raw, label="Cruda", color='gray', alpha=0.5)
    ax1.plot(tiempo, ecg_clean, label="Filtrada", color='blue')
    ax1.scatter(tiempo[picos_r], ecg_clean[picos_r], color='red', label='Picos R')
    ax1.set_xlim(0, 10) # Primeros 10 segundos
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    st.subheader("Espectro de Potencia (FFT)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fft_clean = np.fft.rfft(ecg_clean)
    frecuencias = np.fft.rfftfreq(len(ecg_clean), d=1/fs)
    ax2.semilogy(frecuencias, np.abs(fft_clean)**2, color='green')
    ax2.set_xlim(0, 60)
    st.pyplot(fig2)

with tab3:
    st.subheader("Complejo QRS Promedio")
    epochs = nk.ecg_segment(ecg_clean, rpeaks=picos_r, sampling_rate=fs, show=False)
    epochs_df = nk.epochs_to_df(epochs)
    latidos_pivot = epochs_df.pivot(index='Time', columns='Label', values='Signal')
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(latidos_pivot.index, latidos_pivot.values, color='lightgray', alpha=0.5)
    ax3.plot(latidos_pivot.mean(axis=1).index, latidos_pivot.mean(axis=1).values, color='red', linewidth=2)
    st.pyplot(fig3)