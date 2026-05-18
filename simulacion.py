import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import neurokit2 as nk
import wfdb

# ==========================================
# 1. CARGA Y SELECCIÓN DE DATOS
# ==========================================
# Reemplaza 'path_to_file/test05_45s' con la ruta local donde descargaste el archivo
# O utiliza la API de wfdb para descargarlo directamente si tienes conexión.
try:
    # Intentamos cargar el registro localmente (debe estar el .dat y .hea en el mismo directorio)
    record = wfdb.rdrecord('test05_45s')
    signal_data = record.p_signal
    fs = record.fs  # Frecuencia de muestreo
except FileNotFoundError:
    print("Archivo local no encontrado. Intentando descargar directamente de PhysioNet...")
    record = wfdb.rdrecord('test05_45s', pn_dir='motion-artifact-ecg/1.0.0')
    signal_data = record.p_signal
    fs = record.fs

# Seleccionamos un único canal de ECG (el primero disponible, índice 0)
ecg_raw = signal_data[:, 0]

# ==========================================
# 2. ANÁLISIS TEMPORAL
# ==========================================
num_muestras = len(ecg_raw)
duracion_segundos = num_muestras / fs

print("--- ANÁLISIS TEMPORAL ---")
print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Número total de muestras: {num_muestras}")
print(f"Duración total del registro: {duracion_segundos:.2f} segundos")
print("-" * 25)

# Crear vector de tiempo para graficar
tiempo = np.linspace(0, duracion_segundos, num_muestras)

# ==========================================
# 3. FILTRADO DE LA SEÑAL
# ==========================================
# Usamos NeuroKit2 para limpiar la señal de artefactos de movimiento y ruido de línea (50/60 Hz)
# El método 'neurokit' aplica un filtro pasa-bandas Butterworth de orden 5 (0.5 a 40 Hz)
ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=fs, method="neurokit")

# ==========================================
# 4. ANÁLISIS DE FRECUENCIA (FFT)
# ==========================================
# Calculamos la FFT de la señal cruda y de la señal filtrada para comparar
fft_raw = np.fft.rfft(ecg_raw)
fft_clean = np.fft.rfft(ecg_clean)
frecuencias = np.fft.rfftfreq(num_muestras, d=1/fs)

# Magnitud del espectro de potencia
potencia_raw = np.abs(fft_raw) ** 2
potencia_clean = np.abs(fft_clean) ** 2

# ==========================================
# 5. DETECCIÓN DE EVENTOS (PICOS QRS)
# ==========================================
# NeuroKit2 detecta los picos R automáticamente usando el algoritmo seleccionado (ej. 'neurokit')
signals, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="neurokit")
picos_r = info["ECG_R_Peaks"]

# ==========================================
# 6. SEGMENTACIÓN Y PROMEDIADO
# ==========================================
# Extraemos los ciclos cardíacos (epochs) alrededor de cada pico R
# Definimos ventanas: -0.2 segundos antes del pico R y 0.4 segundos después
epochs = nk.ecg_segment(ecg_clean, rpeaks=picos_r, sampling_rate=fs, show=False)

# Convertimos los epochs a un DataFrame para promediar fácilmente
epochs_df = nk.epochs_to_df(epochs)

# Pivotamos el DataFrame para tener los tiempos como índice y cada latido como una columna
latidos_pivot = epochs_df.pivot(index='Time', columns='Label', values='Signal')

# Calculamos el latido promedio (media a lo largo de las columnas)
latido_promedio = latidos_pivot.mean(axis=1)

# ==========================================
# VISUALIZACIÓN DE RESULTADOS (GRAFICAS)
# ==========================================
plt.figure(figsize=(14, 10))

# Gráfica 1: Señal en el Tiempo (Comparativa)
plt.subplot(3, 1, 1)
plt.plot(tiempo, ecg_raw, label="Señal Cruda (Con Artefactos)", color='gray', alpha=0.6)
plt.plot(tiempo, ecg_clean, label="Señal Filtrada", color='blue', alpha=0.9)
plt.scatter(tiempo[picos_r], ecg_clean[picos_r], color='red', marker='o', label='Picos R Detectados')
plt.title("Análisis Temporal: Filtrado y Detección de Picos QRS")
plt.xlabel("Tiempo (segundos)")
plt.ylabel("Amplitud (mV)")
plt.xlim(0, 10)  # Mostramos los primeros 10 segundos para mejor detalle visual
plt.legend(loc="upper right")
plt.grid(True)

# Gráfica 2: Espectro de Frecuencia (FFT)
plt.subplot(3, 1, 2)
plt.semilogy(frecuencias, potencia_raw, label="Espectro Crudo", color='gray', alpha=0.6)
plt.semilogy(frecuencias, potencia_clean, label="Espectro Filtrado", color='green', alpha=0.9)
plt.title("Análisis de Frecuencia: Espectro de Potencia (FFT)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (log)")
plt.xlim(0, 60)  # El espectro útil del ECG suele estar por debajo de los 50-60 Hz
plt.legend(loc="upper right")
plt.grid(True)

# Gráfica 3: Segmentación y Latido Promedio
plt.subplot(3, 1, 3)
# Graficamos todos los latidos individuales en gris tenue
plt.plot(latidos_pivot.index, latidos_pivot.values, color='lightgray', alpha=0.5)
# Graficamos el latido promedio en un color fuerte
plt.plot(latido_promedio.index, latido_promedio.values, color='red', linewidth=2.5, label='Latido Promedio')
plt.title("Segmentación: Complejo QRS / Latido Promedio")
plt.xlabel("Tiempo relativo al pico R (segundos)")
plt.ylabel("Amplitud (mV)")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()