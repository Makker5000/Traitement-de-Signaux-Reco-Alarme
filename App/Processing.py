import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Fonction pour charger un fichier WAV
def load_wav(file_path):
    fs, signal = wavfile.read(file_path)
    if len(signal.shape) > 1:  #si stéréo, convertir en mono
        signal = np.mean(signal, axis=1)
    signal = signal / np.max(np.abs(signal))  
    return fs, signal

# Fonction pour appliquer un filtre passe-bande
def apply_bandpass_filter(signal, fs, lowcut, highcut, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Fonction pour combiner plusieurs filtres passe-bande
def combine_bandpass_filters(signal, fs, lowcuts, highcuts):
    filtered_signals = []
    for lowcut, highcut in zip(lowcuts, highcuts):
        filtered_signal = apply_bandpass_filter(signal, fs, lowcut, highcut)
        filtered_signals.append(filtered_signal)
    return np.sum(filtered_signals, axis=0)


# Programme principal
def Processing(chemin_fichier): # ,chemin_destination) : # ex : "../Ressources/Sons-De-Test/Hyper-chien.waw" && "../Output/son_traiter.waw"
    file_path = chemin_fichier  # Remplacez par le chemin du fichier à traiter
    fs, signal = load_wav(file_path)
    fs = 48000
    # Définir les bandes de fréquences (domianntes)
    lowcuts = [1300, 1400, 1490, 1610, 1715, 3920.0, 4215.0, 4485.0, 4840.0, 5155.0]
    highcuts = [1320, 1420, 1510, 1630, 1735, 3950.0, 4245.0, 4515.0, 4870.0, 5185.0]

    # Appliquer les filtres et combiner les résultats
    signal_reconstructed = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    return fs, signal_reconstructed

