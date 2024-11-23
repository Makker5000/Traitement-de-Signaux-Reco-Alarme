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
def apply_bandpass_filter(signal, fs, lowcut, highcut, order=4):
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

# Fonction pour calculer la transformée de Fourier
def compute_fft(signal, fs):
    fft_result = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1 / fs)
    return freqs[:len(freqs) // 2], fft_result[:len(freqs) // 2]

# Fonction pour sauvegarder un fichier WAV
def save_wav(file_path, fs, signal):
    scaled_signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(file_path, fs, scaled_signal)

# Fonction pour tracer les résultats
def plot_results(time, original_signal, reconstructed_signal, freqs, fft_reconstructed):
    plt.figure(figsize=(14, 10))

    # Signal Original (Temporel)
    plt.subplot(3, 1, 1)
    plt.plot(time, original_signal, label="Signal Original", color="green")
    plt.title("Signal Original (Temporel)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Signal Reconstruit (Temporel)
    plt.subplot(3, 1, 2)
    plt.plot(time, reconstructed_signal, label="Signal Reconstruit", color="orange")
    plt.title("Signal Reconstruit (Temporel)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    # Spectre Fréquentiel du Signal Reconstruit
    plt.subplot(3, 1, 3)
    plt.plot(freqs, fft_reconstructed, label="Spectre Reconstruit", color="blue")
    plt.title("Spectre Fréquentiel (Signal Reconstruit)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()





# Programme principal
def Processing(chemin_fichier): # ,chemin_destination) : # ex : "../Ressources/Sons-De-Test/Hyper-chien.waw" && "../Output/son_traiter.waw"
    file_path = chemin_fichier  # Remplacez par le chemin du fichier à traiter
    fs, signal = load_wav(file_path)

    # Définir les bandes de fréquences (domianntes)
    lowcuts = [1300, 1400, 1490, 1610, 1715, 3920.0, 4215.0, 4485.0, 4840.0, 5155.0]
    highcuts = [1320, 1420, 1510, 1630, 1735, 3950.0, 4245.0, 4515.0, 4870.0, 5185.0]

    # Appliquer les filtres et combiner les résultats
    signal_reconstructed = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    # Calculer la transformée de Fourier
    freqs, fft_reconstructed = compute_fft(signal_reconstructed, fs)

    # Enregistrer le signal reconstruit dans un fichier WAV
     
            # output_file = chemin_destination
            # save_wav(output_file, fs, signal_reconstructed)

    # Tracer les résultats
    time = np.linspace(0, len(signal) / fs, len(signal))
    plot_results(time, signal, signal_reconstructed, freqs, fft_reconstructed)

    # print(f"Le signal reconstruit a été enregistré dans le fichier : {output_file}")

    return signal_reconstructed

