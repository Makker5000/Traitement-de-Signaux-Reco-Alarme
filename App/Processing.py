import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Fonction pour charger un fichier WAV
def load_wav(file_path):
    fs, signal = wavfile.read(file_path)
    if len(signal.shape) > 1:  # Convertir en mono si stéréo
        signal = np.mean(signal, axis=1)
    signal = signal / np.max(np.abs(signal))  # Normalisation
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

# Fonction pour afficher le spectre temporel et fréquentiel
def plot_signal_and_spectrogram(fs, signal, title, ax_time, ax_freq, ax_spec):
    # Spectre temporel
    t = np.arange(len(signal)) / fs
    ax_time.plot(t, signal)
    ax_time.set_title(f"Time Domain: {title}")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")

    # Spectre fréquentiel
    freqs = fftfreq(len(signal), 1/fs)
    fft_values = np.abs(fft(signal))
    ax_freq.plot(freqs[:len(freqs)//2], fft_values[:len(fft_values)//2])
    ax_freq.set_title(f"Frequency Domain: {title}")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")

    # Spectrogramme
    f, t_spec, Sxx = spectrogram(signal, fs)
    ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    ax_spec.set_title(f"Spectrogram: {title}")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")

# Programme principal
def process_and_plot(file_path):
    # Charger le signal
    fs, signal = load_wav(file_path)
    fs = 48000
    
    # Définir les bandes de fréquences
    # lowcuts = [1300, 1400, 1490, 1610, 1715,  3920.0, 4215.0, 4485.0, 4840.0, 5155.0]
    # highcuts = [1320, 1420, 1510, 1630, 1735,  3950.0, 4245.0, 4515.0, 4870.0, 5185.0]
    lowcuts = [3900.0, 4200.0, 4470.0, 4830.0, 5145.0]
    highcuts = [3950.0, 4250.0, 4520.0, 4880.0, 5195.0]
    



    # Appliquer les filtres
    filtered_signal = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    # Configuration de la figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plot_signal_and_spectrogram(fs, signal, "Original Signal", axes[0, 0], axes[1, 0], axes[2, 0])
    plot_signal_and_spectrogram(fs, filtered_signal, "Filtered Signal", axes[0, 1], axes[1, 1], axes[2, 1])

    plt.tight_layout()
    plt.show()

# Remplacez 'your_audio_file.wav' par le chemin du fichier WAV à traiter.
process_and_plot('Ressources\Sons-de-Test\Hyper-discussion_1.wav')
