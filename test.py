import numpy as np
from scipy.fftpack import fft
from scipy.signal import resample
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Fonction pour normaliser un signal entre -1 et 1
def normalize_signal(signal):
    return signal / np.max(np.abs(signal))

# Fonction pour calculer les fréquences dominantes
def get_dominant_frequencies(signal, fs, n=5):
    """
    Trouve les n premières fréquences dominantes dans un signal.
    
    Parameters:
        signal (np.array): Signal audio.
        fs (int): Fréquence d'échantillonnage (Hz).
        n (int): Nombre de fréquences dominantes à extraire.
        
    Returns:
        list: Les n fréquences dominantes (en Hz).
    """
    # Calcul de la FFT
    fft_values = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)

    # Ne garder que la moitié du spectre (fréquences positives)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft_values = fft_values[:len(fft_values)//2]

    # Trouver les indices des n plus grandes amplitudes
    dominant_indices = np.argsort(positive_fft_values)[-n:][::-1]
    dominant_frequencies = positive_freqs[dominant_indices]

    return dominant_frequencies

# Fonction principale pour tester différentes fréquences d'échantillonnage
def test_sampling_rates(file_path, sampling_rates, n=10):
    """
    Teste différentes fréquences d'échantillonnage et identifie les fréquences dominantes.

    Parameters:
        file_path (str): Chemin du fichier WAV.
        sampling_rates (list): Liste de fréquences d'échantillonnage à tester.
        n (int): Nombre de fréquences dominantes à extraire.
    """
    # Charger le signal audio
    fs, signal = wavfile.read(file_path)
    if len(signal.shape) > 1:  # Convertir en mono si stéréo
        signal = np.mean(signal, axis=1)
    signal = normalize_signal(signal)  # Normalisation

    # Afficher les résultats pour chaque fréquence d'échantillonnage
    for new_fs in sampling_rates:
        if new_fs != fs:
            # Resampling pour adapter à la nouvelle fréquence
            num_samples = int(len(signal) * (new_fs / fs))
            resampled_signal = resample(signal, num_samples)
        else:
            resampled_signal = signal

        # Trouver les fréquences dominantes
        dominant_frequencies = get_dominant_frequencies(resampled_signal, new_fs, n=n)

        # Affichage des résultats
        print(f"Fréquence d'échantillonnage: {new_fs} Hz")
        print(f"Fréquences dominantes: {dominant_frequencies} Hz\n")

        # Optionnel : visualiser le spectre
        plt.figure(figsize=(8, 4))
        freqs = np.fft.fftfreq(len(resampled_signal), 1/new_fs)
        fft_values = np.abs(fft(resampled_signal))
        plt.plot(freqs[:len(freqs)//2], fft_values[:len(freqs)//2])
        plt.title(f"Spectre pour {new_fs} Hz")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

# Exemple d'utilisation
file_path = "Ressources/Sons-de-Test/resampled_audio.wav"  # Remplacez par le chemin de votre fichier
sampling_rates = [8000, 16000, 32000, 44100, 48000]  # Différentes fréquences d'échantillonnage à tester
test_sampling_rates(file_path, sampling_rates, n=10)
