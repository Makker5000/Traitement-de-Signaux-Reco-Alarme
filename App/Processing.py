"""
Module de traitement et de filtrage de signaux audio.

Ce module permet de :
1. Ré-échantillonner un fichier audio à une fréquence cible.
2. Appliquer un ou plusieurs filtres passe-bande à un signal.
3. Traiter un fichier audio en combinant ces étapes et retourner le signal filtré.

Dépendances :
- numpy
- scipy.signal (butter, filtfilt, resample_poly)
- soundfile (sf)

Fonctions :
- resample_audio : Ré-échantillonne un signal audio à une fréquence cible.
- apply_bandpass_filter : Applique un filtre passe-bande à un signal.
- combine_bandpass_filters : Applique et combine plusieurs filtres passe-bande.
- process_and_plot : Ré-échantillonne, filtre et retourne le signal filtré.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, resample_poly, spectrogram
import soundfile as sf
import matplotlib.pyplot as plt


"""
TESSSSSSSSSSSSSSSSSSSSSST
"""

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

def compute_fft(signal, rate, n_points=None):
    # Ajustement de la longueur du signal (padding ou troncature)
    if n_points is not None:
        if len(signal) > n_points:
            signal = signal[:n_points]
        elif len(signal) < n_points:
            signal = np.pad(signal, (0, n_points - len(signal)), mode='constant')
    
    # Calcul de la transformée de Fourier
    spectrum = np.abs(fft(signal))

    # Fréquences associées
    freqs = np.fft.fftfreq(len(spectrum), 1/rate)

    # Prend uniquement la moitié positive du spectre
    # "//2" veut dire que je divise par 2 via une division entière (donc les virgules décimales sont supprimées).
    #  L'élément à "//2" donc à la moitié n'est PAS inclus.
    return freqs[:len(freqs)//2], spectrum[:len(spectrum)//2]

def align_spectra(ref_spectrum, test_spectrum, freqs_ref, freqs_test):
    # Identifiez le pic principal dans chaque spectre
    ref_peak_idx = np.argmax(ref_spectrum)
    test_peak_idx = np.argmax(test_spectrum)
    
    # Trouvez le ratio de décalage de fréquence
    shift_ratio = freqs_test[test_peak_idx] / freqs_ref[ref_peak_idx]

    # Appliquez la correction sur le spectre de test
    adjusted_test_spectrum = np.interp(
        freqs_ref,  # Fréquences de référence
        freqs_test / shift_ratio,  # Décalage des fréquences
        test_spectrum,
        left=0,
        right=0,
    )
    return adjusted_test_spectrum
"""
FIN DE TEST
"""

def resample_audio(file_path, target_fs=44100):
    """
    Ré-échantillonne un fichier audio à une fréquence cible.

    Parameters:
        file_path (str): Chemin vers le fichier audio.
        target_fs (int, optional): Nouvelle fréquence d'échantillonnage en Hz. Par défaut : 44100 Hz.

    Returns:
        tuple:
            - int: Fréquence d'échantillonnage cible (target_fs).
            - numpy.ndarray: Signal ré-échantillonné.

    Raises:
        FileNotFoundError: Si le fichier audio spécifié n'existe pas.
        ValueError: Si les paramètres de ré-échantillonnage sont invalides.

    Exemple:
        >>> fs, signal = resample_audio("audio.wav", target_fs=16000)
        >>> print(f"Signal ré-échantillonné à {fs} Hz")
    """
    signal, original_fs = sf.read(file_path)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    gcd = np.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd

    resampled_signal = resample_poly(signal, up, down)

    return target_fs, resampled_signal

def apply_bandpass_filter(signal, fs, lowcut, highcut, order=3):
    """
    Applique un filtre passe-bande à un signal.

    Parameters:
        signal (numpy.ndarray): Signal d'entrée.
        fs (float): Fréquence d'échantillonnage du signal (en Hz).
        lowcut (float): Fréquence de coupure basse (en Hz).
        highcut (float): Fréquence de coupure haute (en Hz).
        order (int, optional): Ordre du filtre de Butterworth. Par défaut : 3.

    Returns:
        numpy.ndarray: Signal filtré.

    Raises:
        ValueError: Si `lowcut` ou `highcut` sont invalides.

    Exemple:
        >>> filtered_signal = apply_bandpass_filter(signal, 44100, 3900, 4200)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def combine_bandpass_filters(signal, fs, lowcuts, highcuts):
    """
    Applique et combine plusieurs filtres passe-bande à un signal.

    Parameters:
        signal (numpy.ndarray): Signal d'entrée.
        fs (float): Fréquence d'échantillonnage du signal (en Hz).
        lowcuts (list of float): Liste des fréquences de coupure basse (en Hz).
        highcuts (list of float): Liste des fréquences de coupure haute (en Hz).

    Returns:
        numpy.ndarray: Signal combiné après application des filtres.

    Raises:
        ValueError: Si les listes `lowcuts` et `highcuts` n'ont pas la même longueur.

    Exemple:
        >>> lowcuts = [3900, 4200]
        >>> highcuts = [3950, 4250]
        >>> combined_signal = combine_bandpass_filters(signal, 44100, lowcuts, highcuts)
    """
    if len(lowcuts) != len(highcuts):
        raise ValueError("Les listes lowcuts et highcuts doivent avoir la même longueur.")

    filtered_signals = []
    for lowcut, highcut in zip(lowcuts, highcuts):
        filtered_signal = apply_bandpass_filter(signal, fs, lowcut, highcut)
        filtered_signals.append(filtered_signal)
    return np.sum(filtered_signals, axis=0)

def process(file_path):
    """
    Ré-échantillonne, applique plusieurs filtres passe-bande et retourne le signal filtré.

    Parameters:
        file_path (str): Chemin vers le fichier audio à traiter.

    Returns:
        tuple:
            - int: Fréquence d'échantillonnage utilisée (44100 Hz).
            - numpy.ndarray: Signal filtré.

    Exemple:
        >>> fs, filtered_signal = process("audio.wav")
        >>> print(f"Signal filtré à {fs} Hz")
    """
    # Charger le signal et le re-échantilloné
    fs, signal = resample_audio(file_path, target_fs=44100)
   


    # Définir les bandes de fréquences
    lowcuts = [3900.0, 4200.0, 4470.0, 4830.0, 5145.0]
    highcuts = [3950.0, 4250.0, 4520.0, 4880.0, 5300.0]

    # Appliquer les filtres
    filtered_signal = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plot_signal_and_spectrogram(fs, signal, "Original Signal", axes[0, 0], axes[1, 0], axes[2, 0])
    plot_signal_and_spectrogram(fs, filtered_signal, "Filtered Signal", axes[0, 1], axes[1, 1], axes[2, 1])
    
    plt.show()
    return fs, filtered_signal
