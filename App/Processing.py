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
from scipy.signal import butter, filtfilt, resample_poly
import soundfile as sf

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
    highcuts = [3950.0, 4250.0, 4520.0, 4880.0, 5195.0]

    # Appliquer les filtres
    filtered_signal = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    return fs, filtered_signal
