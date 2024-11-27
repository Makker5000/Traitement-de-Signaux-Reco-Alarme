import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

def resample_audio(file_path, target_fs=44100):
    """
    Ré-échantillonne un signal audio à une nouvelle fréquence d'échantillonnage sans altérer le spectre fréquentiel.

    Parameters:
        file_path (str): Chemin vers le fichier audio.
        target_fs (int): Nouvelle fréquence d'échantillonnage.

    Returns:
        np.ndarray: Signal ré-échantillonné.
        int: Nouvelle fréquence d'échantillonnage.
    """
    # Charger le fichier audio
    signal, original_fs = sf.read(file_path)

    # Convertir en mono si le signal est stéréo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Calcul des facteurs de ré-échantillonnage
    gcd = np.gcd(original_fs, target_fs)
    up = target_fs // gcd
    down = original_fs // gcd

    # Utilisation de scipy.signal.resample_poly pour le ré-échantillonnage
    resampled_signal = resample_poly(signal, up, down)

    return resampled_signal, target_fs

# Exemple d'utilisation


file_path = "Ressources\Sons-de-Test\Hyper-discussion_1.wav"  # Remplacez par votre fichier
target_fs = 44100  # Nouvelle fréquence d'échantillonnage

resampled_signal, new_fs = resample_audio(file_path, target_fs)

