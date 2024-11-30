import numpy as np
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
import sounddevice as sd
import matplotlib.pyplot as plt
import Processing
from Processing import plot_signal_and_spectrogram

#-------------------------------------------------------------------------------------------------
# EXTRACTION DU SON HYPER/HYPO
#-------------------------------------------------------------------------------------------------

def calculer_spectrogramme(signal, fe, taille_fenetre=1024, overlap=512):
    """
    Calcule le spectrogramme d'un signal.
    
    Parameters:
        signal (array): Le signal audio.
        fe (int): La fréquence d'échantillonnage.
        taille_fenetre (int): La taille de chaque segment pour le spectrogramme.
        overlap (int): Le recouvrement entre les fenêtres.

    Returns:
        frequencies (array): Les fréquences du spectrogramme.
        times (array): Les temps du spectrogramme.
        spectro (array): Les valeurs du spectrogramme.
    """
    frequencies, times, spectro = spectrogram(signal, fs=fe, nperseg=taille_fenetre, noverlap=overlap)
    return frequencies, times, spectro

def extraire_regions_specifiques(signal, fe, frequencies, times, spectro, plages_frequences, seuil_facteur=1.5):
    """
    Extrait les régions du signal correspondant à des plages de fréquences spécifiques.
    
    Parameters:
        signal (array): Le signal filtré.
        fe (int): La fréquence d'échantillonnage.
        frequencies (array): Les fréquences du spectrogramme.
        times (array): Les temps du spectrogramme.
        spectro (array): Les valeurs du spectrogramme.
        plages_frequences (list): Liste de tuples (f_min, f_max) des plages de fréquences spécifiques.
        seuil_facteur (float): Facteur pour déterminer le seuil d'énergie significative.

    Returns:
        signal_reduit (array): Le signal extrait correspondant aux plages spécifiques.
        t_reduit (array): Les temps associés au signal extrait.
    """
    masque_temps = np.zeros(len(times), dtype=bool)

    for f_min, f_max in plages_frequences:
        indices_freq = np.where((frequencies >= f_min) & (frequencies <= f_max))[0]
        energie_frequence = np.sum(spectro[indices_freq, :], axis=0)
        seuil = np.mean(energie_frequence) + seuil_facteur * np.std(energie_frequence)
        masque_temps |= (energie_frequence > seuil)

    indices_valides = np.where(masque_temps)[0]
    if len(indices_valides) > 0:
        debut = int(times[indices_valides[0]] * fe)
        fin = int(times[indices_valides[-1]] * fe)
        signal_reduit = signal[debut:fin]
        t_reduit = np.arange(debut, fin) / fe
        return signal_reduit, t_reduit
    else:
        print("Aucune région contenant les plages de fréquences spécifiées n'a été détectée.")
        return None, None

def extraire_son_hyper_hypo(fe, signal_filtre):
    """
    Parameters:
        signal_filtre (array): Le signal audio filtré.
        fe (int): La fréquence d'échantillonnage.

    Returns:
        fe (int): La fréquence d'échantillonnage du signal réduit.
        signal_reduit (array): Le signal extrait correspondant aux plages spécifiques.
    """
    plages_frequences = [
        (3900, 4000), (4150, 4300), (4400, 4600), (4750, 5000), (5100, 5250),  # Hyperglycémique initiales
        (1280, 1340), (1380, 1440), (1470, 1540), (1580, 1660), (1690, 1760)   # Nouvelles plages ajoutées
    ]

    #Plages de fréquence dominantes au cas où les harmoniques en plus posent problème
    # plages_frequences = [
    # (3900, 4000),
    # (4150, 4300),
    # (4400, 4600),
    # (4750, 5000),
    # (5100, 5250)
    # ]

    taille_fenetre = 1024
    overlap = 512

    # Calcul du spectrogramme
    frequencies, times, spectro = calculer_spectrogramme(signal_filtre, fe, taille_fenetre, overlap)

    # Extraction des régions spécifiques
    signal_reduit, _ = extraire_regions_specifiques(
        signal_filtre, fe, frequencies, times, spectro, plages_frequences
    )

    # Affichage des résultats
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plot_signal_and_spectrogram(fe, signal_filtre, "Filtered Signal", axes[0, 0], axes[1, 0], axes[2, 0])
    plot_signal_and_spectrogram(fe, signal_reduit, "Signal Réduit", axes[0, 1], axes[1, 1], axes[2, 1])
    plt.show()

    # Retourner la fréquence d'échantillonnage et le signal réduit
    return fe, signal_reduit