import numpy as np
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
import sounddevice as sd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------
# PLAGES DE FRÉQUENCES
#-------------------------------------------------------------------------------------------------

# plage de fréquences fondamentales du son hyper/hypo : 1200-5600 Hz

#-------------------------------------------------------------------------------------------------
# FONCTION DE CHARGEMENT DES FICHIERS AUDIOS
#-------------------------------------------------------------------------------------------------

def load_audio(filename):
    # Charge le fichier audio et normalise le signal
    rate, data = read(filename)

    # Pour les fichiers stéréo, garder un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Normalisation du signal
    data = data / np.max(np.abs(data))
    return rate, data

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

def afficher_signal(t, signal, titre="Signal"):
    """
    Affiche un signal en fonction du temps.
    
    Parameters:
        t (array): Les temps du signal.
        signal (array): Le signal à afficher.
        titre (str): Le titre du graphique.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal)
    plt.title(titre)
    plt.xlabel("Temps [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

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

    # Retourner la fréquence d'échantillonnage et le signal réduit
    return fe, signal_reduit

# Appel de la fonction avec le signal filtré et sa fréquence d'échantillonnage
# frequence_echantillonnage, signal_reduit = extraire_son_hyper_hypo()

# Affichage des résultats
# print(f"Fréquence d'échantillonnage : {frequence_echantillonnage} Hz")
# if signal_reduit is not None:
#     print(f"Signal réduit disponible avec {len(signal_reduit)} échantillons.")
# else:
#     print("Aucun signal réduit n'a été détecté.")

# Écoute du son réduit
# if signal_reduit is not None:
#     print("Lecture du signal réduit...")
#     sd.play(signal_reduit, frequence_echantillonnage)
#     sd.wait()
#     print("Lecture terminée.")
# else:
#     print("Aucun signal réduit disponible pour la lecture.")