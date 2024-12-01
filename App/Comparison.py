import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate, find_peaks, spectrogram, butter, filtfilt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from Processing import plot_signal_and_spectrogram, process
from Analysis import extraire_son_hyper_hypo

# -----------------------------------------------------
# Assignation des chemins de fichiers des alarmes
# -----------------------------------------------------
alarme_hypo = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
alarme_hyper = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"

# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-bruit-Strident-derriere.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-Pitch-vers-le-Haut-100cents.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Hyper-discussion_1.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Hyper-chien.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Chien-qui-aboie.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Discussion-en-fond.wav"

# On charge les fichiers audio
def load_audio(filename):
    """
    Charge un fichier audio et normalise le signal audio.

    Cette fonction utilise la bibliothèque `scipy.io.wavfile` pour lire un fichier audio au format WAV.
    Elle retourne la fréquence d'échantillonnage et le signal audio normalisé.

    Args:
        filename (str): Chemin du fichier audio à charger.

    Returns:
        tuple: 
            - rate (int): Fréquence d'échantillonnage du fichier audio en Hertz (Hz).
            - data (numpy.ndarray): Signal audio normalisé entre -1 et 1. Si le fichier contient plusieurs canaux
              (stéréo), seul le premier canal est conservé.

    Exemple:
        >>> rate, data = load_audio("audio.wav")
        >>> print(f"Fréquence d'échantillonnage : {rate} Hz")
        >>> print(f"Durée de l'audio : {len(data) / rate:.2f} secondes")

    Remarque:
        Cette fonction ne supporte que les fichiers WAV lisibles par `scipy.io.wavfile.read`.
        Assurez-vous que le fichier est compatible avec cette bibliothèque.
    """
    # Charge le fichier audio et normalise le signal
    rate, data = read(filename)

    # Pour les fichiers stéréo, garder un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Normalisation du signal
    data = data / np.max(np.abs(data))
    return rate, data

# Fonction qui applique un filtre sur la bande de fréquences intéressante
def butter_bandpass_filter(fs, data, lowcut, highcut, order=2):
    """
    Applique un filtre passe-bande de Butterworth à un signal audio.

    Cette fonction utilise un filtre passe-bande de type Butterworth pour filtrer un signal audio 
    entre deux fréquences de coupure spécifiées. Le filtrage est effectué de manière bidirectionnelle 
    à l'aide de la fonction `filtfilt`, afin d'éviter tout déphasage du signal filtré.

    Args:
        fs (int): Fréquence d'échantillonnage du signal audio en Hertz (Hz).
        data (numpy.ndarray): Signal audio à filtrer.
        lowcut (float): Fréquence de coupure basse du filtre en Hertz.
        highcut (float): Fréquence de coupure haute du filtre en Hertz.
        order (int, optionnel): Ordre du filtre de Butterworth. Par défaut, `order=2`.

    Returns:
        tuple:
            - fs (int): Fréquence d'échantillonnage du signal audio (identique à l'entrée).
            - y (numpy.ndarray): Signal audio filtré par le filtre passe-bande.

    Exemple:
        >>> fs = 44100  # Fréquence d'échantillonnage en Hz
        >>> data = np.random.randn(44100)  # Signal audio simulé d'une seconde
        >>> lowcut = 300.0  # Fréquence de coupure basse en Hz
        >>> highcut = 3000.0  # Fréquence de coupure haute en Hz
        >>> fs, filtered_data = butter_bandpass_filter(fs, data, lowcut, highcut)
        >>> print(f"Signal filtré entre {lowcut} Hz et {highcut} Hz")

    Remarque:
        Le filtre passe-bande est conçu pour des signaux dont la fréquence d'échantillonnage est 
        au moins deux fois supérieure à la fréquence de coupure haute (selon le théorème de Nyquist).
        Assurez-vous que `highcut` < `fs / 2`.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return fs, y

# Fonction pour calculer le spectre d'un signal et ajuster la longueur
def compute_fft(signal, rate, n_points=None):
    """
    Calcule la transformée de Fourier rapide (FFT) d'un signal audio et retourne les fréquences 
    ainsi que l'amplitude du spectre correspondant.

    Cette fonction ajuste la longueur du signal d'entrée à un nombre spécifié de points (par padding 
    ou troncature) avant de calculer la transformée de Fourier. Elle retourne uniquement la moitié 
    positive du spectre de fréquence, qui contient les composantes utiles pour les signaux réels.

    Args:
        signal (numpy.ndarray): Signal audio à transformer.
        rate (int): Fréquence d'échantillonnage du signal en Hertz (Hz).
        n_points (int, optionnel): Nombre de points pour la FFT. Si spécifié, le signal sera ajusté 
                                   à cette longueur par troncature ou par ajout de zéros. Par défaut, 
                                   la FFT utilise la longueur du signal d'entrée.

    Returns:
        tuple:
            - freqs (numpy.ndarray): Tableau des fréquences (en Hertz) correspondant aux composantes du spectre.
            - spectrum (numpy.ndarray): Amplitude du spectre de fréquence (valeurs positives uniquement).

    Exemple:
        >>> import numpy as np
        >>> from scipy.fft import fft
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # Signal de 1 seconde à 440 Hz
        >>> rate = 44100  # Fréquence d'échantillonnage en Hz
        >>> freqs, spectrum = compute_fft(signal, rate)
        >>> print(f"Fréquence dominante : {freqs[np.argmax(spectrum)]:.2f} Hz")

    Remarque:
        Cette fonction retourne uniquement la moitié positive du spectre, car les fréquences négatives 
        ne sont pas pertinentes pour les signaux réels.
    """
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

# Fonction pour corriger le décalage spectral (pitch shift)
def align_spectra(ref_spectrum, test_spectrum, freqs_ref, freqs_test):
    """
    Aligne un spectre de test sur un spectre de référence en ajustant le décalage fréquentiel 
    basé sur les pics principaux des deux spectres.

    Cette fonction identifie les pics dominants des deux spectres (référence et test), 
    calcule le décalage fréquentiel entre eux, et ajuste le spectre de test pour l'aligner 
    sur le spectre de référence.

    Args:
        ref_spectrum (numpy.ndarray): Amplitude du spectre de référence.
        test_spectrum (numpy.ndarray): Amplitude du spectre de test à aligner.
        freqs_ref (numpy.ndarray): Fréquences associées au spectre de référence (en Hertz).
        freqs_test (numpy.ndarray): Fréquences associées au spectre de test (en Hertz).

    Returns:
        numpy.ndarray: Spectre de test ajusté et aligné sur le spectre de référence.

    Exemple:
        >>> import numpy as np
        >>> ref_spectrum = np.array([0, 1, 5, 10, 5, 1, 0])  # Spectre de référence avec un pic à la 4e position
        >>> test_spectrum = np.array([0, 0.5, 2, 8, 4, 0.8, 0])  # Spectre de test avec un pic décalé
        >>> freqs_ref = np.linspace(0, 1000, len(ref_spectrum))  # Fréquences associées au spectre de référence
        >>> freqs_test = np.linspace(0, 1000, len(test_spectrum))  # Fréquences associées au spectre de test
        >>> aligned_spectrum = align_spectra(ref_spectrum, test_spectrum, freqs_ref, freqs_test)
        >>> print("Spectre aligné :", aligned_spectrum)

    Remarque:
        - La fonction utilise l'interpolation linéaire pour ajuster le spectre de test, ce qui peut 
          introduire des approximations pour des spectres avec des pics complexes.
        - Le spectre ajusté est défini à 0 pour les fréquences en dehors de la plage du spectre de test.
    """
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

# Fonction pour calculer les similarités entre les spectres de fréquences
def calculate_similarity(reference_1, reference_2, test_signal):
    """
    Calcule le pourcentage de similarité entre un signal de test et deux signaux de référence 
    à l'aide de plusieurs mesures de distance et de corrélation.

    Cette fonction normalise les signaux et utilise les métriques suivantes pour évaluer la similarité :
    - Distance euclidienne (valeurs proches de 1 indiquent une forte similarité).
    - Similarité cosinus (valeurs proches de 1 indiquent une forte similarité).
    - Corrélation croisée (mesure la correspondance des formes des signaux).

    Les scores finaux sont une pondération de ces trois métriques, exprimés en pourcentage.

    Args:
        reference_1 (numpy.ndarray): Premier signal de référence.
        reference_2 (numpy.ndarray): Deuxième signal de référence.
        test_signal (numpy.ndarray): Signal de test à comparer aux références.

    Returns:
        tuple:
            - score_1 (float): Pourcentage de similarité entre le signal de test et `reference_1`.
            - score_2 (float): Pourcentage de similarité entre le signal de test et `reference_2`.

    Exemple:
        >>> import numpy as np
        >>> from scipy.spatial.distance import euclidean, cosine
        >>> from scipy.signal import correlate
        >>> ref1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> ref2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        >>> test = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        >>> score_1, score_2 = calculate_similarity(ref1, ref2, test)
        >>> print(f"Similarité avec reference_1 : {score_1:.2f}%")
        >>> print(f"Similarité avec reference_2 : {score_2:.2f}%")

    Remarque:
        - Les scores sont compris entre 0% (aucune similarité) et 100% (identiques ou très similaires).
        - La pondération des scores peut être ajustée en fonction des besoins spécifiques de l'application.
        - Les signaux d'entrée doivent être de taille similaire pour garantir une comparaison cohérente.
    """
    # Normalisation des spectres pour la comparaison
    ref1_norm = reference_1 / np.linalg.norm(reference_1)
    ref2_norm = reference_2 / np.linalg.norm(reference_2)
    test_norm = test_signal / np.linalg.norm(test_signal)

    # Calcul de la distance euclidienne (0: différents, 1: identique)
    similarity_euclidean_1 = 1 / (1 + euclidean(ref1_norm, test_norm))
    similarity_euclidean_2 = 1 / (1 + euclidean(ref2_norm, test_norm))

    # Calcul de la similarité/distance cosinus (plus elle est proche de 0, plus les sons sont similaires)
    similarity_cosine_1 = 1 - cosine(ref1_norm, test_norm)
    similarity_cosine_2 = 1 - cosine(ref2_norm, test_norm)

    # Calcul de la corrélation croisée (plus les maximum sont élevés, plus ils sont similaires)
    correlation_1 = np.max(correlate(ref1_norm, test_norm)) / len(ref1_norm)
    correlation_2 = np.max(correlate(ref2_norm, test_norm)) / len(ref2_norm)

    # Score final (pondération ajustable)
    score_1 = 0.4 * similarity_euclidean_1 + 0.6 * similarity_cosine_1 + 0.4 * correlation_1
    score_2 = 0.4 * similarity_euclidean_2 + 0.6 * similarity_cosine_2 + 0.4 * correlation_2

    return score_1 * 100, score_2 * 100  # Pourcentage de ressemblance

# Fonction pour calculer un spectrogramme
def compute_spectrogram(signal, rate, nperseg=256):
    """
    Calcule le spectrogramme d'un signal audio.

    Cette fonction utilise la transformation de Fourier à court terme (STFT) pour générer un spectrogramme, 
    qui représente l'évolution de la puissance spectrale du signal en fonction du temps et de la fréquence.
    Le spectrogramme est renvoyé sur une échelle logarithmique en décibels (dB).

    Args:
        signal (numpy.ndarray): Signal audio à analyser.
        rate (int): Fréquence d'échantillonnage du signal en Hertz (Hz).
        nperseg (int, optionnel): Nombre d'échantillons par segment pour la transformation de Fourier. 
                                  Par défaut, `nperseg=256`.

    Returns:
        tuple:
            - freqs (numpy.ndarray): Fréquences associées au spectrogramme (en Hertz).
            - times (numpy.ndarray): Temps associés au spectrogramme (en secondes).
            - Sxx (numpy.ndarray): Spectrogramme en échelle logarithmique (dB).

    Exemple:
        >>> import numpy as np
        >>> from scipy.signal import spectrogram
        >>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # Signal sinusoïdal à 440 Hz
        >>> rate = 44100  # Fréquence d'échantillonnage en Hz
        >>> freqs, times, Sxx = compute_spectrogram(signal, rate)
        >>> print(f"Fréquences: {freqs.shape}, Temps: {times.shape}, Spectrogramme: {Sxx.shape}")

    Remarque:
        - Le paramètre `nperseg` détermine la résolution fréquentielle et temporelle du spectrogramme.
          Une valeur plus élevée améliore la résolution en fréquence mais réduit la résolution temporelle.
        - Le spectrogramme est converti en décibels (dB) pour une meilleure interprétation visuelle et auditive.
        - Une petite valeur constante (`1e-10`) est ajoutée pour éviter les erreurs dues aux logarithmes de zéro.
    """
    freqs, times, Sxx = spectrogram(signal, fs=rate, nperseg=nperseg)
    Sxx = 10 * np.log10(Sxx + 1e-10)  # Échelle en dB
    return freqs, times, Sxx

# Fonction pour comparer des spectrogrammes
def compare_spectrograms(S1, S2):
    """
    Compare deux spectrogrammes en calculant la corrélation croisée maximale entre eux.

    Cette fonction normalise les deux spectrogrammes d'entrée, puis calcule la corrélation croisée 
    entre leurs versions aplaties (vecteurs unidimensionnels). La valeur maximale de cette corrélation 
    est renvoyée pour quantifier la similarité.

    Args:
        S1 (numpy.ndarray): Premier spectrogramme à comparer.
        S2 (numpy.ndarray): Deuxième spectrogramme à comparer.

    Returns:
        float: Valeur maximale de la corrélation croisée entre les deux spectrogrammes normalisés. 
               Une valeur proche de 1 indique une forte similarité, tandis qu'une valeur plus faible 
               indique une faible similarité.

    Exemple:
        >>> import numpy as np
        >>> from scipy.signal import spectrogram, correlate
        >>> S1 = np.random.rand(100, 100)  # Spectrogramme aléatoire 1
        >>> S2 = np.random.rand(100, 100)  # Spectrogramme aléatoire 2
        >>> max_corr = compare_spectrograms(S1, S2)
        >>> print(f"Corrélation maximale entre les deux spectrogrammes : {max_corr:.2f}")

    Remarque:
        - Les spectrogrammes doivent être de taille comparable pour garantir une comparaison cohérente.
        - La normalisation permet de rendre la comparaison indépendante de l'amplitude absolue des spectrogrammes.
    """
    # Normaliser les spectrogrammes
    S1_norm = S1 / np.linalg.norm(S1)
    S2_norm = S2 / np.linalg.norm(S2)
    # Corrélation croisée
    correlation = correlate(S1_norm.flatten(), S2_norm.flatten(), mode='valid')
    max_corr = np.max(correlation)
    return max_corr

def determine_alarm_type(freqs, times, Sxx, score_alarm, threshold=50, freq_min=3900, freq_max=5250):
    """
    Détermine le type d'alarme (hypoglycémie, hyperglycémie ou indéterminé) à partir d'un spectrogramme.

    Cette fonction analyse le spectrogramme d'un signal audio en fonction de sa fréquence dominante 
    dans une plage de fréquences spécifiée, et détecte une tendance générale ascendante ou descendante 
    pour classifier le type d'alarme.

    Args:
        freqs (numpy.ndarray): Tableau des fréquences associées au spectrogramme (en Hertz).
        times (numpy.ndarray): Tableau des instants de temps associés au spectrogramme (en secondes).
        Sxx (numpy.ndarray): Spectrogramme (puissance spectrale en fonction du temps et de la fréquence).
        score_alarm (float): Score de similarité entre le signal de test et un signal d'alarme de référence.
        threshold (float, optionnel): Seuil minimal du score de similarité pour considérer le signal comme une alarme. 
                                      Par défaut, `threshold=50`.
        freq_min (float, optionnel): Fréquence minimale de la plage d'analyse (en Hertz). Par défaut, `freq_min=3900`.
        freq_max (float, optionnel): Fréquence maximale de la plage d'analyse (en Hertz). Par défaut, `freq_max=5250`.

    Returns:
        str: Type d'alarme détecté :
            - `"Hyperglycémie"` si les fréquences dominantes montrent une tendance ascendante.
            - `"Hypoglycémie"` si les fréquences dominantes montrent une tendance descendante.
            - `"Indéterminé"` si aucune tendance claire ne se dégage.
            - `"Ce n'est pas une alarme"` si le score de similarité est inférieur au seuil.

    Exemple:
        >>> import numpy as np
        >>> freqs = np.linspace(3000, 6000, 100)  # Fréquences simulées
        >>> times = np.linspace(0, 10, 50)  # Temps simulés
        >>> Sxx = np.random.rand(100, 50)  # Spectrogramme simulé
        >>> score_alarm = 60  # Score de similarité supérieur au seuil
        >>> result = determine_alarm_type(freqs, times, Sxx, score_alarm)
        >>> print(f"Type d'alarme détecté : {result}")

    Remarque:
        - La plage de fréquences est définie par `freq_min` et `freq_max`, correspondant à la plage attendue pour 
          l'analyse d'alarmes spécifiques.
        - La fonction vérifie si les fréquences dominantes dans chaque tranche temporelle montrent une tendance 
          ascendante (hyperglycémie) ou descendante (hypoglycémie).
    """
    if score_alarm < threshold:
        return "Ce n'est pas une alarme"
    
    # Filtrer les fréquences dans la plage souhaitée
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_filtered = freqs[freq_mask]
    Sxx_filtered = Sxx[freq_mask, :]

    # Détection des fréquences dominantes dans chaque tranche temporelle (sur la plage filtrée)
    dominant_freqs = []
    for col in range(Sxx_filtered.shape[1]):
        dominant_idx = np.argmax(Sxx_filtered[:, col])
        dominant_freqs.append(freqs_filtered[dominant_idx])

    # Vérifier l'ordre des fréquences (montant ou descendant)
    differences = np.diff(dominant_freqs)  # Différences entre les fréquences successives
    
    # Déterminer la tendance générale
    if np.all(differences > 0):  # Toutes les différences sont positives (montée)
        return "Hyperglycémie"
    elif np.all(differences < 0):  # Toutes les différences sont négatives (descente)
        return "Hypoglycémie"
    else:
        # Analyse des tendances générales (majorité montante ou descendante)
        upward_trend = np.sum(differences > 0)
        downward_trend = np.sum(differences < 0)
        
        if upward_trend > downward_trend:
            return "Hyperglycémie"
        elif downward_trend > upward_trend:
            return "Hypoglycémie"
        else:
            return "Indéterminé"

# -----------------------------------------------------
# Fonction principale du traitement de Comparaison
# -----------------------------------------------------
def runComparison(rate_test, test_alarm):
    """
    Compare un signal d'alarme de test avec des alarmes de référence pour déterminer son type 
    (hypoglycémie, hyperglycémie ou non).

    Cette fonction effectue plusieurs étapes, notamment :
    - Chargement et filtrage des fichiers audio de référence (alarme hypo et hyper).
    - Calcul et affichage des signaux et spectrogrammes avant et après filtrage.
    - Calcul des spectres de fréquence à l'aide de la transformée de Fourier (FFT).
    - Alignement en fréquence (correction de pitch) du spectre de test.
    - Calcul des scores de similarité entre le spectre de test et les spectres de référence.
    - Détermination finale du type d'alarme en fonction des fréquences dominantes et des scores.

    Args:
        rate_test (int): Taux d'échantillonnage du signal d'alarme de test.
        test_alarm (numpy.ndarray): Signal audio de test à analyser.

    Returns:
        tuple:
            - score_hypo (float): Pourcentage de similarité spectrale avec l'alarme d'hypoglycémie.
            - score_hyper (float): Pourcentage de similarité spectrale avec l'alarme d'hyperglycémie.
            - score_spectro_hypo (float): Score de similarité du spectrogramme avec l'alarme d'hypoglycémie.
            - score_spectro_hyper (float): Score de similarité du spectrogramme avec l'alarme d'hyperglycémie.
            - alarm_message (str): Message indiquant si le signal est une alarme ou non.
            - alarm_type (str): Type d'alarme détecté ("Hypoglycémie", "Hyperglycémie", ou "Ce n'est pas une alarme").

    Exemple:
        >>> rate_test = 44100  # Taux d'échantillonnage du signal de test
        >>> test_alarm = np.random.rand(44100)  # Signal audio de test simulé
        >>> score_hypo, score_hyper, score_spectro_hypo, score_spectro_hyper, alarm_message, alarm_type = runComparison(rate_test, test_alarm)
        >>> print(f"Score Hypo : {score_hypo:.2f}%")
        >>> print(f"Score Hyper : {score_hyper:.2f}%")
        >>> print(f"Type d'alarme : {alarm_type}")

    Remarque:
        - La fonction vérifie que les taux d'échantillonnage des fichiers audio de référence et du signal de test sont identiques.
        - Les scores de similarité sont calculés à la fois pour les spectres de fréquence (FFT) et les spectrogrammes.
        - Le type d'alarme est déterminé en analysant la tendance des fréquences dominantes dans le spectrogramme.
        - Le seuil de similarité pour valider une alarme est défini à 50 par défaut.

    Erreurs:
        ValueError: Si les taux d'échantillonnage des signaux de référence et de test sont différents.
    """
    # Chargement des fichiers de sons d'alarme Hypo et Hyper
    #r_hypo, a_hypo = load_audio(alarme_hypo)
    #r_hyper, a_hyper = load_audio(alarme_hyper)
    r_filtre_hypo, a_filtre_hypo = process(alarme_hypo)
    r_filtre_hyper, a_filtre_hyper = process(alarme_hyper)

    rate_f_a_hypo, alarm_f_a_hypo = extraire_son_hyper_hypo(r_filtre_hypo, a_filtre_hypo)
    rate_f_a_hyper, alarm_f_a_hyper = extraire_son_hyper_hypo(r_filtre_hyper, a_filtre_hyper)
    #rate_hypo, alarm_hypo = extraire_son_hyper_hypo(r_hypo, a_hypo)
    #rate_hyper, alarm_hyper = extraire_son_hyper_hypo(r_hyper, a_hyper)

    # fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    # plot_signal_and_spectrogram(r_hypo, a_hypo, "Original Hypo", axes[0, 0], axes[1, 0], axes[2, 0])
    # plot_signal_and_spectrogram(rate_hypo, alarm_hypo, "filtre Hypo", axes[0, 1], axes[1, 1], axes[2, 1])
    # plt.show()

    # plot_signal_and_spectrogram(r_hyper, a_hyper, "Original Hyper", axes[0, 0], axes[1, 0], axes[2, 0])
    # plot_signal_and_spectrogram(rate_hyper, alarm_hyper, "filtre Hyper", axes[0, 1], axes[1, 1], axes[2, 1])
    # plt.show()

    
    # Vérification des taux d'échantillonnage
    if rate_f_a_hypo != rate_f_a_hyper or rate_f_a_hypo != rate_test:
        raise ValueError("Les fichiers audio doivent avoir le même taux d'échantillonnage.")

    # Nombre de points pour la FFT
    n_points = min(len(alarm_f_a_hypo), len(alarm_f_a_hyper), len(test_alarm))

    # Calcul des FFT
    freqs_hypo, spectrum_hypo = compute_fft(alarm_f_a_hypo, rate_f_a_hypo, n_points=n_points)
    freqs_hyper, spectrum_hyper = compute_fft(alarm_f_a_hyper, rate_f_a_hyper, n_points=n_points)
    freqs_test, spectrum_test = compute_fft(test_alarm, rate_test, n_points=n_points)

    # Correction de pitch sur le spectre de test
    spectrum_test = align_spectra(spectrum_hypo, spectrum_test, freqs_hypo, freqs_test)

    # Calcul des scores de similarité
    score_hypo, score_hyper = calculate_similarity(spectrum_hypo, spectrum_hyper, spectrum_test)
    score_alarm = max(score_hypo, score_hyper)  # Score global pour validation

    # Calcul des spectrogrammes avec ajustement dynamique
    nperseg = min(len(test_alarm) // 10, 8192)  # Taille dynamique de la fenêtre
    freqs_hypo_s, times_hypo, Sxx_hypo = compute_spectrogram(alarm_f_a_hypo, rate_f_a_hypo, nperseg=nperseg)
    freqs_hyper_s, times_hyper, Sxx_hyper = compute_spectrogram(alarm_f_a_hyper, rate_f_a_hyper, nperseg=nperseg)
    freqs_test_s, times_test, Sxx_test = compute_spectrogram(test_alarm, rate_test, nperseg=nperseg)

    # Calcul des similarités des spectrogrammes
    score_spectro_hypo = compare_spectrograms(Sxx_hypo, Sxx_test)
    score_spectro_hyper = compare_spectrograms(Sxx_hyper, Sxx_test)

    # Détermination du type d'alarme
    alarm_type = determine_alarm_type(freqs_test_s, times_test, Sxx_test, score_alarm, threshold=50)

    # Affichage des résultats
    print(f"Score spectre Hypoglycémie : {score_hypo:.2f}%")
    print(f"Score spectre Hyperglycémie : {score_hyper:.2f}%")
    print(f"Score spectrogramme Hypoglycémie : {score_spectro_hypo:.2f}")
    print(f"Score spectrogramme Hyperglycémie : {score_spectro_hyper:.2f}")
    alarm_message = "C'est une alarme" if score_alarm > 50 else "Ce n'est pas une alarme"
    print(f"Alarme ou non ? : {alarm_message}")
    print(f"Type d'alarme : {alarm_type}")

    return score_hypo, score_hyper, score_spectro_hypo, score_spectro_hyper, alarm_message, alarm_type