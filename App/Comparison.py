import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate, find_peaks, spectrogram, butter, filtfilt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from Processing import plot_signal_and_spectrogram
from Analysis import extraire_son_hyper_hypo

# -----------------------------------------------------
# Assignation des chemins de fichiers des alarmes
# -----------------------------------------------------
alarme_hypo = "../Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
alarme_hyper = "../Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"

# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-bruit-Strident-derriere.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-Pitch-vers-le-Haut-100cents.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Hyper-discussion_1.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Hyper-chien.wav"

# On charge les fichiers audio
def load_audio(filename):
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
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return fs, y

# Fonction pour calculer le spectre d'un signal et ajuster la longueur
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

# Fonction pour corriger le décalage spectral (pitch shift)
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

# Fonction pour calculer les similarités entre les spectres de fréquences
def calculate_similarity(reference_1, reference_2, test_signal):
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
    score_1 = 0.4 * similarity_euclidean_1 + 0.4 * similarity_cosine_1 + 0.2 * correlation_1
    score_2 = 0.4 * similarity_euclidean_2 + 0.4 * similarity_cosine_2 + 0.2 * correlation_2

    return score_1 * 100, score_2 * 100  # Pourcentage de ressemblance

# Fonction pour calculer un spectrogramme
def compute_spectrogram(signal, rate, nperseg=256):
    freqs, times, Sxx = spectrogram(signal, fs=rate, nperseg=nperseg)
    Sxx = 10 * np.log10(Sxx + 1e-10)  # Échelle en dB
    return freqs, times, Sxx

# Fonction pour comparer des spectrogrammes
def compare_spectrograms(S1, S2):
    # Normaliser les spectrogrammes
    S1_norm = S1 / np.linalg.norm(S1)
    S2_norm = S2 / np.linalg.norm(S2)
    # Corrélation croisée
    correlation = correlate(S1_norm.flatten(), S2_norm.flatten(), mode='valid')
    max_corr = np.max(correlation)
    return max_corr

def determine_alarm_type(freqs, times, Sxx, score_alarm, threshold=50, freq_min=3900, freq_max=5250):
    """
    Analyse le spectrogramme pour déterminer le type d'alarme.
    
    Args:
        freqs (ndarray): Fréquences du spectrogramme.
        times (ndarray): Temps associés aux colonnes du spectrogramme.
        Sxx (ndarray): Intensité spectrale (amplitudes) du spectrogramme.
        score_alarm (float): Score du premier test (validation de l'alarme).
        threshold (float): Seuil pour confirmer que c'est une alarme.
        
    Returns:
        str: Type d'alarme détectée ("Hypoglycémie", "Hyperglycémie", ou "Indéterminé").
    """
    if score_alarm < threshold:
        return "Ce n'est pas une alarme"
    
    # Filtrer les fréquences dans la plage souhaitée
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    freqs_filtered = freqs[freq_mask]
    Sxx_filtered = Sxx[freq_mask, :]

    # Détection des fréquences dominantes dans chaque tranche temporelle
    # dominant_freqs = []
    # for col in range(Sxx.shape[1]):
    #     # Identifier la fréquence avec l'amplitude maximale pour chaque tranche temporelle
    #     dominant_idx = np.argmax(Sxx[:, col])
    #     dominant_freqs.append(freqs[dominant_idx])

    # Détection des fréquences dominantes dans chaque tranche temporelle (sur la plage filtrée)
    dominant_freqs = []
    for col in range(Sxx_filtered.shape[1]):
        dominant_idx = np.argmax(Sxx_filtered[:, col])
        dominant_freqs.append(freqs_filtered[dominant_idx])

    # Vérifier l'ordre des fréquences (montant ou descendant)
    differences = np.diff(dominant_freqs)  # Différences entre les fréquences successives

    # tolerance = 1e-3 # Valeur de tolérance de variation entre les valeurs
    
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
    # Chargement des fichiers de sons d'alarme Hypo et Hyper
    r_hypo, a_hypo = load_audio(alarme_hypo)
    r_hyper, a_hyper = load_audio(alarme_hyper)
    rate_hypo, alarm_hypo = extraire_son_hyper_hypo(r_hypo, a_hypo)
    rate_hyper, alarm_hyper = extraire_son_hyper_hypo(r_hyper, a_hyper)
    # rate_hypo, alarm_hypo = butter_bandpass_filter(r_hypo, a_hypo, 3900, 5250)
    # rate_hyper, alarm_hyper = butter_bandpass_filter(r_hyper, a_hyper, 3900, 5250)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    plot_signal_and_spectrogram(r_hypo, a_hypo, "Original Hypo", axes[0, 0], axes[1, 0], axes[2, 0])
    plot_signal_and_spectrogram(rate_hypo, alarm_hypo, "filtre Hypo", axes[0, 1], axes[1, 1], axes[2, 1])
    plt.show()

    plot_signal_and_spectrogram(r_hyper, a_hyper, "Original Hyper", axes[0, 0], axes[1, 0], axes[2, 0])
    plot_signal_and_spectrogram(rate_hyper, alarm_hyper, "filtre Hyper", axes[0, 1], axes[1, 1], axes[2, 1])
    plt.show()

    
    # Vérification des taux d'échantillonnage
    if rate_hypo != rate_hyper or rate_hypo != rate_test:
        raise ValueError("Les fichiers audio doivent avoir le même taux d'échantillonnage.")

    # Nombre de points pour la FFT
    n_points = min(len(alarm_hypo), len(alarm_hyper), len(test_alarm))

    # Calcul des FFT
    freqs_hypo, spectrum_hypo = compute_fft(alarm_hypo, rate_hypo, n_points=n_points)
    freqs_hyper, spectrum_hyper = compute_fft(alarm_hyper, rate_hyper, n_points=n_points)
    freqs_test, spectrum_test = compute_fft(test_alarm, rate_test, n_points=n_points)

    # Correction de pitch sur le spectre de test
    spectrum_test = align_spectra(spectrum_hypo, spectrum_test, freqs_hypo, freqs_test)

    # Calcul des scores de similarité
    score_hypo, score_hyper = calculate_similarity(spectrum_hypo, spectrum_hyper, spectrum_test)
    score_alarm = max(score_hypo, score_hyper)  # Score global pour validation

    # Calcul des spectrogrammes avec ajustement dynamique
    nperseg = min(len(test_alarm) // 10, 8192)  # Taille dynamique de la fenêtre
    freqs_hypo_s, times_hypo, Sxx_hypo = compute_spectrogram(alarm_hypo, rate_hypo, nperseg=nperseg)
    freqs_hyper_s, times_hyper, Sxx_hyper = compute_spectrogram(alarm_hyper, rate_hyper, nperseg=nperseg)
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

    # Plot des spectres de fréquence
    # plt.figure(figsize=(12, 6))
    # plt.plot(freqs_hypo, spectrum_hypo, label="Spectre Hypoglycémie", color='blue', alpha=0.7)
    # plt.plot(freqs_hyper, spectrum_hyper, label="Spectre Hyperglycémie", color='red', alpha=0.7)
    # plt.plot(freqs_test, spectrum_test, label="Spectre Test", color='green', linestyle='--', alpha=0.7)
    # plt.xlabel("Fréquence (Hz)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.title("Comparaison des spectres de fréquence")
    # plt.grid()
    # plt.show()

    # # Plot des spectrogrammes
    # plt.figure(figsize=(18, 6))
    # plt.subplot(1, 3, 1)
    # plt.pcolormesh(times_hypo, freqs_hypo_s, Sxx_hypo, shading='gouraud', cmap='viridis')
    # plt.title("Spectrogramme Hypoglycémie")
    # plt.ylabel("Fréquence (Hz)")
    # plt.xlabel("Temps (s)")
    # plt.colorbar()

    # plt.subplot(1, 3, 2)
    # plt.pcolormesh(times_hyper, freqs_hyper_s, Sxx_hyper, shading='gouraud', cmap='viridis')
    # plt.title("Spectrogramme Hyperglycémie")
    # plt.xlabel("Temps (s)")
    # plt.colorbar()

    # plt.subplot(1, 3, 3)
    # plt.pcolormesh(times_test, freqs_test_s, Sxx_test, shading='gouraud', cmap='viridis')
    # plt.title("Spectrogramme Test")
    # plt.xlabel("Temps (s)")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()

    return score_hypo, score_hyper, score_spectro_hypo, score_spectro_hyper, alarm_message, alarm_type