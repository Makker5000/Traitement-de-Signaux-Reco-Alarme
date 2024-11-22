import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate, find_peaks, spectrogram
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

alarme_hypo = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
alarme_hyper = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-bruit-Strident-derriere.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-Pitch-vers-le-Haut-100cents.wav"

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

# Détection des pics dans le spectre
# def detect_peaks(freqs, spectrum, threshold=0.1, distance=20):
#     peaks, _ = find_peaks(spectrum, height=threshold * np.max(spectrum), distance=distance)
#     peak_freqs = freqs[peaks]
#     return peak_freqs

# Calcul de similarité en utilisant les fréquences dominantes
# def calculate_peak_similarity(peaks_ref, peaks_test):
#     shared_peaks = np.intersect1d(peaks_ref, peaks_test, assume_unique=True)
#     similarity = len(shared_peaks) / max(len(peaks_ref), len(peaks_test))
#     return similarity * 100

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

    # Tester de faire une corrélation sur base d'un Spectrogram de mon Signal.
    # La STFT est une fonction utilisée justement lorsqu'on fait un Spectrogram.

    return score_1 * 100, score_2 * 100  # Pourcentage de ressemblance

def runComparison():
    # Chargement des fichiers de sons d'alarme Hypo et Hyper
    rate_hypo, alarm_hypo = load_audio(alarme_hypo)
    rate_hyper, alarm_hyper = load_audio(alarme_hyper)
    rate_test, test_alarm = load_audio(alarme_test)

    # Vérification des taux d'échantillonnage
    if rate_hypo != rate_hyper or rate_hypo != rate_test:
        raise ValueError("Les fichiers audio doivent avoir le même taux d'échantillonnage.")
    
    # Nombre de points pour le calcul de la FFT
    n_points = min(len(alarm_hypo), len(alarm_hyper), len(test_alarm))

    # Calcul des FFT pour chaque signal
    freqs_hypo, spectrum_hypo = compute_fft(alarm_hypo, rate_hypo, n_points=n_points)
    freqs_hyper, spectrum_hyper = compute_fft(alarm_hyper, rate_hyper, n_points=n_points)
    freqs_test, spectrum_test = compute_fft(test_alarm, rate_test, n_points=n_points)

    # Détection des pics dans les spectres
    # peaks_hypo = detect_peaks(freqs_hypo, spectrum_hypo)
    # peaks_hyper = detect_peaks(freqs_hyper, spectrum_hyper)
    # peaks_test = detect_peaks(freqs_test, spectrum_test)

    # # Similarité basée sur les pics
    # peak_similarity_hypo = calculate_peak_similarity(peaks_hypo, peaks_test)
    # peak_similarity_hyper = calculate_peak_similarity(peaks_hyper, peaks_test)

    # Correction de pitch sur le spectre de test
    spectrum_test = align_spectra(spectrum_hypo, spectrum_test, freqs_hypo, freqs_test)

    # Calcul des scores de similarité
    score_hypo, score_hyper = calculate_similarity(spectrum_hypo, spectrum_hyper, spectrum_test)

    # Pondération des résultats
    # combined_score_hypo = 0.6 * score_hypo + 0.4 * peak_similarity_hypo
    # combined_score_hyper = 0.6 * score_hyper + 0.4 * peak_similarity_hyper

    # # Détermination du type d'alarme
    if score_hypo > score_hyper and score_hypo > 50:  # Ajuster le seuil selon les besoins
        result = "Hypoglycémie"
    elif score_hyper > score_hypo and score_hyper > 50:
        result = "Hyperglycémie"
    else:
        result = "Indéterminé"

    # if combined_score_hypo > combined_score_hyper and combined_score_hypo > 50:
    #     result = "Hypoglycémie"
    # elif combined_score_hyper > combined_score_hypo and combined_score_hyper > 50:
    #     result = "Hyperglycémie"
    # else:
    #     result = "Indéterminé"

    # Affichage des résultats
    print(f"Score de similarité avec l'alarme Hypoglycémie : {score_hypo:.2f}%  OU {score_hypo}")
    print(f"Score de similarité avec l'alarme Hyperglycémie : {score_hyper:.2f}% OU {score_hyper}")
    print(f"Résultat de la classification : {result}")

    # print(f"Score de similarité avec l'alarme Hypoglycémie : {combined_score_hypo:.2f}%")
    # print(f"Score de similarité avec l'alarme Hyperglycémie : {combined_score_hyper:.2f}%")
    # print(f"Résultat de la classification : {result}")

    # # (Optionnel) Visualisation des spectres pour vérifier les similarités
    # Affichage des spectres
    plt.figure(figsize=(12, 6))
    plt.plot(freqs_hypo, spectrum_hypo, label="Spectre Hypoglycémie", color='blue', alpha=0.7)
    plt.plot(freqs_hyper, spectrum_hyper, label="Spectre Hyperglycémie", color='red', alpha=0.7)
    plt.plot(freqs_test, spectrum_test, label="Spectre Test", color='green', linestyle='--', alpha=0.7)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Comparaison des spectres de fréquence")
    plt.show()