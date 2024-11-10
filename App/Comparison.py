import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

alarme_hypo = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
alarme_hyper = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-bruit-Strident-derriere.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
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
    return freqs[:len(freqs)//2], spectrum[:len(spectrum)//2]

# Fonction pour calculer les similarités entre les spectres de fréquences
def calculate_similarity(reference_1, reference_2, test_signal):
    # Normalisation des spectres pour la comparaison
    ref1_norm = reference_1 / np.linalg.norm(reference_1)
    ref2_norm = reference_2 / np.linalg.norm(reference_2)
    test_norm = test_signal / np.linalg.norm(test_signal)

    # Calcul de la distance euclidienne
    similarity_euclidean_1 = 1 / (1 + euclidean(ref1_norm, test_norm))
    similarity_euclidean_2 = 1 / (1 + euclidean(ref2_norm, test_norm))

    # Calcul de la similarité cosinus
    similarity_cosine_1 = 1 - cosine(ref1_norm, test_norm)
    similarity_cosine_2 = 1 - cosine(ref2_norm, test_norm)

    # Calcul de la corrélation croisée
    correlation_1 = np.max(correlate(ref1_norm, test_norm)) / len(ref1_norm)
    correlation_2 = np.max(correlate(ref2_norm, test_norm)) / len(ref2_norm)

    # Score final (pondération ajustable)
    score_1 = 0.4 * similarity_euclidean_1 + 0.4 * similarity_cosine_1 + 0.2 * correlation_1
    score_2 = 0.4 * similarity_euclidean_2 + 0.4 * similarity_cosine_2 + 0.2 * correlation_2

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


    # Calcul des scores de similarité
    score_hypo, score_hyper = calculate_similarity(spectrum_hypo, spectrum_hyper, spectrum_test)

    # Détermination du type d'alarme
    if score_hypo > score_hyper and score_hypo > 50:  # Ajuster le seuil selon les besoins
        result = "Hypoglycémie"
    elif score_hyper > score_hypo and score_hyper > 50:
        result = "Hyperglycémie"
    else:
        result = "Indéterminé"

    # Affichage des résultats
    print(f"Score de similarité avec l'alarme Hypoglycémie : {score_hypo:.2f}%")
    print(f"Score de similarité avec l'alarme Hyperglycémie : {score_hyper:.2f}%")
    print(f"Résultat de la classification : {result}")

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