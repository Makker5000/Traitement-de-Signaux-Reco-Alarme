# def Comparison():
#     return 0

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io.wavfile import read
# from scipy.fft import fft, fftfreq
# from scipy.spatial.distance import euclidean

# alarme_hypo = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav"
# alarme_hyper = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav"
# alarme_test = "../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-bruit-Strident-derriere.wav"

# def load_audio(filename):
#     # Charge le fichier audio et normalise le signal
#     rate, data = read(filename)
#     if data.ndim > 1:  # Pour les fichiers stéréo, garder un canal
#         data = data[:, 0]
#     data = data / np.max(np.abs(data))  # Normalisation
#     return rate, data

# def compute_fft(signal, rate, n_points=None):
#     # Si `n_points` n'est pas précisé, utilise la taille du signal
#     if n_points is None:
#         n_points = len(signal)
#     elif n_points > len(signal):
#         raise ValueError("n_points ne peut pas être supérieur à la longueur du signal")
        
#     # Troncation ou padding du signal pour correspondre à n_points
#     if len(signal) > n_points:
#         signal = signal[:n_points]  # Tronque le signal si nécessaire
#     else:
#         # Ajoute des zéros si le signal est plus court
#         signal = np.pad(signal, (0, n_points - len(signal)), 'constant')

#     # Calcul de la transformée de Fourier
#     yf = fft(signal)
#     xf = fftfreq(n_points, 1 / rate)
#     return xf[:n_points // 2], np.abs(yf[:n_points // 2]) # Retourne uniquement les fréquences positives


# def compare_signals(reference_1, reference_2, test_signal):
#     # Calcul de la similarité basée sur la distance euclidienne des spectres de fréquence
#     dist_1 = euclidean(reference_1, test_signal)
#     dist_2 = euclidean(reference_2, test_signal)
    
#     # Détermine la classification en fonction des distances
#     if dist_1 < dist_2 and dist_1 < 2.0:  # Seuil à ajuster
#         return "Hypoglycémie"
#     elif dist_2 < dist_1 and dist_2 < 2.0:
#         return "Hyperglycémie"
#     else:
#         return "Indéterminé"

# def runComparison():
#     # Chargement des fichiers audio de référence et du signal test
#     rate_ref_hypo, ref_hypo = load_audio(alarme_hypo)
#     rate_ref_hyper, ref_hyper = load_audio(alarme_hyper)
#     rate_test, test_signal = load_audio(alarme_test)

#     print(f"Taux echantillonage hypo {rate_ref_hypo}")
#     print(f"Taux echantillonage hyper {rate_ref_hyper}")
#     print(f"Taux echantillonage test {rate_test}")

#     # Obtenir le nombre de points minimal parmi les signaux
#     n_points = min(len(ref_hypo), len(ref_hyper), len(test_signal))

#     # Vérifie que les taux d'échantillonnage sont identiques
#     if rate_ref_hypo != rate_test & rate_ref_hyper != rate_test:
#         raise ValueError("Les taux d'échantillonnage des fichiers doivent être identiques.")

#     # Calcul des spectres de fréquence
#     xf, spectrum_hypo = compute_fft(ref_hypo, rate_ref_hypo, n_points)
#     xf, spectrum_hyper = compute_fft(ref_hyper, rate_ref_hyper, n_points)
#     xf, spectrum_test = compute_fft(test_signal, rate_test, n_points)

#     # Affichage des spectres
#     plt.figure(figsize=(10, 6))
#     plt.plot(xf, spectrum_hypo, label="Hypoglycémie", color='blue')
#     plt.plot(xf, spectrum_hyper, label="Hyperglycémie", color='red')
#     plt.plot(xf, spectrum_test, label="Signal Test", color='green')
#     plt.xlabel("Fréquence (Hz)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.title("Comparaison des spectres de fréquence")
#     plt.show()

#     # Comparaison et classification
#     result = compare_signals(spectrum_hypo, spectrum_hyper, spectrum_test)
#     print(f"Résultat de la classification : {result}")