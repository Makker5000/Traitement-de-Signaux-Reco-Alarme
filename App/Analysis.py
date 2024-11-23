import numpy as np
from scipy.io.wavfile import read
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
import sounddevice as sd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------
# PLAGES DE FRÉQUENCES
#-------------------------------------------------------------------------------------------------

# plage de fréquences fondamentales du son hyper : 1200-5600 Hz
# plage de fréquences fondamentales du son hyper : 


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
# SON ORIGINAL
#-------------------------------------------------------------------------------------------------

# Lecture du son Original
# Importation et ouverture du fichier son

#fe, signal_bruite = load_audio('../Ressources/Sons-avec-bruit/Hyper-chien.wav')
fe, signal_bruite = load_audio('../Ressources/Sons-avec-bruit/Hyper-discussion_1.wav')
#fe, signal_bruite = load_audio('../Ressources/Sons-de-Ref/Son-Alarme-Hyper-Clean.wav')
signal_bruite = signal_bruite / np.max(np.abs(signal_bruite)) # Normaliser le signal si besoin (pour les fichiers wav int16)

print(f"Fréquence d'échantillonnage: {fe} Hz")
print(f"Nombre d'échantillons: {len(signal_bruite)}")

#-------------------------------------------------------------------------------------------------
# ÉCOUTE DU SON ORIGINALE + ANALYSE SPECTRALE
#-------------------------------------------------------------------------------------------------

# # Ecouter le son Original
# sd.play(signal_bruite, fe)
# sd.wait()
# print(f"Lecture terminée")

# Analyse spectrale pour identifier la fréquence du bruit
t_original1 = np.arange(len(signal_bruite)) / fe
TF_original = np.fft.fft(signal_bruite)
frequencies = np.fft.fftfreq(len(TF_original), d=1/fe)

# Affichage du spectre pour visualiser le signal bruité
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(TF_original[:len(frequencies)//2]))
plt.title('Spectre de la chanson bruitée')
plt.xlabel('Fréquence [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)

#-------------------------------------------------------------------------------------------------
# FRÉQUENCES DE COUPURE POUR FILTRE PASS-BANDE
#-------------------------------------------------------------------------------------------------

#fc1, fc2 = 1000, 5600  # Hyper
#fc1, fc2 =  # Hypo

#-------------------------------------------------------------------------------------------------
# FILTRE
#-------------------------------------------------------------------------------------------------

# FILTRE PASSE-BANDE pour conserver uniquement la plage de fréquences correspondant au son hyper/hypo

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

# Appliquer le filtre passe-bande
lowcut = 3000  # Borne inférieure (1000 Hz)
highcut = 5600  # Borne supérieure (5600 Hz)
b, a = butter_bandpass(lowcut, highcut, fe, order=4)
signal_filtre = filtfilt(b, a, signal_bruite)



# FILTRE COUPE-BANDE pour éliminer les bruits parasites (ex : 400 Hz et 800 Hz)
# fc_bruit1, fc_bruit2 = [1500, 3500]  # Fréquences à atténuer
# for fc in [fc_bruit1, fc_bruit2]:
#     b, a = butter(4, [fc - 50, fc + 50], btype="bandstop", fs=fe)
#     signal_filtre = filtfilt(b, a, signal_filtre)


# Affichage temporel le signal filtré
plt.figure(figsize=(10, 6))
plt.plot(signal_filtre)
plt.title("Signal filtré (hyperglycémie isolée)")
plt.xlabel("Temps [échantillons]")
plt.ylabel("Amplitude")
plt.grid(True)


# Écouter le signal filtré
# sd.play(signal_filtre, fe)
# sd.wait()

#Spectre fréqueniel du signal filtré(TF)
TF_filtered = np.fft.fft(signal_filtre)
frequencies_filtered = np.fft.fftfreq(len(TF_filtered), 1 / fe)

#Analyse spectrale du signal filtré
plt.figure(figsize=(10, 6))
plt.plot(frequencies_filtered[:len(frequencies_filtered)//2], np.abs(TF_filtered[:len(TF_filtered)//2]))
plt.title("Spectre après filtrage")
plt.xlabel("Fréquence [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)


#-------------------------------------------------------------------------------------------------
# EXTRACTION DU SON HYPER/HYPO
#-------------------------------------------------------------------------------------------------

# # Paramètres pour le fenêtrage
# taille_fenetre = 1024  # Taille de la fenêtre FFT
# overlap = 512  # Recouvrement entre fenêtres
# frequencies, times, spectro = spectrogram(signal_filtre, fs=fe, nperseg=taille_fenetre, noverlap=overlap)

# # Filtrer le spectrogramme pour ne garder que les fréquences hyperglycémiques
# indices_freq_hyper = np.where((frequencies >= 3000) & (frequencies <= 5600))[0]
# spectro_hyper = spectro[indices_freq_hyper, :]

# # Calculer l'énergie moyenne pour chaque fenêtre temporelle dans la plage hyperglycémique
# energie_hyper = np.mean(spectro_hyper, axis=0)

# # Détecter les fenêtres où l'énergie hyperglycémique est significative
# seuil_energie = np.mean(energie_hyper) + 1.5 * np.std(energie_hyper)  # Seuil basé sur la moyenne et l'écart-type
# fenetres_significatives = np.where(energie_hyper > seuil_energie)[0]

# # Extraire le signal temporel correspondant
# if len(fenetres_significatives) > 0:
#     debut = int(times[fenetres_significatives[0]] * fe)
#     fin = int(times[fenetres_significatives[-1]] * fe)
#     signal_reduit = signal_filtre[debut:fin]
#     t_reduit = t_original1[debut:fin]
# else:
#     print("Aucune région avec une énergie hyperglycémique significative détectée.")
#     signal_reduit = None
#     t_reduit = None

# # Affichage du signal réduit
# if signal_reduit is not None:
#     plt.figure(figsize=(10, 6))
#     plt.plot(t_reduit, signal_reduit)
#     plt.title("Signal réduit basé sur les fréquences hyperglycémiques")
#     plt.xlabel("Temps [s]")
#     plt.ylabel("Amplitude")
#     plt.grid(True)

#     # Écoute du signal réduit
#     sd.play(signal_reduit, fe)
#     sd.wait()



plt.show()