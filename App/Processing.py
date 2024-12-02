import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt, resample_poly
import soundfile as sf

def compute_fft(signal, rate, n_points=None):
    """
    Calcule la Transformée de Fourier rapide (FFT) d'un signal et retourne les fréquences associées et l'amplitude du spectre.

    Args:
        signal (numpy.ndarray): Signal à analyser, représenté comme un tableau 1D.
        rate (int): Fréquence d'échantillonnage du signal en Hz.
        n_points (int, optionnel): Nombre de points souhaités pour le calcul de la FFT. 
            - Si le signal est plus long, il sera tronqué.
            - Si le signal est plus court, il sera complété par des zéros (padding).
            - Si `None`, la FFT est calculée sur la longueur du signal.

    Returns:
        tuple:
            - numpy.ndarray: Tableau contenant les fréquences associées (uniquement la moitié positive du spectre).
            - numpy.ndarray: Tableau contenant l'amplitude du spectre (uniquement la moitié positive du spectre).

    Comportement:
        - Si `n_points` est spécifié, ajuste la taille du signal avant de calculer la FFT.
        - La moitié positive du spectre est retournée, car la FFT produit un spectre symétrique pour les signaux réels.

    Exemple:
        >>> import numpy as np
        >>> from scipy.signal import chirp
        >>> fs = 1000  # Fréquence d'échantillonnage
        >>> t = np.linspace(0, 1, fs, endpoint=False)  # 1 seconde de signal
        >>> signal = chirp(t, f0=10, f1=100, t1=1, method='linear')  # Signal chirp
        >>> freqs, spectrum = compute_fft(signal, fs)
        >>> print(freqs[:10])  # Affiche les 10 premières fréquences
        >>> print(spectrum[:10])  # Affiche les 10 premières amplitudes
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

def align_spectra(ref_spectrum, test_spectrum, freqs_ref, freqs_test):
    """
    Aligne un spectre de test sur un spectre de référence en corrigeant les décalages de fréquence.

    Args:
        ref_spectrum (numpy.ndarray): Spectre de référence, contenant les amplitudes des fréquences.
        test_spectrum (numpy.ndarray): Spectre de test, contenant les amplitudes des fréquences.
        freqs_ref (numpy.ndarray): Tableau des fréquences associées au spectre de référence.
        freqs_test (numpy.ndarray): Tableau des fréquences associées au spectre de test.

    Returns:
        numpy.ndarray: Spectre de test ajusté pour correspondre au spectre de référence.

    Comportement :
        - Identifie le pic principal dans chaque spectre.
        - Calcule le ratio de décalage de fréquence entre les spectres.
        - Applique une correction pour aligner les fréquences du spectre de test avec celles du spectre de référence.
        - Utilise une interpolation linéaire pour ajuster les valeurs du spectre de test.

    Exemple:
        >>> import numpy as np
        >>> ref_spectrum = np.array([0, 1, 3, 7, 5, 2, 1])  # Spectre de référence
        >>> test_spectrum = np.array([0, 2, 6, 14, 10, 4, 2])  # Spectre de test décalé
        >>> freqs_ref = np.linspace(0, 100, len(ref_spectrum))  # Fréquences associées à ref_spectrum
        >>> freqs_test = np.linspace(0, 120, len(test_spectrum))  # Fréquences associées à test_spectrum
        >>> adjusted_test_spectrum = align_spectra(ref_spectrum, test_spectrum, freqs_ref, freqs_test)
        >>> print(adjusted_test_spectrum)
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
    highcuts = [3950.0, 4250.0, 4520.0, 4880.0, 5300.0]

    # Appliquer les filtres
    filtered_signal = combine_bandpass_filters(signal, fs, lowcuts, highcuts)

    return fs, filtered_signal
