from Analysis import *
from Processing import *
from Comparison import *

def main():
    fs, signal = Processing("Ressources\Sons-de-Test\Son-Alarme-Hypo-bruit-Strident-derriere.wav")
    fs_fen, signal_fen = extraire_son_hyper_hypo(fs, signal)
    runComparison(fs_fen, signal_fen)

main()