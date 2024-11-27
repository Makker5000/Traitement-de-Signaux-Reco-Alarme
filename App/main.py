from Analysis import *
from Processing import *
from Comparison import *
from Result import *

def main():

    fs, signal = process("Ressources\Sons-de-Test\Son-Alarme-Hypo-bruit-Strident-derriere.wav")
    fs_fen, signal_fen = extraire_son_hyper_hypo(fs, signal)
    score_hypo, score_hyper, score_spectro_hypo, score_spectro_hyper, isAlarm_result, alarm_type = runComparison(fs_fen, signal_fen)

    generate_alarm_image(alarm_type)


main()