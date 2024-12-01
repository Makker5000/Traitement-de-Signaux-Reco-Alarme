from Analysis import *
from Processing import *
from Comparison import *
from Result import *

def main():
    #-------------------------------------------------------------------------------------------------
    # LECTURE DU SIGNAL ORIGINAL ET FILTRAGE DU SIGNAL ORIGINAL
    #-------------------------------------------------------------------------------------------------
    fs, signal = process("../Ressources/Sons-de-Ref/Son-Alarme-Hypo-Clean.wav")
    #-------------------------------------------------------------------------------------------------
    # EXTRACTION DU SON HYPER/HYPO
    #-------------------------------------------------------------------------------------------------
    fs_fen, signal_fen = extraire_son_hyper_hypo(fs, signal)
    #-------------------------------------------------------------------------------------------------
    # COMPARAISON AVEC LES SIGNAUX AUDIOS DE RÉFÉRENCES HYPERGLYCÉMIQUE/HYPOGLYCÉMIQUE
    #-------------------------------------------------------------------------------------------------
    score_hypo, score_hyper, score_spectro_hypo, score_spectro_hyper, isAlarm_result, alarm_type = runComparison(fs_fen, signal_fen)
    #-------------------------------------------------------------------------------------------------
    # IMAGE MONTRANT LE RÉSULTAT
    #-------------------------------------------------------------------------------------------------
    generate_alarm_image(alarm_type)


main()