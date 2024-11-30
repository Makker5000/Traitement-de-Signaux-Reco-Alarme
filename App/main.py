from Analysis import *
from Processing import *
from Comparison import *
from Result import *

def main():
    #-------------------------------------------------------------------------------------------------
    # LECTURE DU SIGNAL ORIGINAL ET FILTRAGE DU SIGNAL ORIGINAL
    #-------------------------------------------------------------------------------------------------
    fs, signal = process("../Traitement-de-Signaux-Reco-Alarme/Ressources/Sons-de-Test/Son-Alarme-Hypo-Pitch-vers-le-Haut-100cents.wav")
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