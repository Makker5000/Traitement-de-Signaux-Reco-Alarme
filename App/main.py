import os
from Analysis import *
from Processing import *
from Comparison import *
from Result import *

def main():
    #-------------------------------------------------------------------------------------------------
    # RÉPERTOIRE CONTENANT LES FICHIERS AUDIO
    #-------------------------------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Base directory (Traitement-de-Signaux-Reco-Alarme)
    test_sounds_dir = os.path.join(base_dir, "Ressources", "Sons-de-Test")
    
    print("Veuillez sélectionner un fichier parmi les suivants :")
    files = [f for f in os.listdir(test_sounds_dir) if os.path.isfile(os.path.join(test_sounds_dir, f))]
    
    if not files:
        print("Aucun fichier trouvé dans le répertoire :", test_sounds_dir)
        return

    for i, file in enumerate(files):
        print(f"{i + 1}. {file}")
    
    while True:
        try:
            choice = int(input("Entrez le numéro correspondant au fichier : "))
            if 1 <= choice <= len(files):
                selected_file = os.path.join(test_sounds_dir, files[choice - 1])
                print(f"Vous avez sélectionné : {files[choice - 1]}")
                break
            else:
                print("Numéro invalide, veuillez réessayer.")
        except ValueError:
            print("Entrée invalide, veuillez entrer un numéro.")
    
    #-------------------------------------------------------------------------------------------------
    # LECTURE DU SIGNAL ORIGINAL ET FILTRAGE DU SIGNAL ORIGINAL
    #-------------------------------------------------------------------------------------------------
    fs, signal = process(selected_file)
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


if __name__ == "__main__":
    main()