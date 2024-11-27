from Comparison import runComparison
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_alarm_image(alarm_type):
    width, height = 800, 400
    # Choisir les couleurs de base pour le dégradé selon le type d'alarme
    if alarm_type == "Hyperglycémie":
        base_color_start = (255, 0, 0)  # Rouge
        base_color_end = (255, 255, 0)  # Jaune
        icon = "↑"  # Flèche montante pour Hyperglycémie
        arrow_color = (255, 255, 255)  # Blanc pour la flèche
    elif alarm_type == "Hypoglycémie":
        base_color_start = (0, 0, 255)  # Bleu
        base_color_end = (0, 255, 0)  # Vert
        icon = "↓"  # Flèche descendante pour Hypoglycémie
        arrow_color = (255, 255, 255)  # Blanc pour la flèche
    else:
        base_color_start = (169, 169, 169)  # Gris
        base_color_end = (169, 169, 169)  # Gris
        icon = "?"  # Icône neutre pour inconnu
        arrow_color = (169, 169, 169)  # Gris pour une flèche neutre

    # Créer l'image avec un fond dégradé dynamique
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Créer un dégradé de couleur pour le fond
    for i in range(height):
        ratio = i / height
        color = tuple(
            int(start + ratio * (end - start)) for start, end in zip(base_color_start, base_color_end)
        )
        draw.line((0, i, width, i), fill=color)

    # Ajouter une onde sonore (graphique)
    wave = np.sin(np.linspace(0, 2 * np.pi, width)) * 50 + height // 2
    wave = wave.astype(int)
    
    # Tracer la courbe (ligne de la vague)
    for x, y in enumerate(wave):
        draw.line((x, height // 2, x, y), fill=(255, 255, 255))

    # Déplacer la courbe vers le bas pour laisser de l'espace pour le texte
    text_area_offset = 100
    wave_offset = text_area_offset

    # Ajouter une flèche stylisée
    if alarm_type == "Hyperglycémie":
        draw.line((width // 2, wave_offset, width // 2, wave_offset - 50), fill=arrow_color, width=6)  # Tige de la flèche
        draw.line([(width // 2 - 20, wave_offset - 50), (width // 2 + 20, wave_offset - 50)], fill=arrow_color, width=6)  # Bas de la flèche
    elif alarm_type == "Hypoglycémie":
        draw.line((width // 2, wave_offset, width // 2, wave_offset + 50), fill=arrow_color, width=6)  # Tige de la flèche
        draw.line([(width // 2 - 20, wave_offset + 50), (width // 2 + 20, wave_offset + 50)], fill=arrow_color, width=6)  # Bas de la flèche

    # Ajouter du texte pour le type d'alarme (avec fond pour la lisibilité)
    text = alarm_type
    try:
        font = ImageFont.truetype("arial.ttf", 80)
    except IOError:
        font = ImageFont.load_default()

    # Calculer la position du texte
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]  # Largeur du texte
    text_height = text_bbox[3] - text_bbox[1]  # Hauteur du texte
    position = ((width - text_width) // 2, wave_offset + 50)  # Placer le texte sous la flèche

    # Ajouter un fond sombre sous le texte pour améliorer la lisibilité
    draw.rectangle([position[0] - 20, position[1] - 10, position[0] + text_width + 20, position[1] + text_height + 10], fill=(0, 0, 0))

    # Ajouter le texte de l'alarme
    draw.text(position, text, fill="white", font=font)

    # Afficher l'image
    img.show()
