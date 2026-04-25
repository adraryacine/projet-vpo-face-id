"""
==============================================================================
FICHIER : detection.py
ROLE    : Détection du visage par cascade de Haar (algorithme de Viola-Jones)
          et alignement géométrique basé sur la position des yeux.

THÉORIE
-------
L'algorithme de Viola-Jones (2001) utilise des classificateurs en cascade
entraînés sur des caractéristiques de Haar (différences de luminosité entre
zones rectangulaires adjacentes). Il est très rapide car il rejette rapidement
les zones sans visage (fenêtre glissante).

Étapes de ce module :
  1. Convertir en niveaux de gris (la détection Haar ne nécessite pas la couleur)
  2. Localiser le (ou les) visage(s) → boîte englobante (x, y, w, h)
  3. Prendre le plus grand visage si plusieurs sont détectés
  4. Détecter les yeux dans la région du visage (ROI = Region of Interest)
  5. Calculer l'angle d'inclinaison via arctan(Δy / Δx) entre les deux yeux
  6. Corriger l'inclinaison par rotation affine
  7. Recadrer, redimensionner à 128×128 pixels
  8. Égaliser l'histogramme (normalisation de la luminosité)

SORTIE : image 128×128 en niveaux de gris, prête pour le Snake
==============================================================================
"""

import cv2
import numpy as np
import os


# ==============================================================================
# CHARGEMENT DES CLASSIFICATEURS HAAR
# Les fichiers XML sont fournis avec OpenCV. On les cherche dans le dossier
# local 'haarcascades/' d'abord, sinon dans les données OpenCV installées.
# ==============================================================================

def _trouver_haar(nom_fichier):
    """
    Cherche le fichier de cascade Haar dans deux endroits :
    1. Dossier local haarcascades/ (à créer dans le projet)
    2. Répertoire de données OpenCV (installé avec le package)

    Paramètre
    ---------
    nom_fichier : str — ex: 'haarcascade_frontalface_default.xml'

    Retourne
    --------
    str : chemin absolu vers le fichier XML
    """
    # Chemin local (à côté de ce fichier)
    chemin_local = os.path.join(
        os.path.dirname(__file__), 'haarcascades', nom_fichier
    )
    if os.path.exists(chemin_local):
        return chemin_local

    # Chemin OpenCV (fourni automatiquement avec cv2)
    chemin_opencv = os.path.join(
        cv2.data.haarcascades, nom_fichier
    )
    if os.path.exists(chemin_opencv):
        return chemin_opencv

    raise FileNotFoundError(
        f"Fichier introuvable : {nom_fichier}\n"
        "Placez-le dans haarcascades/ ou installez opencv-python correctement."
    )


# Chargement une seule fois au démarrage (optimisation : évite de recharger
# les XML à chaque appel de fonction)
CASCADE_VISAGE = cv2.CascadeClassifier(
    _trouver_haar('haarcascade_frontalface_default.xml')
)
CASCADE_VISAGE_ALT2 = cv2.CascadeClassifier(
    _trouver_haar('haarcascade_frontalface_alt2.xml')
)
CASCADE_YEUX = cv2.CascadeClassifier(
    _trouver_haar('haarcascade_eye.xml')
)


def _detecter_visages(gris):
    gris_eq = cv2.equalizeHist(gris)
    for cascade, voisins in ((CASCADE_VISAGE, 5), (CASCADE_VISAGE_ALT2, 4)):
        visages = cascade.detectMultiScale(
            gris_eq,
            scaleFactor=1.1,
            minNeighbors=voisins,
            minSize=(60, 60)
        )
        if len(visages) > 0:
            return visages
    return []


def _choisir_paire_yeux(yeux, largeur_roi, hauteur_roi):
    if len(yeux) < 2:
        return None

    centres = []
    for ex, ey, ew, eh in yeux:
        centres.append((ex, ey, ew, eh, ex + ew / 2.0, ey + eh / 2.0))

    meilleur = None
    meilleur_score = -1.0
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            e1, e2 = centres[i], centres[j]
            sep_x = abs(e2[4] - e1[4])
            sep_y = abs(e2[5] - e1[5])
            if sep_x < largeur_roi * 0.25:
                continue
            if sep_y > hauteur_roi * 0.18:
                continue

            score = sep_x - sep_y * 2.0
            if score > meilleur_score:
                meilleur_score = score
                meilleur = (e1[:4], e2[:4])

    if meilleur is None:
        return None
    return sorted(meilleur, key=lambda e: e[0])


# ==============================================================================
# FONCTION PRINCIPALE
# ==============================================================================

def detecter_et_aligner(image, taille_sortie=128):
    """
    Détecte le visage principal dans une image et le normalise.

    Paramètres
    ----------
    image        : numpy array BGR (image couleur de la webcam)
    taille_sortie: int — côté du carré de sortie en pixels (défaut 128)

    Retourne
    --------
    visage_normalise : numpy array uint8 de forme (128, 128)
                       en niveaux de gris, égalisé histogramme
    None             : si aucun visage n'est détecté
    """

    # --- Étape 1 : Conversion en niveaux de gris ---
    # Haar fonctionne sur des images monocanal.
    # cvtColor convertit BGR (format OpenCV) → GRAY
    if len(image.shape) == 3:
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gris = image.copy()   # Déjà en niveaux de gris

    # --- Étape 2 : Détection des visages ---
    # detectMultiScale analyse l'image à plusieurs échelles (pyramide d'images)
    # scaleFactor = 1.1  → l'image est réduite de 10% à chaque niveau
    # minNeighbors = 5   → un visage doit être confirmé par 5 détections voisines
    #                       (réduit les faux positifs)
    # minSize         → taille minimale d'un visage acceptable (filtre le bruit)
    visages = _detecter_visages(gris)

    if len(visages) == 0:
        return None   # Aucun visage → le pipeline ne peut pas continuer

    # --- Étape 3 : Sélection du plus grand visage ---
    # Si plusieurs visages sont détectés (ex: deux personnes dans le champ),
    # on prend celui avec la plus grande aire (w×h), qui est probablement
    # le visage le plus proche de la caméra.
    x, y, w, h = max(visages, key=lambda v: v[2] * v[3])

    # Extraire la région du visage (ROI = Region Of Interest)
    roi_gris = gris[y:y+h, x:x+w]

    # --- Étape 4 : Détection des yeux dans le ROI ---
    # On cherche les yeux uniquement dans la moitié haute du visage
    # (les yeux sont toujours dans les 60% supérieurs)
    hauteur_roi = roi_gris.shape[0]
    roi_yeux = roi_gris[0:int(hauteur_roi * 0.6), :]
    roi_yeux = cv2.equalizeHist(roi_yeux)

    yeux = CASCADE_YEUX.detectMultiScale(
        roi_yeux,
        scaleFactor=1.1,
        minNeighbors=10,   # Plus strict pour réduire les faux positifs
        minSize=(20, 20)
    )
    paire_yeux = _choisir_paire_yeux(yeux, roi_gris.shape[1], roi_yeux.shape[0])

    # Si on ne détecte pas exactement 2 yeux, on fait un cadrage simple
    # sans correction d'angle (le Snake compensera en partie)
    if paire_yeux is None:
        visage = roi_gris.copy()
        visage = cv2.resize(visage, (taille_sortie, taille_sortie))
        visage = cv2.equalizeHist(visage)
        return visage

    # Garder uniquement les 2 premiers yeux détectés
    yeux = paire_yeux

    # Calculer le centre de chaque œil
    (ex1, ey1, ew1, eh1) = yeux[0]
    (ex2, ey2, ew2, eh2) = yeux[1]

    # Centre de l'œil gauche (dans le repère du ROI)
    cx_gauche = x + ex1 + ew1 // 2
    cy_gauche = y + ey1 + eh1 // 2

    # Centre de l'œil droit
    cx_droit  = x + ex2 + ew2 // 2
    cy_droit  = y + ey2 + eh2 // 2

    # --- Étape 5 : Calcul de l'angle d'inclinaison ---
    # On utilise arctan2 pour obtenir l'angle entre la droite reliant les deux
    # yeux et l'axe horizontal.
    # arctan2(Δy, Δx) donne l'angle en radians → converti en degrés
    #
    # Exemple : si l'œil droit est 5px plus haut que l'œil gauche,
    #           angle ≈ arctan(-5/distance_inter_oculaire) → négatif → rotation horaire
    delta_y = cy_droit - cy_gauche
    delta_x = cx_droit - cx_gauche
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    # --- Étape 6 : Rotation de l'image entière ---
    # On tourne autour du centre du visage détecté
    centre_visage = (int(x + w // 2), int(y + h // 2))
    angle_deg = float(angle_deg)
    # getRotationMatrix2D retourne une matrice de transformation affine 2×3
    # Arguments : centre de rotation, angle (négatif = sens antihoraire), échelle
    M = cv2.getRotationMatrix2D(centre_visage, angle_deg, scale=1.0)

    # warpAffine applique la transformation affine à l'image entière
    # On conserve les dimensions originales (les bords peuvent être noirs)
    image_tournee = cv2.warpAffine(
        gris, M,
        (gris.shape[1], gris.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE   # Remplir les bords avec les pixels adjacents
    )

    # --- Étape 7 : Recadrage et redimensionnement ---
    # On recadre la région du visage dans l'image tournée
    # Note : après rotation, le visage est toujours approximativement au même endroit
    # car on a tourné autour de son centre
    visage = image_tournee[y:y+h, x:x+w]

    # Sécurité : si le recadrage sort de l'image (bords), on ajuste
    if visage.shape[0] == 0 or visage.shape[1] == 0:
        visage = roi_gris   # Fallback : ROI sans correction d'angle

    # Redimensionner à 128×128 (taille standard du projet)
    # INTER_AREA est recommandé pour la réduction (meilleure qualité)
    visage = cv2.resize(visage, (taille_sortie, taille_sortie),
                        interpolation=cv2.INTER_AREA)

    # --- Étape 8 : Égalisation de l'histogramme ---
    # L'égalisation étale la distribution des niveaux de gris sur [0, 255],
    # ce qui rend l'image robuste aux variations d'éclairage.
    # Exemple : une image sombre verra ses contrastes amplifiés automatiquement.
    visage = cv2.equalizeHist(visage)

    return visage   # numpy array uint8, shape (128, 128)


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def dessiner_detection(image):
    """
    Variante de visualisation : dessine les boîtes de détection
    directement sur l'image originale (utile pour le débogage).

    Paramètre
    ---------
    image : numpy array BGR

    Retourne
    --------
    image avec rectangles dessinés autour des visages/yeux détectés
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_debug = image.copy()

    # Détecter les visages
    visages = CASCADE_VISAGE.detectMultiScale(gris, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in visages:
        # Rectangle autour du visage (bleu)
        cv2.rectangle(img_debug, (x, y), (x+w, y+h), (255, 100, 0), 2)
        cv2.putText(img_debug, "Visage", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,0), 2)

        # Chercher les yeux dans le ROI
        roi = gris[y:y+h, x:x+w]
        roi_yeux = roi[0:int(h*0.6), :]
        yeux = CASCADE_YEUX.detectMultiScale(roi_yeux, 1.1, 10)

        for (ex, ey, ew, eh) in yeux[:2]:
            # Rectangle autour de chaque œil (vert)
            cv2.rectangle(img_debug,
                          (x+ex, y+ey), (x+ex+ew, y+ey+eh),
                          (0, 255, 0), 2)

    return img_debug
