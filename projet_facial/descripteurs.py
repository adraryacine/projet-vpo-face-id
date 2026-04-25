"""
==============================================================================
FICHIER : descripteurs.py
ROLE    : Extraction des points caractéristiques faciaux et construction
          du vecteur de descripteurs géométriques 30 dimensions (30D).

THÉORIE : DESCRIPTEURS GÉOMÉTRIQUES FACIAUX
--------------------------------------------
L'approche géométrique représente un visage par un ensemble de DISTANCES
entre des points anatomiques clés (landmarks) plutôt que par les pixels bruts.

Avantages :
  - Invariant à la luminosité (on ne regarde pas les intensités de pixels)
  - Compact (30 valeurs vs 128×128 = 16384 pixels)
  - Interprétable (chaque dimension a une signification anatomique)

Landmarks utilisés (9 points sur un visage 128×128) :
  1. Œil gauche (centre)
  2. Œil droit  (centre)
  3. Nez        (pointe)
  4. Bouche gauche (commissure)
  5. Bouche droite (commissure)
  6. Menton     (bas du visage)
  7. Front      (haut du visage)
  8. Tempe gauche
  9. Tempe droite

Distances calculées :
  On calcule toutes les C(9,2) = 36 paires possibles.
  On en garde 25 (les plus discriminantes selon la littérature).
  Ces 25 distances + 5 features Snake = vecteur 30D.

Normalisation :
  Toutes les distances sont divisées par la DISTANCE INTER-OCULAIRE
  (écart entre les deux yeux). Cela rend le vecteur invariant à l'échelle
  (une même personne loin ou proche de la caméra donnera le même vecteur).

NOTE SUR LA LOCALISATION DES LANDMARKS
---------------------------------------
Sans deep learning, on utilise des positions ANATOMIQUES RELATIVES fixes
sur le visage normalisé 128×128. Ces positions sont définies comme fractions
de la taille de l'image, basées sur les proportions statistiques moyennes
d'un visage humain (règle des tiers, etc.).

Une amélioration possible : utiliser dlib ou mediapipe pour des landmarks
précis (68 points), mais cela sort du cadre "sans apprentissage" du projet.
==============================================================================
"""

import cv2
import numpy as np


def _indices_zigzag(taille=8, nb_coeffs=25):
    """Retourne les premiers coefficients basse frequence en parcours zigzag."""
    indices = []
    for somme in range(taille * 2 - 1):
        lignes = range(somme + 1)
        if somme % 2 == 0:
            lignes = reversed(list(lignes))

        for i in lignes:
            j = somme - i
            if i >= taille or j >= taille:
                continue
            if i == 0 and j == 0:
                continue
            indices.append((i, j))
            if len(indices) == nb_coeffs:
                return indices
    return indices


_DCT_INDICES = _indices_zigzag(taille=8, nb_coeffs=25)


def _features_dct(visage_128, nb_features=25):
    """
    Extrait des coefficients DCT basse frequence depuis le visage reel.

    L'ancienne version utilisait surtout des landmarks fixes, donc le vecteur
    changeait tres peu d'une personne a l'autre. Ces coefficients capturent
    directement la structure de l'image normalisee.
    """
    if visage_128 is None or visage_128.size == 0:
        return np.zeros(nb_features, dtype=np.float64)

    if len(visage_128.shape) == 3:
        gris = cv2.cvtColor(visage_128, cv2.COLOR_BGR2GRAY)
    else:
        gris = visage_128

    if gris.dtype != np.uint8:
        gris = np.clip(gris, 0, 255).astype(np.uint8)

    petit = cv2.resize(gris, (32, 32), interpolation=cv2.INTER_AREA)
    petit = cv2.equalizeHist(petit)
    img = petit.astype(np.float32) / 255.0
    img = img - float(img.mean())

    ecart_type = float(img.std())
    if ecart_type < 1e-6:
        return np.zeros(nb_features, dtype=np.float64)
    img = img / ecart_type

    coeffs = cv2.dct(img)
    valeurs = np.array(
        [coeffs[i, j] for i, j in _DCT_INDICES[:nb_features]],
        dtype=np.float64
    )

    valeurs = np.tanh(valeurs / 4.0)
    norme = np.linalg.norm(valeurs)
    if norme > 1e-8:
        valeurs = valeurs / norme
    return valeurs

from snake import extraire_features_contour   # Features complémentaires du Snake


# ==============================================================================
# DÉFINITION DES LANDMARKS ANATOMIQUES
# Positions en fractions de la taille 128×128.
# Basées sur les proportions standard d'un visage humain (de face).
#
# Repère : (0,0) = coin haut-gauche de l'image 128×128
# ==============================================================================

# Dictionnaire : nom du landmark → (fraction_x, fraction_y)
# fraction_x : position horizontale (0=gauche, 1=droite)
# fraction_y : position verticale   (0=haut,   1=bas)
LANDMARKS_RELATIFS = {
    'oeil_gauche'   : (0.33, 0.38),   # Œil gauche — tiers gauche, 38% de hauteur
    'oeil_droit'    : (0.67, 0.38),   # Œil droit  — tiers droit, même hauteur
    'nez'           : (0.50, 0.57),   # Nez — centré, un peu en dessous des yeux
    'bouche_gauche' : (0.36, 0.72),   # Commissure gauche de la bouche
    'bouche_droite' : (0.64, 0.72),   # Commissure droite
    'bouche_centre' : (0.50, 0.75),   # Centre de la lèvre inférieure
    'menton'        : (0.50, 0.90),   # Bas du menton
    'front'         : (0.50, 0.15),   # Milieu du front
    'tempe_gauche'  : (0.10, 0.42),   # Tempe gauche
    'tempe_droite'  : (0.90, 0.42),   # Tempe droite
}


def _features_contour_stables(contour, nb_features=5):
    """Centre et reduit les mesures du Snake pour les rendre comparables."""
    brutes = extraire_features_contour(contour, nb_features=nb_features)
    if len(brutes) < nb_features:
        brutes = np.pad(brutes, (0, nb_features - len(brutes)))

    largeur, hauteur, rapport, perimetre, echantillon = brutes[:5]
    features = np.array([
        (largeur - 0.75) * 3.0,
        (hauteur - 0.90) * 3.0,
        (rapport - 1.15) * 1.5,
        (perimetre - 2.60) * 0.5,
        echantillon * 0.75,
    ], dtype=np.float64)
    return np.clip(features, -1.0, 1.0)


def calculer_landmarks(taille=128):
    """
    Convertit les positions relatives en coordonnées pixel absolues.

    Paramètre
    ---------
    taille : int — côté de l'image carrée (défaut 128)

    Retourne
    --------
    dict : nom → (x_pixel, y_pixel)
    """
    landmarks = {}
    for nom, (fx, fy) in LANDMARKS_RELATIFS.items():
        landmarks[nom] = (int(fx * taille), int(fy * taille))
    return landmarks


def distance_euclidienne(pt1, pt2):
    """
    Calcule la distance euclidienne entre deux points 2D.

    Paramètres
    ----------
    pt1, pt2 : tuples (x, y)

    Retourne
    --------
    float : distance en pixels
    """
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def calculer_vecteur_30D(visage_128, contour=None):
    """
    Calcule le vecteur caractéristique 30D d'un visage.

    Ce vecteur est la "signature biométrique géométrique" de la personne.
    Il sera stocké dans dataset.csv et utilisé pour l'identification.

    Paramètres
    ----------
    visage_128 : numpy array uint8 shape (128, 128) — visage normalisé
    contour    : numpy array (N, 2) — contour Snake (optionnel)
                 Si None, les features Snake seront remplacées par des zéros.

    Retourne
    --------
    numpy array float64 shape (30,) — vecteur 30 dimensions normalisé

    Structure du vecteur
    --------------------
    Indices 0-24  : 25 distances inter-landmarks normalisées
    Indices 25-29 : 5 features extraites du contour Snake
    """

    # Nouveau descripteur: 25 coefficients DCT du visage reel + 5 mesures Snake.
    # Il remplace les anciennes distances entre landmarks fixes, qui etaient
    # identiques pour toutes les personnes et rendaient l'identification instable.
    features_image = _features_dct(visage_128, nb_features=25)
    features_snake = _features_contour_stables(contour, nb_features=5)

    vecteur = np.concatenate([features_image, features_snake]).astype(np.float64)
    norme = np.linalg.norm(vecteur)
    if norme > 1e-8:
        vecteur = vecteur / norme

    assert len(vecteur) == 30, f"Vecteur de taille incorrecte : {len(vecteur)} != 30"
    return vecteur

    # --- Calculer les positions pixel des landmarks ---
    landmarks = calculer_landmarks(taille=visage_128.shape[0])

    # Liste ordonnée des noms (pour garantir un ordre constant des features)
    noms = list(landmarks.keys())    # 10 landmarks
    points = [landmarks[n] for n in noms]

    # --- Calculer toutes les distances inter-paires ---
    # C(10, 2) = 45 paires possibles
    # On en sélectionne 25 : les paires les plus discriminantes anatomiquement
    paires_selectionnees = [
        # --- Distances liées aux yeux ---
        ('oeil_gauche',    'oeil_droit'),       # Écartement inter-oculaire (référence)
        ('oeil_gauche',    'nez'),              # Œil gauche → nez
        ('oeil_droit',     'nez'),              # Œil droit → nez
        ('oeil_gauche',    'bouche_gauche'),    # Œil → commissure homologue
        ('oeil_droit',     'bouche_droite'),
        ('oeil_gauche',    'menton'),           # Œil → menton
        ('oeil_droit',     'menton'),
        ('oeil_gauche',    'front'),            # Œil → front
        ('oeil_droit',     'front'),
        # --- Distances liées au nez ---
        ('nez',            'bouche_gauche'),    # Nez → commissure
        ('nez',            'bouche_droite'),
        ('nez',            'bouche_centre'),    # Nez → centre bouche
        ('nez',            'menton'),           # Nez → menton
        ('nez',            'front'),            # Hauteur nez (front → menton)
        # --- Distances liées à la bouche ---
        ('bouche_gauche',  'bouche_droite'),    # Largeur de la bouche
        ('bouche_gauche',  'bouche_centre'),
        ('bouche_droite',  'bouche_centre'),
        ('bouche_centre',  'menton'),           # Bouche → menton
        # --- Distances latérales (largeur du visage) ---
        ('tempe_gauche',   'tempe_droite'),     # Largeur totale du visage
        ('tempe_gauche',   'oeil_gauche'),      # Tempe → œil
        ('tempe_droite',   'oeil_droit'),
        # --- Hauteurs verticales ---
        ('front',          'menton'),           # Hauteur totale du visage
        ('front',          'nez'),              # Front → nez
        ('nez',            'menton'),           # Nez → menton (déjà calculé, variante)
        ('oeil_gauche',    'bouche_centre'),    # Diagonale œil → bouche
    ]

    # Calculer les 25 distances
    distances_brutes = []
    for nom1, nom2 in paires_selectionnees:
        pt1 = landmarks[nom1]
        pt2 = landmarks[nom2]
        d = distance_euclidienne(pt1, pt2)
        distances_brutes.append(d)

    # --- Normalisation par la distance inter-oculaire ---
    # La distance inter-oculaire est la distance entre l'œil gauche et l'œil droit.
    # C'est la référence de normalisation car :
    #   - Elle est robuste (les yeux sont toujours présents et bien détectés)
    #   - Elle varie peu entre différentes poses
    #   - Elle permet de comparer des visages à différentes distances de la caméra
    dist_inter_oculaire = distances_brutes[0]  # Premier élément = paire yeux

    if dist_inter_oculaire < 1e-8:
        # Protection contre une division par zéro (ne devrait pas arriver
        # si l'alignement est correct)
        dist_inter_oculaire = 1.0

    distances_normalisees = [d / dist_inter_oculaire for d in distances_brutes]

    # --- Ajouter les features du contour Snake ---
    # Le Snake apporte des informations sur la FORME de l'ovale facial
    # (largeur au niveau des joues, hauteur du front, etc.)
    if contour is not None:
        features_snake = extraire_features_contour(contour, nb_features=5)
    else:
        features_snake = np.zeros(5)

    # --- Assembler le vecteur final 30D ---
    # 25 distances normalisées + 5 features Snake = 30 dimensions
    vecteur = np.array(distances_normalisees + list(features_snake),
                       dtype=np.float64)

    # Vérification de sécurité : s'assurer qu'on a exactement 30 valeurs
    assert len(vecteur) == 30, f"Vecteur de taille incorrecte : {len(vecteur)} ≠ 30"

    return vecteur


def visualiser_landmarks(visage_128):
    """
    Affiche le visage avec les landmarks dessinés dessus (débogage).

    Paramètre
    ---------
    visage_128 : image 128×128 niveaux de gris

    Affiche une fenêtre OpenCV avec les points et leurs noms.
    """
    import cv2

    # Agrandir pour mieux voir (128 → 384)
    scale = 3
    img = cv2.cvtColor(visage_128, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (128*scale, 128*scale))

    landmarks = calculer_landmarks(128)

    couleurs = {
        'oeil_gauche': (0, 255, 255),
        'oeil_droit':  (0, 255, 255),
        'nez':         (255, 100, 0),
        'bouche_gauche': (0, 100, 255),
        'bouche_droite': (0, 100, 255),
        'bouche_centre': (0, 100, 255),
        'menton':       (100, 255, 0),
        'front':        (255, 255, 0),
        'tempe_gauche': (200, 0, 200),
        'tempe_droite': (200, 0, 200),
    }

    for nom, (px, py) in landmarks.items():
        # Mettre à l'échelle
        px_s, py_s = px * scale, py * scale
        couleur = couleurs.get(nom, (255, 255, 255))

        # Cercle au landmark
        cv2.circle(img, (px_s, py_s), 6, couleur, -1)

        # Texte (nom abrégé)
        abr = nom[:4]
        cv2.putText(img, abr, (px_s+7, py_s+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, couleur, 1)

    cv2.imshow("Landmarks faciaux", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Landmarks faciaux")


def comparer_vecteurs(v1, v2):
    """
    Compare deux vecteurs 30D et retourne plusieurs métriques de distance.

    Paramètres
    ----------
    v1, v2 : numpy arrays 30D

    Retourne
    --------
    dict avec :
      'euclidienne' : distance L2 (utilisée pour l'identification)
      'manhattan'   : distance L1
      'cosinus'     : similarité cosinus (1 = identiques, 0 = orthogonaux)
      'chebyshev'   : distance L-infini (max des différences absolues)
    """
    diff = v1 - v2
    return {
        'euclidienne': np.linalg.norm(diff),
        'manhattan'  : np.sum(np.abs(diff)),
        'cosinus'    : np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8),
        'chebyshev'  : np.max(np.abs(diff)),
    }
