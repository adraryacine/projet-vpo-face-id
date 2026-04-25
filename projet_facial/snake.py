"""
==============================================================================
FICHIER : snake.py
ROLE    : Implémentation du modèle de contour actif (Snake / Serpent actif)
          pour extraire le contour précis de l'ovale facial.

THÉORIE DU SNAKE (Kass, Witkin, Terzopoulos — 1988)
-----------------------------------------------------
Un Snake est une courbe paramétrique C(s) = [x(s), y(s)] qui se déplace
dans l'image pour minimiser une énergie totale :

    E_snake = E_interne + E_externe

E_interne contrôle la forme du contour :
  - Terme élasticité (α) : s'oppose à l'allongement → la courbe reste lisse
  - Terme rigidité  (β) : s'oppose aux courbures → évite les angles vifs

E_externe attire le contour vers les bords de l'image (gradients) :
  - Calculé à partir du gradient de l'image (zones de fort contraste)
  - Le Snake "glisse" vers les contours détectés

Algorithme d'optimisation :
  La minimisation est faite itérativement via l'algorithme de Williams & Shah
  ou par descente de gradient. scikit-image implémente une version efficace.

APPLICATION DANS CE PROJET
---------------------------
- Initialisation : ellipse centrée sur le visage 128×128
- Le Snake converge vers le contour du visage (front, joues, menton)
- Le contour final (200 points) encode la FORME géométrique du visage
- Ces points contribuent au vecteur de descripteurs 30D

PARAMÈTRES IMPORTANTS (à ajuster empiriquement)
------------------------------------------------
alpha : élasticité   — plus élevé = courbe plus courte/serrée
beta  : rigidité     — plus élevé = courbe plus lisse (moins d'angles)
gamma : pas de temps — taille des pas d'optimisation
==============================================================================
"""

import numpy as np
from skimage.segmentation import active_contour   # Algorithme Snake de scikit-image
from skimage.filters import gaussian              # Flou gaussien (lissage de l'image)
import cv2                                        # Pour visualisation éventuelle


# ==============================================================================
# PARAMÈTRES PAR DÉFAUT DU SNAKE
# Ces valeurs ont été choisies pour un visage normalisé 128×128.
# Vous pouvez les modifier et observer l'effet sur la convergence.
# ==============================================================================
ALPHA         = 0.015   # Élasticité : résistance à l'étirement
BETA          = 0.10    # Rigidité   : résistance aux courbures
GAMMA         = 0.001   # Pas de temps de l'optimisation
SIGMA_LISSAGE = 3.0     # Écart-type du flou gaussien avant Snake
NB_POINTS     = 200     # Nombre de points sur le contour initial (et final)
NB_ITERATIONS = 500     # Nombre d'itérations de l'optimisation


def _creer_ellipse_initiale(cx, cy, rx, ry, nb_points=NB_POINTS):
    """
    Crée le contour initial en forme d'ellipse.

    Le Snake a besoin d'une position de départ proche du vrai contour
    pour converger correctement. On l'initialise comme une ellipse
    centrée sur le visage.

    Paramètres
    ----------
    cx, cy    : centre de l'ellipse (en pixels)
    rx, ry    : demi-axes horizontal et vertical (en pixels)
    nb_points : nombre de points discrets sur l'ellipse

    Retourne
    --------
    numpy array shape (nb_points, 2) — coordonnées (row=y, col=x)
    Note : scikit-image attend (row, col) et non (x, y) !
    """
    # Paramétrage de l'ellipse : t varie de 0 à 2π
    t = np.linspace(0, 2 * np.pi, nb_points, endpoint=False)

    # Coordonnées x et y de l'ellipse
    x_ellipse = cx + rx * np.cos(t)
    y_ellipse = cy + ry * np.sin(t)

    # scikit-image utilise la convention (row, col) = (y, x)
    init = np.column_stack([y_ellipse, x_ellipse])

    return init


def appliquer_snake(visage_128, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """
    Applique le modèle de contour actif Snake sur le visage normalisé.

    Paramètres
    ----------
    visage_128 : numpy array uint8 shape (128, 128) — visage en niveaux de gris
    alpha      : coefficient d'élasticité
    beta       : coefficient de rigidité
    gamma      : pas de temps de l'optimisation

    Retourne
    --------
    contour : numpy array float64 shape (NB_POINTS, 2)
              Coordonnées (row, col) des points du contour final.
              Représente l'ovale facial détecté.
    None    : si une erreur survient (image invalide, etc.)
    """
    try:
        # --- Étape 1 : Normalisation [0, 1] ---
        # active_contour de scikit-image attend des valeurs flottantes dans [0, 1]
        img_float = visage_128.astype(np.float64) / 255.0

        # --- Étape 2 : Lissage gaussien ---
        # On lisse l'image AVANT de donner les gradients au Snake.
        # Pourquoi ? Sans lissage, le gradient est bruité et le Snake
        # peut se coincer dans des détails (pores, poils, reflets).
        # Le flou gaussien "nettoie" les petites variations locales,
        # ne conservant que les contours principaux (bords du visage).
        img_lissee = gaussian(img_float, sigma=SIGMA_LISSAGE)

        # --- Étape 3 : Définir le contour initial (ellipse) ---
        # Centre : milieu de l'image 128×128 → (64, 64)
        # Rayons : rx=50 pixels (horizontal), ry=58 pixels (vertical)
        # → légèrement plus haut que large (forme ovale d'un visage)
        cx, cy = 64, 64    # Centre de l'image
        rx     = 50        # Rayon horizontal (largeur du visage ≈ 100px)
        ry     = 58        # Rayon vertical   (hauteur du visage ≈ 116px)

        init = _creer_ellipse_initiale(cx, cy, rx, ry, NB_POINTS)

        # --- Étape 4 : Optimisation du Snake ---
        # active_contour minimise l'énergie totale du Snake.
        # L'image lissée est utilisée pour calculer les forces externes
        # (les zones de fort gradient attirent le contour).
        #
        # max_num_iter : nombre maximum d'itérations.
        # convergence  : seuil de déplacement en dessous duquel on arrête.
        contour = active_contour(
            img_lissee,             # Image source (lissée)
            init,                   # Contour initial (ellipse)
            alpha=alpha,            # Élasticité
            beta=beta,              # Rigidité
            gamma=gamma,            # Pas de temps
            max_num_iter=NB_ITERATIONS,
            boundary_condition='periodic'   # Contour fermé (tête de serpent = queue)
        )

        # contour shape : (NB_POINTS, 2) — (row, col) pour chaque point
        return contour

    except Exception as e:
        print(f"[SNAKE] Erreur lors de l'optimisation : {e}")
        # En cas d'erreur, retourner l'ellipse initiale non optimisée
        # pour ne pas bloquer le pipeline
        cx, cy, rx, ry = 64, 64, 50, 58
        return _creer_ellipse_initiale(cx, cy, rx, ry, NB_POINTS)


def extraire_features_contour(contour, nb_features=10):
    """
    Extrait des mesures synthétiques depuis le contour Snake.
    Ces mesures enrichissent le vecteur 30D avec des informations de forme.

    Mesures extraites :
    - Largeur et hauteur du contour (bounding box)
    - Périmètre approximatif (somme des distances entre points consécutifs)
    - Rapport d'aspect (largeur / hauteur)
    - Positions normalisées de points échantillonnés sur le contour

    Paramètres
    ----------
    contour     : numpy array (N, 2) — contour Snake (row, col)
    nb_features : nombre de valeurs à extraire

    Retourne
    --------
    numpy array float64 shape (nb_features,)
    """
    if contour is None or len(contour) == 0:
        return np.zeros(nb_features)

    # Extraire colonnes (col = x, row = y)
    rows = contour[:, 0]   # coordonnées Y (verticales)
    cols = contour[:, 1]   # coordonnées X (horizontales)

    features = []

    # --- Feature 1 : Largeur normalisée du contour ---
    # (max_x - min_x) / 128 → valeur entre 0 et 1
    largeur = (cols.max() - cols.min()) / 128.0
    features.append(largeur)

    # --- Feature 2 : Hauteur normalisée du contour ---
    hauteur = (rows.max() - rows.min()) / 128.0
    features.append(hauteur)

    # --- Feature 3 : Rapport d'aspect ---
    # Un visage rond ≈ 1.0, un visage allongé > 1.0
    rapport = hauteur / (largeur + 1e-8)
    features.append(rapport)

    # --- Feature 4 : Périmètre normalisé ---
    # Somme des distances euclidiennes entre points consécutifs
    diffs = np.diff(contour, axis=0)
    perimetre = np.sum(np.sqrt(np.sum(diffs**2, axis=1))) / 128.0
    features.append(perimetre)

    # --- Features 5 à nb_features : points échantillonnés sur le contour ---
    # On échantillonne des points régulièrement répartis sur le contour
    # et on normalise leurs coordonnées par 128
    nb_pts_supplementaires = nb_features - 4
    if nb_pts_supplementaires > 0:
        indices = np.linspace(0, len(contour)-1,
                              nb_pts_supplementaires, dtype=int)
        for idx in indices:
            # Normalisation : (coordonnée - 64) / 64 → entre -1 et 1
            features.append((contour[idx, 1] - 64) / 64.0)   # coord X

    return np.array(features[:nb_features], dtype=np.float64)


def visualiser_snake(visage_128, contour, titre="Snake — Contour Actif"):
    """
    Affiche le visage avec le contour Snake superposé (utilisation : débogage).

    Paramètres
    ----------
    visage_128 : image 128×128 niveaux de gris
    contour    : array (N, 2) — contour Snake
    titre      : titre de la fenêtre OpenCV
    """
    # Convertir en BGR pour pouvoir dessiner en couleur
    img_couleur = cv2.cvtColor(visage_128, cv2.COLOR_GRAY2BGR)
    img_grande  = cv2.resize(img_couleur, (384, 384))  # Agrandir pour voir

    if contour is not None:
        # Mettre à l'échelle (128 → 384 : facteur 3)
        echelle = 384 / 128.0
        pts = contour[:, [1, 0]] * echelle  # Convertir (row,col) → (x,y) et scaler
        pts = pts.astype(np.int32).reshape((-1, 1, 2))

        # Dessiner le contour en vert (fermé = True)
        cv2.polylines(img_grande, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=2)

        # Marquer quelques points clés en rouge
        for i in range(0, len(contour), len(contour)//20):
            px = int(contour[i, 1] * echelle)
            py = int(contour[i, 0] * echelle)
            cv2.circle(img_grande, (px, py), 3, (0, 0, 255), -1)

    cv2.imshow(titre, img_grande)
    cv2.waitKey(0)
    cv2.destroyWindow(titre)
