"""
==============================================================================
FICHIER : capture.py
ROLE    : Gestion du dataset CSV.
          - Enregistrement d'une nouvelle personne (Mode E)
          - Chargement du dataset complet depuis le fichier CSV
          - Construction automatique du dataset par capture webcam en lot

FORMAT DU DATASET (dataset.csv)
---------------------------------
Chaque ligne = un enregistrement d'une personne à un instant donné.
Format : nom;v0;v1;v2;...;v29
  - Séparateur : point-virgule (;) — évite les conflits avec les virgules décimales
  - Une personne peut avoir PLUSIEURS lignes (20 enregistrements requis)
  - Aucune image n'est stockée : seulement le vecteur 30D

Exemple de ligne dans dataset.csv :
  Alice;1.0;0.85;1.23;0.67;...;0.42

Pourquoi stocker des vecteurs et pas des images ?
  - Fichier très léger (quelques Ko vs plusieurs Mo pour des images)
  - Recherche ultra-rapide (on ne calcule que des distances vectorielles)
  - Facilement transportable et versionnable (fichier texte)
==============================================================================
"""

import csv
import os
import numpy as np
import cv2
import time

from detection    import detecter_et_aligner
from snake        import appliquer_snake
from descripteurs import calculer_vecteur_30D


# ==============================================================================
# CONSTANTES
# ==============================================================================
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
DIMENSION    = 30                  # Taille du vecteur (doit correspondre à descripteurs.py)


# ==============================================================================
# OPÉRATIONS SUR LE FICHIER CSV
# ==============================================================================

def enregistrer_personne(nom, vecteur_30D, chemin=DATASET_PATH):
    """
    Ajoute une ligne dans le fichier CSV pour une personne.

    Si le fichier n'existe pas encore, il est créé automatiquement.
    Chaque appel AJOUTE une ligne (mode 'a' = append), donc plusieurs
    enregistrements pour la même personne sont possibles et souhaitables
    (robustesse : 20 enregistrements par personne recommandés).

    Paramètres
    ----------
    nom        : str — identifiant de la personne (ex: "Alice", "Mehdi")
    vecteur_30D: numpy array float64 shape (30,)
    chemin     : str — chemin du fichier CSV (défaut : dataset.csv)
    """
    if len(vecteur_30D) != DIMENSION:
        raise ValueError(
            f"Vecteur de mauvaise taille : {len(vecteur_30D)} au lieu de {DIMENSION}"
        )

    with open(chemin, 'a', newline='', encoding='utf-8') as fichier:
        writer = csv.writer(fichier, delimiter=';')
        # Une ligne = nom + 30 valeurs flottantes arrondies à 6 décimales
        ligne = [nom] + [f"{v:.6f}" for v in vecteur_30D]
        writer.writerow(ligne)

    print(f"[CSV] Enregistrement ajouté pour '{nom}' → {chemin}")


def charger_dataset(chemin=DATASET_PATH):
    """
    Charge l'intégralité du dataset depuis le fichier CSV.

    Gère les cas : fichier inexistant, lignes corrompues, vecteurs incomplets.

    Paramètre
    ---------
    chemin : str — chemin du fichier CSV

    Retourne
    --------
    personnes : list[str]             — liste des noms (une entrée par ligne CSV)
    vecteurs  : list[numpy array 30D] — vecteurs correspondants

    Note : L'index i est cohérent : personnes[i] ↔ vecteurs[i]
    """
    personnes = []
    vecteurs  = []

    if not os.path.exists(chemin):
        print(f"[CSV] Fichier '{chemin}' introuvable. Dataset vide.")
        return personnes, vecteurs

    lignes_invalides = 0

    with open(chemin, 'r', encoding='utf-8') as fichier:
        reader = csv.reader(fichier, delimiter=';')
        for num_ligne, ligne in enumerate(reader, start=1):
            # Vérification : la ligne doit avoir exactement 1 + 30 colonnes
            if len(ligne) != DIMENSION + 1:
                print(f"[CSV] Ligne {num_ligne} ignorée "
                      f"(taille={len(ligne)}, attendu={DIMENSION+1})")
                lignes_invalides += 1
                continue

            try:
                nom     = ligne[0].strip()
                vecteur = np.array(ligne[1:], dtype=np.float64)
                personnes.append(nom)
                vecteurs.append(vecteur)
            except ValueError as e:
                print(f"[CSV] Ligne {num_ligne} invalide : {e}")
                lignes_invalides += 1

    print(f"[CSV] Chargé : {len(personnes)} enregistrements, "
          f"{len(set(personnes))} personnes, "
          f"{lignes_invalides} lignes ignorées.")

    if vecteurs:
        normes = np.linalg.norm(np.vstack(vecteurs), axis=1)
        indices_valides = [i for i, norme in enumerate(normes) if norme <= 2.0]
        nb_anciens = len(vecteurs) - len(indices_valides)
        if nb_anciens > 0:
            print(f"[CSV] ATTENTION: {nb_anciens} anciens vecteurs ignores. "
                  "Refaites l'enregistrement avec E.")
            personnes = [personnes[i] for i in indices_valides]
            vecteurs = [vecteurs[i] for i in indices_valides]

    return personnes, vecteurs


def lister_personnes(chemin=DATASET_PATH):
    """
    Retourne la liste des personnes uniques dans le dataset,
    avec leur nombre d'enregistrements.

    Retourne
    --------
    dict : {nom: nb_enregistrements}
    """
    personnes, _ = charger_dataset(chemin)
    stats = {}
    for nom in personnes:
        stats[nom] = stats.get(nom, 0) + 1
    return stats


def supprimer_personne(nom_a_supprimer, chemin=DATASET_PATH):
    """
    Supprime tous les enregistrements d'une personne du dataset.

    Procédure : relire tout le CSV, filtrer les lignes du nom voulu,
    réécrire le fichier filtré. (Opération de maintenance rare.)

    Paramètres
    ----------
    nom_a_supprimer : str
    chemin          : str
    """
    personnes, vecteurs = charger_dataset(chemin)
    avant = len(personnes)

    # Filtrer
    paires_filtrees = [(n, v) for n, v in zip(personnes, vecteurs)
                       if n != nom_a_supprimer]

    # Réécrire le fichier
    with open(chemin, 'w', newline='', encoding='utf-8') as fichier:
        writer = csv.writer(fichier, delimiter=';')
        for nom, vecteur in paires_filtrees:
            ligne = [nom] + [f"{v:.6f}" for v in vecteur]
            writer.writerow(ligne)

    apres = len(paires_filtrees)
    print(f"[CSV] Supprimé {avant - apres} enregistrements de '{nom_a_supprimer}'.")


# ==============================================================================
# CAPTURE EN LOT (DATASET BUILDER)
# ==============================================================================

def capturer_dataset_webcam(nom, nb_images=20, delai_entre_captures=0.5):
    """
    Capture automatiquement plusieurs images d'une personne via la webcam
    et enregistre chaque vecteur dans le dataset CSV.

    PROTOCOLE DE CAPTURE (20 images selon les specs du projet) :
    - Images 1-5  : face neutre (immobile, expression naturelle)
    - Images 6-10 : avec expressions (sourire, surprise, yeux plissés, ...)
    - Images 11-15: légère rotation latérale (tourner la tête ±15°)
    - Images 16-20: éclairage différent (lumière gauche, droite, contre-jour)

    Paramètres
    ----------
    nom                 : str — nom de la personne à enregistrer
    nb_images           : int — nombre total d'images à capturer (défaut 20)
    delai_entre_captures: float — secondes entre deux captures auto (défaut 0.5s)

    Note : une capture se déclenche automatiquement toutes les `delai` secondes.
           L'utilisateur doit changer de pose entre les captures.
    """
    print(f"\n{'='*60}")
    print(f"  CAPTURE DATASET — Personne : {nom}")
    print(f"  {nb_images} images seront capturées automatiquement")
    print(f"{'='*60}")
    print("\nINSTRUCTIONS :")
    print("  Images  1- 5 : restez immobile, expression neutre")
    print("  Images  6-10 : changez d'expression (sourire, surprise...)")
    print("  Images 11-15 : tournez légèrement la tête (±15°)")
    print("  Images 16-20 : changez l'éclairage (lampe gauche, droite...)")
    print("\nAppuyez sur ENTRÉE pour commencer...")
    input()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Webcam introuvable.")
        return

    compteur     = 0        # Nombre d'images valides capturées
    nb_tentatives = 0       # Nombre total de frames lues
    dernier_temps = 0       # Timestamp de la dernière capture réussie

    print(f"\n[GO] Démarrage de la capture pour '{nom}'...")
    print("Appuyez sur Q pour arrêter avant la fin.\n")

    while compteur < nb_images:
        ret, frame = cap.read()
        if not ret:
            continue

        nb_tentatives += 1
        temps_actuel = time.time()

        # --- Affichage de la progression ---
        frame_affichage = frame.copy()
        cv2.putText(frame_affichage,
                    f"{nom} — {compteur}/{nb_images} images capturees",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Afficher la phase en cours
        if compteur < 5:
            phase = "Phase 1/4 : Expression neutre"
        elif compteur < 10:
            phase = "Phase 2/4 : Changez d'expression !"
        elif compteur < 15:
            phase = "Phase 3/4 : Tournez legerement la tete"
        else:
            phase = "Phase 4/4 : Changez l'eclairage"
        cv2.putText(frame_affichage, phase,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        cv2.imshow(f"Capture dataset — {nom}", frame_affichage)

        # --- Capture automatique avec délai ---
        if temps_actuel - dernier_temps >= delai_entre_captures:
            # Tenter d'extraire le vecteur
            visage = detecter_et_aligner(frame)

            if visage is None:
                cv2.putText(frame_affichage, "!!! Aucun visage detecte !!!",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow(f"Capture dataset — {nom}", frame_affichage)
            else:
                contour = appliquer_snake(visage)
                vecteur = calculer_vecteur_30D(visage, contour)
                enregistrer_personne(nom, vecteur)
                compteur += 1
                dernier_temps = temps_actuel
                print(f"  [{compteur:02d}/{nb_images}] ✓ Image capturée — {phase}")

                # Flash vert pour signaler la capture
                frame_flash = frame_affichage.copy()
                cv2.rectangle(frame_flash, (0,0),
                              (frame.shape[1], frame.shape[0]),
                              (0, 255, 0), 8)
                cv2.imshow(f"Capture dataset — {nom}", frame_flash)

        # Quitter avec Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[ARRÊT] Capture interrompue par l'utilisateur.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[TERMINÉ] {compteur} images enregistrées pour '{nom}'.")
    if compteur < nb_images:
        print(f"  ⚠ Objectif non atteint ({nb_images} requis). "
              f"Relancez la capture pour compléter.")
