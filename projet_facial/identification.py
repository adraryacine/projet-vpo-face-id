"""
==============================================================================
FICHIER : identification.py
ROLE    : Identification d'un visage inconnu par recherche du plus proche
          voisin (1-NN = 1-Nearest Neighbor) dans le dataset.

ALGORITHME : PLUS PROCHE VOISIN (1-NN)
----------------------------------------
Principe :
  Pour identifier une personne inconnue (vecteur V_inconnu),
  on calcule la distance entre V_inconnu et TOUS les vecteurs du dataset.
  Le vecteur le plus proche → c'est la personne identifiée.

  En pseudocode :
    distances = [euclidienne(V_inconnu, V_i) for V_i in dataset]
    idx_min   = argmin(distances)
    identifie = personnes[idx_min]

Pourquoi l'Euclidienne ?
  La distance euclidienne L2 entre deux vecteurs mesure leur "similarité"
  géométrique. Elle est simple, rapide, et très efficace pour des vecteurs
  de descripteurs normalisés comme le nôtre.

  d(u, v) = sqrt( Σ (u_i - v_i)² )

Seuil de décision :
  Si la plus petite distance > seuil → "Inconnu" (personne non enregistrée)
  Ce seuil est un hyperparamètre crucial à calibrer empiriquement.
  Valeur typique pour notre vecteur 30D : entre 0.3 et 0.8

Score de confiance :
  confiance = max(0, (1 - distance/seuil)) × 100
  - distance = 0   → confiance = 100% (vecteur identique)
  - distance = seuil → confiance ≈ 0%
  - distance > seuil → confiance = 0% (Inconnu)

COMPLEXITÉ
----------
  O(N × D) avec N = nb d'enregistrements, D = dimension du vecteur (30)
  Pour N=200 personnes × 20 images = 4000 enregistrements : instantané.
  Pas besoin de KD-Tree ou de structures accélératrices pour ce projet.
==============================================================================
"""

import numpy as np
from collections import Counter   # Pour compter les votes (variante K-NN)


# ==============================================================================
# IDENTIFICATION PRINCIPALE (1-NN)
# ==============================================================================

def identifier(vecteur_inconnu, personnes, vecteurs, seuil=0.5):
    """
    Identifie une personne en cherchant le plus proche voisin dans le dataset.

    Paramètres
    ----------
    vecteur_inconnu : numpy array float64 shape (30,)
                      Vecteur caractéristique de la personne à identifier
    personnes       : list[str]               — noms dans le dataset
    vecteurs        : list[numpy array 30D]   — vecteurs correspondants
    seuil           : float — distance maximale pour accepter une identification
                      Au-delà → "Inconnu"

    Retourne
    --------
    nom_identifie : str   — nom de la personne ou "Inconnu"
    dist_min      : float — distance euclidienne minimale trouvée
    confiance     : float — score en % [0, 100]
    top3          : list  — 3 meilleurs candidats [(nom, dist), ...]

    Lève
    ----
    ValueError si le dataset est vide ou si la dimension ne correspond pas
    """
    if len(personnes) == 0 or len(vecteurs) == 0:
        raise ValueError("Dataset vide — enregistrez des personnes d'abord.")

    # Vérification de la dimension du vecteur
    if len(vecteur_inconnu) != 30:
        raise ValueError(
            f"Vecteur de mauvaise dimension : {len(vecteur_inconnu)} ≠ 30"
        )

    # --- Calcul vectorisé de toutes les distances en une seule opération ---
    # On convertit la liste en matrice numpy pour vectoriser le calcul.
    # matrice_dataset shape : (N, 30) où N = nb d'enregistrements
    matrice_dataset = np.array(vecteurs)   # (N, 30)

    # diff shape : (N, 30) — différence entre l'inconnu et chaque enregistrement
    diff = matrice_dataset - vecteur_inconnu   # Broadcasting automatique

    # distances shape : (N,) — distance euclidienne pour chaque enregistrement
    # np.linalg.norm avec axis=1 calcule la norme L2 de chaque ligne
    distances = np.linalg.norm(diff, axis=1)

    # --- Trouver le top 3 ---
    # argsort retourne les indices qui trieraient le tableau par ordre croissant
    indices_tries = np.argsort(distances)

    # Les 3 premiers sont les plus proches
    top3_indices = indices_tries[:3]
    top3 = [(personnes[i], float(distances[i])) for i in top3_indices]

    # Meilleur candidat
    idx_min   = top3_indices[0]
    dist_min  = float(distances[idx_min])
    meilleur  = personnes[idx_min]

    # --- Calcul du score de confiance ---
    # Formule linéaire : plus la distance est petite, plus la confiance est haute
    # confiance = 100% si distance = 0
    # confiance ≈ 0%  si distance ≈ seuil
    confiance = max(0.0, (1.0 - dist_min / seuil)) * 100.0

    # --- Décision binaire : accepter ou refuser ---
    if dist_min <= seuil:
        nom_final = meilleur
    else:
        nom_final = "Inconnu"
        confiance = 0.0   # Un inconnu a une confiance de 0%

    return nom_final, dist_min, confiance, top3


# ==============================================================================
# VARIANTE K-NN (K PLUS PROCHES VOISINS)
# Plus robuste que 1-NN car utilise un vote majoritaire sur K voisins
# ==============================================================================

def identifier_knn(vecteur_inconnu, personnes, vecteurs, seuil=0.5, k=5):
    """
    Identification par vote majoritaire sur les K plus proches voisins.

    Avantage par rapport à 1-NN :
      Si une personne a 5 voisins et que 4 d'entre eux disent "Alice",
      le résultat est "Alice" même si le 1-NN dit "Bob".
      Plus robuste aux valeurs aberrantes dans le dataset.

    Paramètres
    ----------
    vecteur_inconnu : numpy array 30D
    personnes       : list[str]
    vecteurs        : list[numpy array 30D]
    seuil           : float — seuil de rejet
    k               : int — nombre de voisins à considérer (défaut 5)

    Retourne
    --------
    Même format que identifier() : (nom, dist_min, confiance, top3)
    """
    if len(personnes) == 0:
        raise ValueError("Dataset vide.")

    if not np.all(np.isfinite(vecteur_inconnu)):
        raise ValueError("Vecteur d'entrée invalide (contient NaN ou Inf).")

    k = min(k, len(vecteurs))

    matrice = np.array(vecteurs)

    # Filtrer les vecteurs corrompus (NaN/Inf) du dataset avant le calcul
    masque_valides = np.all(np.isfinite(matrice), axis=1)
    if not np.all(masque_valides):
        nb_invalides = int(np.sum(~masque_valides))
        print(f"[KNN] {nb_invalides} vecteurs invalides ignores dans le dataset.")
        indices_ok = np.where(masque_valides)[0]
        matrice   = matrice[indices_ok]
        personnes = [personnes[i] for i in indices_ok]
        k = min(k, len(personnes))
        if len(personnes) == 0:
            raise ValueError("Aucun vecteur valide dans le dataset.")

    distances = np.linalg.norm(matrice - vecteur_inconnu, axis=1)

    indices_topk = np.argsort(distances)[:k]

    voisins_valides = [(personnes[i], distances[i])
                       for i in indices_topk
                       if distances[i] <= seuil]

    top3_indices = np.argsort(distances)[:3]
    top3 = [(personnes[i], float(distances[i])) for i in top3_indices]

    if not voisins_valides:
        return "Inconnu", float(distances[indices_topk[0]]), 0.0, top3

    # Vote majoritaire avec tie-breaking par distance minimale (évite l'aléatoire)
    votes = Counter(nom for nom, _ in voisins_valides)
    max_votes = votes.most_common(1)[0][1]
    candidats_ex_aequo = [nom for nom, nb in votes.items() if nb == max_votes]

    if len(candidats_ex_aequo) > 1:
        dist_min_par_nom = {}
        for nom, dist in voisins_valides:
            if nom in candidats_ex_aequo:
                if nom not in dist_min_par_nom or dist < dist_min_par_nom[nom]:
                    dist_min_par_nom[nom] = dist
        nom_gagnant = min(candidats_ex_aequo, key=lambda n: dist_min_par_nom[n])
    else:
        nom_gagnant = candidats_ex_aequo[0]

    dist_min = float(distances[indices_topk[0]])

    # Test de marge : rejeter si une autre personne est trop proche du gagnant.
    # Avec beaucoup de personnes, les clusters se rapprochent — une marge de 15%
    # évite les identifications ambiguës (exemple : Alice=0.28, Bob=0.30 → Inconnu).
    dist_min_par_personne = {}
    for i, nom in enumerate(personnes):
        d = float(distances[i])
        if nom not in dist_min_par_personne or d < dist_min_par_personne[nom]:
            dist_min_par_personne[nom] = d

    dist_gagnant = dist_min_par_personne.get(nom_gagnant, dist_min)
    autres_dists = [d for n, d in dist_min_par_personne.items() if n != nom_gagnant]

    if autres_dists and min(autres_dists) < dist_gagnant * 1.15:
        return "Inconnu", dist_min, 0.0, top3

    confiance = max(0.0, (1.0 - dist_min / seuil)) * 100.0
    return nom_gagnant, dist_min, confiance, top3


# ==============================================================================
# CALIBRATION DU SEUIL
# ==============================================================================

def calibrer_seuil(personnes, vecteurs, fraction_test=0.2):
    """
    Détermine empiriquement le seuil optimal sur le dataset.

    Méthode :
      On divise le dataset en train/test (80%/20%).
      On teste différentes valeurs de seuil et on choisit celle qui maximise
      l'accuracy tout en minimisant les faux positifs.

    Paramètres
    ----------
    personnes     : list[str]
    vecteurs      : list[numpy array 30D]
    fraction_test : float — proportion du dataset réservée au test

    Retourne
    --------
    seuil_optimal : float
    rapport       : dict avec les métriques pour chaque seuil testé
    """
    N = len(personnes)
    n_test = max(1, int(N * fraction_test))

    # Mélanger aléatoirement
    indices = np.random.permutation(N)
    idx_test  = indices[:n_test]
    idx_train = indices[n_test:]

    personnes_train = [personnes[i] for i in idx_train]
    vecteurs_train  = [vecteurs[i]  for i in idx_train]
    personnes_test  = [personnes[i] for i in idx_test]
    vecteurs_test   = [vecteurs[i]  for i in idx_test]

    # Tester différents seuils
    seuils_a_tester = np.arange(0.1, 2.0, 0.05)
    rapport = {}
    meilleure_accuracy = -1
    seuil_optimal = 0.5

    print("[CALIBRATION] Test de différents seuils...")

    for seuil in seuils_a_tester:
        corrects = 0
        for nom_reel, vecteur in zip(personnes_test, vecteurs_test):
            try:
                nom_predit, _, _, _ = identifier(
                    vecteur, personnes_train, vecteurs_train, seuil=seuil
                )
                if nom_predit == nom_reel:
                    corrects += 1
            except Exception:
                continue

        accuracy = corrects / n_test if n_test > 0 else 0
        rapport[round(float(seuil), 2)] = accuracy

        if accuracy > meilleure_accuracy:
            meilleure_accuracy = accuracy
            seuil_optimal = float(seuil)

    print(f"[CALIBRATION] Seuil optimal trouvé : {seuil_optimal:.2f} "
          f"(accuracy = {meilleure_accuracy*100:.1f}%)")

    return seuil_optimal, rapport


# ==============================================================================
# ANALYSE DES ERREURS
# ==============================================================================

def analyser_confusions(personnes, vecteurs, seuil=0.5):
    """
    Analyse les cas où le système se trompe (leave-one-out).

    Pour chaque enregistrement :
      - On l'enlève temporairement du dataset
      - On l'identifie avec le reste
      - Si erreur → on note les détails

    Paramètre utilisé pour le débogage et l'amélioration du système.

    Retourne
    --------
    list de dicts, chaque dict décrit une erreur :
      { 'reel', 'predit', 'distance', 'top3' }
    """
    erreurs = []
    N = len(personnes)

    print(f"[ANALYSE] Leave-one-out sur {N} enregistrements...")

    for i in range(N):
        # Dataset sans l'enregistrement i
        p_sans_i = personnes[:i] + personnes[i+1:]
        v_sans_i = vecteurs[:i]  + vecteurs[i+1:]

        if len(p_sans_i) == 0:
            continue

        try:
            nom_predit, dist, conf, top3 = identifier(
                vecteurs[i], p_sans_i, v_sans_i, seuil=seuil
            )
        except Exception:
            continue

        if nom_predit != personnes[i]:
            erreurs.append({
                'reel'     : personnes[i],
                'predit'   : nom_predit,
                'distance' : dist,
                'confiance': conf,
                'top3'     : top3,
            })

    print(f"[ANALYSE] {len(erreurs)} erreurs sur {N} tests "
          f"({len(erreurs)/N*100:.1f}%)")

    # Afficher les confusions les plus fréquentes
    if erreurs:
        print("\n  Confusions les plus fréquentes :")
        pairs = Counter((e['reel'], e['predit']) for e in erreurs)
        for (reel, predit), compte in pairs.most_common(5):
            print(f"    '{reel}' confondu avec '{predit}' : {compte} fois")

    return erreurs
