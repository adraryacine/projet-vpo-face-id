"""
==============================================================================
FICHIER : evaluation.py
ROLE    : Évaluation des performances du système d'identification.
          Calcul de la matrice de confusion, précision, rappel, spécificité,
          F1-score, et analyse des erreurs.

MÉTRIQUES UTILISÉES
--------------------
Supposons C classes (personnes) + la classe "Inconnu".

Pour chaque classe i :
  TP_i (Vrai Positif)  : personne i correctement identifiée comme i
  FP_i (Faux Positif)  : quelqu'un d'autre identifié comme i à tort
  FN_i (Faux Négatif)  : personne i non reconnue (dite "Inconnu" ou autre)
  TN_i (Vrai Négatif)  : quelqu'un d'autre correctement NON identifié comme i

Précision_i  = TP_i / (TP_i + FP_i)   → "Quand on dit i, a-t-on raison ?"
Rappel_i     = TP_i / (TP_i + FN_i)   → "Parmi tous les i, combien retrouve-t-on ?"
Spécificité_i= TN_i / (TN_i + FP_i)   → "Quand ce n'est pas i, le dit-on ?"
F1_i         = 2 × (Prec × Rappel) / (Prec + Rappel)

Accuracy globale = Σ TP_i / N_total

MÉTHODE D'ÉVALUATION : LEAVE-ONE-OUT (LOO)
-------------------------------------------
On ne divise pas le dataset en train/test fixe.
On itère : pour chaque enregistrement, on l'enlève et on teste
l'identification avec le reste. C'est plus robuste avec un petit dataset.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

from identification import identifier


# ==============================================================================
# ÉVALUATION PRINCIPALE
# ==============================================================================

def evaluer_depuis_dataset(personnes, vecteurs, seuil=0.5, methode='loo'):
    """
    Évalue le système complet par leave-one-out et affiche les métriques.

    Paramètres
    ----------
    personnes : list[str]             — noms dans le dataset
    vecteurs  : list[numpy array 30D] — vecteurs correspondants
    seuil     : float — seuil de décision pour l'identification
    methode   : str   — 'loo' (leave-one-out) ou 'split' (80/20)

    Retourne
    --------
    dict : métriques globales et par classe
    """
    print("\n" + "="*60)
    print("  ÉVALUATION DES PERFORMANCES")
    print("="*60)

    if len(personnes) < 2:
        print("[ERREUR] Dataset trop petit pour l'évaluation (minimum 2 enregistrements).")
        return {}

    if methode == 'loo':
        y_reel, y_predit = _evaluation_loo(personnes, vecteurs, seuil)
    else:
        y_reel, y_predit = _evaluation_split(personnes, vecteurs, seuil)

    # Classes présentes (+ "Inconnu" si nécessaire)
    classes = sorted(set(personnes))
    if "Inconnu" in y_predit:
        classes = classes + ["Inconnu"]

    # Afficher les métriques
    metriques = calculer_metriques(y_reel, y_predit, classes)
    afficher_metriques(metriques, classes)

    # Générer la matrice de confusion
    tracer_matrice_confusion(y_reel, y_predit, classes)

    return metriques


def _evaluation_loo(personnes, vecteurs, seuil):
    """
    Évaluation Leave-One-Out.
    Pour chaque enregistrement i : identifier avec le dataset privé de i.

    Retourne (y_reel, y_predit) — deux listes de même longueur.
    """
    y_reel   = []
    y_predit = []
    N = len(personnes)

    print(f"[LOO] Évaluation sur {N} enregistrements...")

    for i in range(N):
        # Dataset sans l'enregistrement i
        p_sans_i = personnes[:i] + personnes[i+1:]
        v_sans_i = vecteurs[:i]  + vecteurs[i+1:]

        try:
            nom_predit, _, _, _ = identifier(vecteurs[i], p_sans_i, v_sans_i, seuil)
        except Exception as e:
            nom_predit = "Erreur"

        y_reel.append(personnes[i])
        y_predit.append(nom_predit)

        # Afficher la progression toutes les 10 itérations
        if (i+1) % 10 == 0:
            print(f"  Progression : {i+1}/{N}")

    return y_reel, y_predit


def _evaluation_split(personnes, vecteurs, seuil, fraction_test=0.2):
    """
    Évaluation train/test split (80% train, 20% test).

    Retourne (y_reel, y_predit).
    """
    N = len(personnes)
    indices = np.random.permutation(N)
    n_test  = max(1, int(N * fraction_test))

    idx_test  = indices[:n_test]
    idx_train = indices[n_test:]

    p_train = [personnes[i] for i in idx_train]
    v_train = [vecteurs[i]  for i in idx_train]

    y_reel, y_predit = [], []
    for i in idx_test:
        try:
            nom_predit, _, _, _ = identifier(vecteurs[i], p_train, v_train, seuil)
        except Exception:
            nom_predit = "Erreur"
        y_reel.append(personnes[i])
        y_predit.append(nom_predit)

    return y_reel, y_predit


# ==============================================================================
# CALCUL DES MÉTRIQUES
# ==============================================================================

def calculer_metriques(y_reel, y_predit, classes):
    """
    Calcule les métriques de performance pour chaque classe et globalement.

    Paramètres
    ----------
    y_reel   : list[str] — étiquettes réelles
    y_predit : list[str] — étiquettes prédites
    classes  : list[str] — liste de toutes les classes

    Retourne
    --------
    dict avec 'global' et par classe : precision, rappel, specificite, f1
    """
    N = len(y_reel)
    assert N == len(y_predit), "y_reel et y_predit doivent avoir la même longueur"

    metriques = {'par_classe': {}, 'global': {}}

    # --- Métriques par classe ---
    for classe in classes:
        TP = sum(1 for r, p in zip(y_reel, y_predit) if r == classe and p == classe)
        FP = sum(1 for r, p in zip(y_reel, y_predit) if r != classe and p == classe)
        FN = sum(1 for r, p in zip(y_reel, y_predit) if r == classe and p != classe)
        TN = sum(1 for r, p in zip(y_reel, y_predit) if r != classe and p != classe)

        precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rappel      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificite = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1          = (2 * precision * rappel / (precision + rappel)
                       if (precision + rappel) > 0 else 0.0)

        metriques['par_classe'][classe] = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision'  : precision,
            'rappel'     : rappel,
            'specificite': specificite,
            'f1'         : f1,
        }

    # --- Métriques globales ---
    nb_corrects = sum(1 for r, p in zip(y_reel, y_predit) if r == p)
    accuracy = nb_corrects / N if N > 0 else 0.0

    # Macro-average (moyenne non pondérée sur les classes)
    classes_sans_inconnu = [c for c in classes if c != "Inconnu"]
    if classes_sans_inconnu:
        macro_precision = np.mean([
            metriques['par_classe'][c]['precision'] for c in classes_sans_inconnu
        ])
        macro_rappel = np.mean([
            metriques['par_classe'][c]['rappel'] for c in classes_sans_inconnu
        ])
        macro_f1 = np.mean([
            metriques['par_classe'][c]['f1'] for c in classes_sans_inconnu
        ])
    else:
        macro_precision = macro_rappel = macro_f1 = 0.0

    nb_inconnus = sum(1 for p in y_predit if p == "Inconnu")
    taux_rejet  = nb_inconnus / N if N > 0 else 0.0

    metriques['global'] = {
        'accuracy'        : accuracy,
        'macro_precision' : macro_precision,
        'macro_rappel'    : macro_rappel,
        'macro_f1'        : macro_f1,
        'taux_rejet'      : taux_rejet,
        'nb_tests'        : N,
        'nb_corrects'     : nb_corrects,
        'nb_erreurs'      : N - nb_corrects,
    }

    return metriques


def afficher_metriques(metriques, classes):
    """Affiche les métriques dans la console de façon lisible."""
    g = metriques['global']

    print(f"\n{'─'*60}")
    print(f"  MÉTRIQUES GLOBALES")
    print(f"{'─'*60}")
    print(f"  Accuracy (exactitude)    : {g['accuracy']*100:.1f}%")
    print(f"  Macro-Précision          : {g['macro_precision']*100:.1f}%")
    print(f"  Macro-Rappel             : {g['macro_rappel']*100:.1f}%")
    print(f"  Macro-F1                 : {g['macro_f1']*100:.1f}%")
    print(f"  Taux de rejet (Inconnu)  : {g['taux_rejet']*100:.1f}%")
    print(f"  Tests : {g['nb_corrects']}/{g['nb_tests']} corrects, "
          f"{g['nb_erreurs']} erreurs")

    print(f"\n{'─'*60}")
    print(f"  MÉTRIQUES PAR PERSONNE")
    print(f"{'─'*60}")
    print(f"  {'Personne':<15} {'Précision':>10} {'Rappel':>8} {'Spécif.':>9} {'F1':>8}")
    print(f"  {'─'*50}")

    for classe in classes:
        m = metriques['par_classe'][classe]
        print(f"  {classe:<15} "
              f"{m['precision']*100:>9.1f}% "
              f"{m['rappel']*100:>7.1f}% "
              f"{m['specificite']*100:>8.1f}% "
              f"{m['f1']*100:>7.1f}%")

    print(f"{'─'*60}\n")


# ==============================================================================
# MATRICE DE CONFUSION
# ==============================================================================

def _construire_matrice(y_reel, y_predit, classes):
    """
    Construit la matrice de confusion numpy.

    Element [i, j] = nombre de fois où la classe réelle était i
                     et la prédiction était j.
    """
    n = len(classes)
    matrice = np.zeros((n, n), dtype=int)
    classe_idx = {c: i for i, c in enumerate(classes)}

    for reel, predit in zip(y_reel, y_predit):
        i = classe_idx.get(reel)
        j = classe_idx.get(predit)
        if i is not None and j is not None:
            matrice[i, j] += 1

    return matrice


def tracer_matrice_confusion(y_reel, y_predit, classes,
                              chemin_sauvegarde="matrice_confusion.png"):
    """
    Génère et affiche la matrice de confusion.

    La diagonale principale représente les bonnes identifications (TP).
    Les éléments hors diagonale sont les erreurs.

    Paramètres
    ----------
    y_reel, y_predit  : listes de labels
    classes           : liste des classes dans l'ordre
    chemin_sauvegarde : où sauvegarder l'image PNG
    """
    matrice = _construire_matrice(y_reel, y_predit, classes)
    n = len(classes)

    # --- Figure matplotlib ---
    taille = max(6, n * 0.8)
    fig, ax = plt.subplots(figsize=(taille + 2, taille))

    # Colormap : blanc (0 erreurs) → rouge (beaucoup d'erreurs)
    # Mais la diagonale sera affichée en vert (bonnes identifications)
    im = ax.imshow(matrice, cmap='Reds', aspect='auto')

    # Superposer la diagonale en vert
    diag = np.zeros_like(matrice, dtype=float)
    for i in range(n):
        diag[i, i] = matrice[i, i]

    ax.imshow(np.ma.masked_where(diag == 0, diag),
              cmap='Greens', aspect='auto', alpha=0.6)

    # Annotations : valeur dans chaque cellule
    for i in range(n):
        for j in range(n):
            valeur = matrice[i, j]
            if valeur > 0:
                couleur_texte = 'white' if (i == j and valeur > matrice.max()/2) else 'black'
                ax.text(j, i, str(valeur),
                        ha='center', va='center',
                        fontsize=max(8, 12 - n//3),
                        color=couleur_texte,
                        fontweight='bold' if i == j else 'normal')

    # Étiquettes des axes
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    ax.set_xlabel("Prédit par le système", fontsize=11)
    ax.set_ylabel("Réel (vérité terrain)", fontsize=11)
    ax.set_title("Matrice de Confusion — Identification Faciale", fontsize=13, fontweight='bold')

    # Légende
    patch_vert  = mpatches.Patch(color='green',  alpha=0.6, label='Bonne identification (TP)')
    patch_rouge = mpatches.Patch(color='red',    alpha=0.6, label='Erreur (FP / FN)')
    ax.legend(handles=[patch_vert, patch_rouge], loc='upper right',
              bbox_to_anchor=(1.35, 1.0))

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    # Sauvegarder et afficher
    plt.savefig(chemin_sauvegarde, dpi=150, bbox_inches='tight')
    print(f"[EVAL] Matrice de confusion sauvegardée → {chemin_sauvegarde}")
    plt.show()
