# 🎯 Système d'Identification Faciale Géométrique
### Projet de Fin de Module — Vision Artificielle
**Master 1 IA — Université de Béjaia | Mme S. Boukerram | 2025/2026**  
**Équipe :** Yanis · Khadidja · Sarah · Mehdi — AFFUD IGERZEN

---

## 📋 Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Raisonnement scientifique](#2-raisonnement-scientifique)
3. [Architecture du projet](#3-architecture-du-projet)
4. [Description détaillée de chaque fichier](#4-description-détaillée-de-chaque-fichier)
5. [Pipeline complet expliqué](#5-pipeline-complet-expliqué)
6. [Installation et lancement](#6-installation-et-lancement)
7. [Utilisation du système](#7-utilisation-du-système)
8. [Paramètres à ajuster](#8-paramètres-à-ajuster)
9. [Comprendre les résultats](#9-comprendre-les-résultats)
10. [Limitations et pistes d'amélioration](#10-limitations-et-pistes-damélioration)

---

## 1. Vue d'ensemble

Ce projet implémente un système complet d'identification faciale **sans apprentissage automatique ni deep learning**, uniquement avec des techniques classiques de vision par ordinateur.

### Pourquoi "sans apprentissage" ?

La contrainte "sans ML" oblige à comprendre et coder chaque étape manuellement :

| Approche ML (interdit ici) | Notre approche (géométrique) |
|---|---|
| Réseau de neurones apprend les features | On choisit les features à la main |
| Boîte noire | Chaque décision est explicable |
| Nécessite des milliers d'images | Fonctionne avec 20 images par personne |
| Lent à entraîner | Pas d'entraînement, identification instantanée |

### Idée centrale

Un visage est représenté par **30 nombres** (distances entre points anatomiques, normalisées). Identifier une personne = trouver les 30 nombres les plus proches dans une base de données.

---

## 2. Raisonnement scientifique

### Pourquoi les distances géométriques ?

L'intuition est simple : **deux photos du même visage, prises dans des conditions différentes, auront des distances inter-landmarks très similaires** — à condition de bien normaliser.

```
Photo 1 (Alice, lumière normale) :
  distance œil-gauche → nez = 28px / 45px (inter-oculaire) = 0.62

Photo 2 (Alice, à 2m de la caméra) :
  distance œil-gauche → nez = 14px / 22px (inter-oculaire) = 0.64

→ Même ratio, malgré l'échelle différente !
```

### Pourquoi le Snake ?

Le contour actif extrait la **forme de l'ovale facial** (front, joues, menton). Cette information complète les distances entre landmarks, car deux personnes peuvent avoir des landmarks à des positions similaires mais des formes de visage différentes (rond vs allongé).

### Pourquoi la distance euclidienne ?

La distance euclidienne entre deux vecteurs 30D est une mesure de leur "ressemblance". Dans notre espace des descripteurs, les vecteurs d'une même personne se regroupent naturellement en **cluster** (nuage de points). L'identification consiste à trouver à quel cluster appartient un vecteur inconnu.

```
Espace 30D simplifié (projeté en 2D pour l'illustration) :

    Alice ●●●      Bob ●●
         ●●           ●●●
              ?←       Inconnu ?
              → distance(?, Alice) < distance(?, Bob)
              → Identifié comme Alice ✓
```

---

## 3. Architecture du projet

```
projet_facial/
│
├── main.py            ← Point d'entrée. Lance la webcam et orchestre tout.
├── detection.py       ← Détection Haar + alignement par les yeux.
├── snake.py           ← Contour actif (modèle Snake de Kass 1988).
├── descripteurs.py    ← Landmarks + distances normalisées → vecteur 30D.
├── capture.py         ← Lecture/écriture du dataset CSV + capture webcam.
├── identification.py  ← Recherche 1-NN, score de confiance, calibration.
├── porte.py           ← Animation Tkinter de la porte sécurisée.
├── evaluation.py      ← Matrice de confusion, précision, rappel, F1.
│
├── dataset.csv        ← Base de données (généré automatiquement).
├── matrice_confusion.png ← Généré lors de l'évaluation.
│
├── haarcascades/      ← Fichiers XML des classificateurs Haar (optionnel).
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
│
├── README.md          ← Ce fichier.
└── CHECKLIST.md       ← Suivi des étapes du projet.
```

---

## 4. Description détaillée de chaque fichier

### `main.py` — Chef d'orchestre
**Rôle :** Point d'entrée unique. Gère la boucle temps réel, les touches clavier, et coordonne tous les modules.

**Ce qu'il fait :**
- Charge le dataset au démarrage
- Ouvre la webcam (OpenCV VideoCapture)
- Lance la simulation de porte dans un thread Tkinter séparé
- Boucle infinie : lit les frames, gère les touches I/E/Q
- Appelle `pipeline_complet()` qui enchaîne toutes les étapes

**Flux d'exécution :**
```
main()
 └─ charger_dataset()         # Lire dataset.csv
 └─ VideoCapture(0)           # Ouvrir webcam
 └─ SimulateurPorte()         # Créer fenêtre Tkinter
 └─ threading.Thread(porte.lancer)  # Lancer Tkinter en parallèle
 └─ while True:               # Boucle principale
      ├─ cap.read()           # Lire une frame webcam
      ├─ [touche I] pipeline_complet() → identifier() → ouvrir/fermer porte
      ├─ [touche E] mode_enregistrement()
      └─ [touche Q] break
```

---

### `detection.py` — Détection et alignement

**Rôle :** Trouver le visage dans une image brute et le normaliser.

**Algorithme de Viola-Jones (Haar Cascades) :**
1. L'image est parcourue par une fenêtre glissante à plusieurs échelles
2. Dans chaque fenêtre, on calcule des "caractéristiques de Haar" (différences de luminosité entre zones rectangulaires)
3. Un classificateur en cascade décide si c'est un visage (rejet rapide des non-visages)

**Ce que la fonction `detecter_et_aligner()` produit :**
```
Image webcam BGR (ex: 640×480)
    ↓ Conversion gris
    ↓ Détection Haar → boîte (x, y, w, h)
    ↓ Détection des yeux → centres (cx_g, cy_g) et (cx_d, cy_d)
    ↓ Calcul angle = arctan(Δy/Δx)
    ↓ Rotation de l'image entière pour aligner les yeux
    ↓ Recadrage + resize 128×128
    ↓ Égalisation histogramme
Image 128×128 niveaux de gris, normalisée
```

**Pourquoi l'alignement est-il critique ?**
Si le visage est incliné de 15°, les landmarks calculés ensuite seront décalés, et le vecteur 30D sera différent de celui calculé sur un visage droit → mauvaise identification.

---

### `snake.py` — Contour actif

**Rôle :** Extraire le contour précis de l'ovale facial.

**Le modèle Snake :**
- Initialisation : une ellipse de 200 points centrée sur le visage
- À chaque itération : chaque point se déplace pour minimiser :
  - **Énergie interne** : maintenir la régularité de la courbe
  - **Énergie externe** : être attiré vers les zones de fort gradient (bords du visage)
- Après 500 itérations : le contour épouse l'ovale facial

**Paramètres ajustables dans `snake.py` :**
```python
ALPHA = 0.015    # Élasticité : augmenter si la courbe s'étire trop
BETA  = 0.10     # Rigidité   : augmenter si la courbe est trop irrégulière
GAMMA = 0.001    # Pas        : diminuer si le Snake diverge
SIGMA_LISSAGE = 3.0  # Flou préalable : augmenter si l'image est très bruitée
```

---

### `descripteurs.py` — Vecteur 30D

**Rôle :** Convertir un visage en 30 nombres caractéristiques.

**Les 9 landmarks utilisés (positions fixes sur 128×128) :**
```
                    front (0.50, 0.15)
tempe_g(0.10,0.42)  oeil_g(0.33,0.38)   oeil_d(0.67,0.38)  tempe_d(0.90,0.42)
                    nez   (0.50, 0.57)
                  bouche_g(0.36,0.72)  bouche_d(0.64,0.72)
                  bouche_centre(0.50, 0.75)
                    menton(0.50, 0.90)
```

**Construction du vecteur :**
- 25 distances entre paires de landmarks (normalisées par l'écart inter-oculaire)
- 5 features du contour Snake (largeur, hauteur, rapport d'aspect, périmètre, ...)
- **Total : 30 valeurs flottantes**

**Importance de la normalisation :**
```
Sans normalisation : distance œil-nez = 28px (personne proche) ou 14px (personne loin)
Avec normalisation : 28/45 = 0.62   ou   14/22 = 0.64   → presque identiques !
```

---

### `capture.py` — Gestion du dataset

**Rôle :** Lire et écrire le fichier `dataset.csv`.

**Format du CSV :**
```
Alice;1.000000;0.623450;1.234567;...;0.418920
Alice;1.000000;0.631200;1.228900;...;0.415100
Bob;1.000000;0.712340;1.098760;...;0.521300
...
```
- Une ligne = un enregistrement (nom + 30 valeurs)
- Une personne peut apparaître plusieurs fois (20 lignes recommandées)
- Séparateur `;` (et non `,`) pour éviter les conflits avec les décimales

**Fonction `capturer_dataset_webcam()` :**
Automatise la capture des 20 images par personne selon le protocole :
- Phase 1 (images 1-5) : expression neutre
- Phase 2 (images 6-10) : expressions variées
- Phase 3 (images 11-15) : rotations ±15°
- Phase 4 (images 16-20) : éclairages différents

---

### `identification.py` — Reconnaissance 1-NN

**Rôle :** Comparer un vecteur inconnu à tous ceux du dataset.

**Algorithme (très simple mais efficace) :**
```python
distances = [euclidienne(v_inconnu, v_i) for v_i in dataset]
idx_min = argmin(distances)
if distances[idx_min] <= seuil:
    return personnes[idx_min]    # Accès autorisé
else:
    return "Inconnu"             # Accès refusé
```

**Score de confiance :**
```
confiance = max(0, (1 - distance/seuil)) × 100%
```
- distance = 0.0 → confiance = 100% (vecteur identique)
- distance = seuil → confiance ≈ 0%
- distance > seuil → confiance = 0%, décision = "Inconnu"

---

### `porte.py` — Simulation Tkinter

**Rôle :** Afficher visuellement la décision d'identification.

**Ce qui s'affiche :**
- Porte fermée + voyant ORANGE → état d'attente
- Porte s'ouvre progressivement + voyant VERT → accès autorisé
- Flash rouge × 3 + porte fermée → accès refusé

**Technique d'animation :**
On utilise `root.after(30ms, prochain_frame)` pour appeler une fonction
toutes les 30ms et incrémenter l'angle d'ouverture de la porte.
L'effet 3D est simulé par `largeur_visible = largeur_max × cos(angle)`.

---

### `evaluation.py` — Métriques de performance

**Rôle :** Mesurer objectivement la qualité du système.

**Métriques calculées :**

| Métrique | Formule | Interprétation |
|---|---|---|
| Accuracy | TP_total / N | % de bonnes identifications |
| Précision | TP / (TP+FP) | Quand on dit "Alice", a-t-on raison ? |
| Rappel | TP / (TP+FN) | Parmi toutes les Alice, combien retrouve-t-on ? |
| Spécificité | TN / (TN+FP) | Quand ce n'est pas Alice, le dit-on correctement ? |
| F1 | 2×P×R/(P+R) | Moyenne harmonique précision/rappel |

**Méthode Leave-One-Out (LOO) :**
Pour chaque enregistrement i, on le retire du dataset et on tente de l'identifier
avec les N-1 restants. On obtient ainsi N prédictions sur tout le dataset,
sans avoir à séparer train/test manuellement.

---

## 5. Pipeline complet expliqué

```
                        ┌─────────────────────────────────┐
                        │        WEBCAM (flux vidéo)       │
                        └──────────────┬──────────────────┘
                                       │ frame BGR (640×480)
                                       ▼
                        ┌─────────────────────────────────┐
                        │  detection.py                    │
                        │  · Haar → boîte visage           │
                        │  · Détection yeux → angle        │
                        │  · Rotation + resize 128×128     │
                        │  · Égalisation histogramme       │
                        └──────────────┬──────────────────┘
                                       │ image 128×128 gris
                                       ▼
                        ┌─────────────────────────────────┐
                        │  snake.py                        │
                        │  · Initialisation ellipse        │
                        │  · Optimisation 500 itérations   │
                        │  · Contour ovale 200 points      │
                        └──────────────┬──────────────────┘
                                       │ contour (200, 2)
                                       ▼
                        ┌─────────────────────────────────┐
                        │  descripteurs.py                 │
                        │  · 9 landmarks → 25 distances    │
                        │  · Normalisation inter-oculaire  │
                        │  · 5 features Snake              │
                        └──────────────┬──────────────────┘
                                       │ vecteur 30D
                        ┌──────────────┴──────────────────┐
                        │                                  │
              Mode E (Enregistrement)         Mode I (Identification)
                        │                                  │
                        ▼                                  ▼
           ┌────────────────────┐           ┌──────────────────────────┐
           │  capture.py        │           │  identification.py        │
           │  Écrire dans CSV   │           │  Comparer à tous les     │
           │  dataset.csv       │           │  vecteurs du dataset     │
           └────────────────────┘           │  → Plus petit voisin     │
                                            │  → Score de confiance    │
                                            └──────────────┬───────────┘
                                                           │ (nom, dist, conf)
                                                           ▼
                                            ┌──────────────────────────┐
                                            │  porte.py                │
                                            │  dist ≤ seuil →  OUVRIR │
                                            │  dist > seuil →  FERMER │
                                            └──────────────────────────┘
```

---

## 6. Installation et lancement

### Prérequis
- Python 3.9 ou supérieur
- Webcam fonctionnelle
- Système : Windows / Linux / macOS

### Installation des dépendances

```bash
pip install opencv-python numpy scipy scikit-image pandas matplotlib
```

### Lancement du système principal

```bash
cd projet_facial/
python main.py
```

### Lancement de la capture dataset (première utilisation)

```python
# Dans un terminal Python ou un script séparé :
from capture import capturer_dataset_webcam
capturer_dataset_webcam("Alice", nb_images=20)
```

### Évaluation des performances

```bash
# La touche V dans main.py lance l'évaluation automatiquement.
# Ou directement :
python -c "
from capture import charger_dataset
from evaluation import evaluer_depuis_dataset
personnes, vecteurs = charger_dataset()
evaluer_depuis_dataset(personnes, vecteurs, seuil=0.5)
"
```

---

## 7. Utilisation du système

### Touches clavier (fenêtre webcam active)

| Touche | Action | Effet |
|--------|--------|-------|
| **I** | Identifier | Analyse le visage → ouvre ou ferme la porte |
| **E** | Enregistrer | Ajoute une personne au dataset CSV |
| **V** | Valider | Lance l'évaluation (matrice de confusion) |
| **Q** | Quitter | Ferme toutes les fenêtres proprement |

### Procédure d'enregistrement d'une nouvelle personne

1. Lancer `main.py`
2. Appuyer sur **E**
3. Suivre les instructions dans le terminal
4. Se placer face à la caméra
5. Appuyer sur ENTRÉE
6. Entrer le nom de la personne
7. Répéter 20 fois (automatique si on utilise `capturer_dataset_webcam`)

### Procédure d'identification

1. Se placer face à la caméra
2. Appuyer sur **I**
3. Observer : porte ouverte (vert) ou fermée (rouge)
4. Voir le nom, la distance et le score de confiance sur l'overlay webcam

---

## 8. Paramètres à ajuster

### Seuil de décision (`main.py`)
```python
SEUIL_DECISION = 0.5  # À modifier selon vos tests
```
- **Trop petit (ex: 0.2)** → Beaucoup de "Inconnu", même pour des personnes enregistrées
- **Trop grand (ex: 1.5)** → Accepte des inconnus (faux positifs de sécurité)
- **Recommandation** : Utiliser `calibrer_seuil()` dans `identification.py`

### Paramètres du Snake (`snake.py`)
```python
ALPHA = 0.015   # Augmenter si le contour s'étire trop loin du visage
BETA  = 0.10    # Augmenter si le contour est trop irrégulier/anguleux
GAMMA = 0.001   # Diminuer si le Snake "saute" et diverge
SIGMA_LISSAGE = 3.0  # Augmenter si l'image est très bruitée
NB_ITERATIONS = 500  # Augmenter pour plus de précision (plus lent)
```

### Positions des landmarks (`descripteurs.py`)
```python
LANDMARKS_RELATIFS = {
    'oeil_gauche' : (0.33, 0.38),  # Modifier si les yeux semblent mal placés
    'nez'         : (0.50, 0.57),  # Ajuster selon vos observations
    # ...
}
```

---

## 9. Comprendre les résultats

### Distance euclidienne
- `0.00` → Vecteur identique (impossible en pratique, même personne même image)
- `< 0.3` → Très proche, haute confiance
- `0.3–0.6` → Proche, confiance modérée
- `> 0.6` → Éloigné → probablement "Inconnu"

### Score de confiance
```
confiance = max(0, (1 - distance/seuil)) × 100%
```
- `90%–100%` → Identification très fiable
- `50%–90%` → Identification probable, mais vérifier le Top 3
- `< 50%`  → Cas ambigu, augmenter le nombre d'enregistrements

### Matrice de confusion
- **Diagonale verte** = bonnes identifications (TP)
- **Hors diagonale** = erreurs (qui est confondu avec qui ?)
- Une colonne "Inconnu" remplie → seuil trop petit
- Une ligne "Personne X" diffuse partout → peu d'enregistrements pour X

---

## 10. Limitations et pistes d'amélioration

### Limitations actuelles

1. **Landmarks fixes** : On utilise des positions anatomiques moyennes, pas des landmarks détectés dynamiquement. Une personne avec un visage très atypique sera moins bien représentée.

2. **Sensibilité aux occultations** : Port de lunettes, masque, barbe → les landmarks oculaires et nasaux sont perturbés.

3. **Profil** : La détection Haar fonctionne mieux de face. Un profil à 90° ne sera pas détecté.

4. **Qualité du Snake** : Pour des visages avec peu de contraste aux bords (peau claire sur fond clair), le Snake peut ne pas converger vers le vrai contour facial.

### Pistes d'amélioration (hors contraintes du projet)

| Amélioration | Impact | Complexité |
|---|---|---|
| Utiliser dlib pour les 68 landmarks | +++ précision descripteurs | Moyen |
| LBP (Local Binary Patterns) | Meilleure robustesse | Faible |
| Normalisation CLAHE à la place de equalizeHist | +robustesse éclairage | Très faible |
| K-NN au lieu de 1-NN | +robustesse aux outliers | Faible |
| PCA sur les vecteurs 30D | Réduction du bruit | Moyen |

---

*Projet réalisé dans le cadre du module Vision Artificielle, Master 1 IA, Université de Béjaia.*  
*Approche 100% classique — Aucun apprentissage automatique utilisé.*
