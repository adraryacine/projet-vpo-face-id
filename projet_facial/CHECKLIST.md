# ✅ CHECKLIST DU PROJET — Identification Faciale Géométrique
### Master 1 IA — Vision Artificielle | Mme S. Boukerram | 2025/2026
**Équipe :** Yanis · Khadidja · Sarah · Mehdi — AFFUD IGERZEN

---

> **Comment utiliser cette checklist :**
> Cochez chaque case `[ ]` en la remplaçant par `[x]` au fur et à mesure.
> Chaque case inclut une explication de ce qu'il faut faire et pourquoi.

---

## 🗂️ PHASE 0 — Préparation de l'environnement
*Durée estimée : 2 heures | À faire en équipe dès le premier jour*

- [ ] **0.1** Installer Python 3.9+ sur tous les postes de l'équipe
  - Vérifier : `python --version` dans le terminal

- [ ] **0.2** Créer le dossier de projet `projet_facial/` avec la structure décrite dans le README

- [ ] **0.3** Installer toutes les dépendances Python
  ```bash
  pip install opencv-python numpy scipy scikit-image pandas matplotlib
  ```

- [ ] **0.4** Vérifier que la webcam fonctionne avec OpenCV
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  print("Webcam OK :", ret)   # Doit afficher True
  cap.release()
  ```

- [ ] **0.5** Vérifier que les fichiers XML Haar sont accessibles
  ```python
  import cv2, os
  chemin = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
  print("Haar OK :", os.path.exists(chemin))   # Doit afficher True
  ```

- [ ] **0.6** Tester l'import de tous les modules du projet
  ```python
  from detection import detecter_et_aligner
  from snake import appliquer_snake
  from descripteurs import calculer_vecteur_30D
  from capture import charger_dataset, enregistrer_personne
  from identification import identifier
  from porte import SimulateurPorte
  from evaluation import evaluer_depuis_dataset
  print("Tous les modules sont importés correctement !")
  ```

- [ ] **0.7** Répartir les tâches entre les membres de l'équipe :
  - Membre 1 : Détection + alignement (`detection.py`)
  - Membre 2 : Snake (`snake.py`) + Descripteurs (`descripteurs.py`)
  - Membre 3 : Identification (`identification.py`) + Dataset (`capture.py`)
  - Membre 4 : Porte (`porte.py`) + Évaluation (`evaluation.py`) + Main

---

## 📸 PHASE 1 — Construction du Dataset
*Durée estimée : 4 heures | Semaine 1*

### Étape 1.1 — Test de la capture webcam
- [ ] **1.1.1** Lancer la webcam et afficher le flux en temps réel
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  while True:
      ret, frame = cap.read()
      cv2.imshow("Test webcam", frame)
      if cv2.waitKey(1) == ord('q'):
          break
  cap.release()
  cv2.destroyAllWindows()
  ```
- [ ] **1.1.2** Vérifier que l'image est claire et bien éclairée

### Étape 1.2 — Test de la détection Haar seule
- [ ] **1.2.1** Tester la fonction `detecter_et_aligner()` sur une image statique
  ```python
  import cv2
  from detection import detecter_et_aligner, dessiner_detection
  img = cv2.imread("une_photo.jpg")  # ou cap.read()
  img_annotee = dessiner_detection(img)
  cv2.imshow("Détection", img_annotee)
  cv2.waitKey(0)
  ```
- [ ] **1.2.2** Vérifier que le visage est bien encadré (rectangle bleu)
- [ ] **1.2.3** Vérifier que les yeux sont détectés (rectangles verts)
- [ ] **1.2.4** Vérifier que l'image 128×128 résultante est bien alignée (yeux horizontaux)

### Étape 1.3 — Capture des enregistrements
- [ ] **1.3.1** Capturer 20 enregistrements pour **Membre 1 de l'équipe**
  ```python
  from capture import capturer_dataset_webcam
  capturer_dataset_webcam("NomMembre1", nb_images=20)
  ```
- [ ] **1.3.2** Capturer 20 enregistrements pour **Membre 2**
- [ ] **1.3.3** Capturer 20 enregistrements pour **Membre 3**
- [ ] **1.3.4** Capturer 20 enregistrements pour **Membre 4**
- [ ] **1.3.5** Capturer 20 enregistrements pour **6 autres personnes** (amis, famille, collègues)
  - Total requis : **minimum 10 personnes × 20 images = 200 lignes CSV**

### Étape 1.4 — Vérification du dataset
- [ ] **1.4.1** Ouvrir `dataset.csv` dans un éditeur texte/Excel et vérifier le format
  - Chaque ligne = 31 colonnes (1 nom + 30 valeurs)
  - Pas de lignes vides
  - Pas de valeurs manquantes

- [ ] **1.4.2** Compter les enregistrements par personne
  ```python
  from capture import lister_personnes
  stats = lister_personnes()
  for nom, nb in stats.items():
      print(f"  {nom}: {nb} enregistrements {'✓' if nb >= 20 else '⚠ insuffisant'}")
  ```
- [ ] **1.4.3** Chaque personne a bien ≥ 20 enregistrements

### Étape 1.5 — Vérification de la diversité
- [ ] **1.5.1** Confirmer la couverture des 4 phases pour chaque personne :
  - [ ] Phase 1 : 5 images de face, expression neutre
  - [ ] Phase 2 : 5 images avec expressions variées (sourire, surprise, yeux plissés)
  - [ ] Phase 3 : 5 images avec rotation ±15° (profil partiel)
  - [ ] Phase 4 : 5 images avec éclairage différent (gauche, droite, contre-jour)

---

## 🔍 PHASE 2 — Détection et Alignement
*Durée estimée : 4 heures | Semaine 1-2*

### Étape 2.1 — Implémentation
- [ ] **2.1.1** Le fichier `detection.py` est complété et sans erreur de syntaxe
- [ ] **2.1.2** La fonction `detecter_et_aligner(image)` retourne bien une image 128×128
- [ ] **2.1.3** La fonction retourne `None` si aucun visage n'est détecté

### Étape 2.2 — Tests de robustesse
- [ ] **2.2.1** Tester sur un visage de face → fonctionne
- [ ] **2.2.2** Tester sur un visage légèrement incliné → l'alignement corrige l'angle
- [ ] **2.2.3** Tester avec une image sans visage → retourne `None` sans planter
- [ ] **2.2.4** Tester avec deux visages dans l'image → prend le plus grand
- [ ] **2.2.5** Tester avec un éclairage sombre → l'égalisation d'histogramme améliore

### Étape 2.3 — Validation visuelle
- [ ] **2.3.1** Afficher côte à côte : image brute et image normalisée 128×128
  - Les yeux doivent être à la même hauteur dans l'image normalisée
  - L'histogramme doit être étalé sur [0, 255]

---

## 🐍 PHASE 3 — Contour Actif (Snake)
*Durée estimée : 6 heures | Semaine 2 — Module le plus difficile !*

### Étape 3.1 — Compréhension théorique
- [ ] **3.1.1** Comprendre la notion d'énergie interne (élasticité α + rigidité β)
- [ ] **3.1.2** Comprendre la notion d'énergie externe (gradient de l'image)
- [ ] **3.1.3** Comprendre le rôle du lissage gaussien préalable

### Étape 3.2 — Implémentation
- [ ] **3.2.1** Le fichier `snake.py` est complété sans erreur
- [ ] **3.2.2** La fonction `appliquer_snake(visage_128)` retourne un array (200, 2)
- [ ] **3.2.3** La fonction `extraire_features_contour(contour, nb_features=5)` retourne 5 valeurs

### Étape 3.3 — Validation visuelle du Snake
- [ ] **3.3.1** Utiliser `visualiser_snake()` pour afficher le contour sur le visage
  ```python
  from detection import detecter_et_aligner
  from snake import appliquer_snake, visualiser_snake
  import cv2
  img = cv2.imread("votre_photo.jpg")
  visage = detecter_et_aligner(img)
  contour = appliquer_snake(visage)
  visualiser_snake(visage, contour)
  ```
- [ ] **3.3.2** Vérifier que le contour vert entoure bien l'ovale facial (pas les cheveux)
- [ ] **3.3.3** Vérifier que le contour est régulier (pas de pics/saillies)

### Étape 3.4 — Ajustement des paramètres
- [ ] **3.4.1** Si le contour ne converge pas vers le visage → réduire ALPHA
- [ ] **3.4.2** Si le contour est trop irrégulier → augmenter BETA
- [ ] **3.4.3** Si le Snake "explose" → réduire GAMMA
- [ ] **3.4.4** Si le Snake s'accroche aux détails → augmenter SIGMA_LISSAGE
- [ ] **3.4.5** Les paramètres finaux retenus sont notés dans le rapport

---

## 📐 PHASE 4 — Descripteurs Géométriques
*Durée estimée : 3 heures | Semaine 2*

### Étape 4.1 — Implémentation
- [ ] **4.1.1** Le fichier `descripteurs.py` est complété sans erreur
- [ ] **4.1.2** La fonction `calculer_vecteur_30D()` retourne bien un array de taille 30
- [ ] **4.1.3** Les landmarks sont dans des positions anatomiquement cohérentes

### Étape 4.2 — Validation des landmarks
- [ ] **4.2.1** Utiliser `visualiser_landmarks()` pour afficher les 9 points sur le visage
  ```python
  from descripteurs import visualiser_landmarks
  from detection import detecter_et_aligner
  import cv2
  img = cv2.imread("votre_photo.jpg")
  visage = detecter_et_aligner(img)
  visualiser_landmarks(visage)
  ```
- [ ] **4.2.2** Les points œil_gauche et œil_droit tombent sur les yeux
- [ ] **4.2.3** Le point nez tombe sur la pointe du nez
- [ ] **4.2.4** Les points bouche sont sur les commissures des lèvres
- [ ] **4.2.5** Si les points sont mal placés → ajuster `LANDMARKS_RELATIFS` dans `descripteurs.py`

### Étape 4.3 — Validation du vecteur
- [ ] **4.3.1** Calculer le vecteur de la même personne sur 5 photos différentes
  ```python
  from descripteurs import calculer_vecteur_30D
  import numpy as np
  # vecteurs_alice = [v1, v2, v3, v4, v5]
  print("Écart-type des vecteurs Alice :", np.std(vecteurs_alice, axis=0).mean())
  ```
- [ ] **4.3.2** L'écart-type intra-personne est faible (< 0.15 idéalement)
- [ ] **4.3.3** Comparer les vecteurs de deux personnes différentes : ils doivent être éloignés
  ```python
  from descripteurs import comparer_vecteurs
  metriques = comparer_vecteurs(v_alice, v_bob)
  print("Distance Alice-Bob :", metriques['euclidienne'])  # Doit être > intra-Alice
  ```

---

## 🔎 PHASE 5 — Identification
*Durée estimée : 3 heures | Semaine 3*

### Étape 5.1 — Implémentation
- [ ] **5.1.1** Le fichier `identification.py` est complété sans erreur
- [ ] **5.1.2** La fonction `identifier()` retourne bien (nom, distance, confiance, top3)
- [ ] **5.1.3** Le Top 3 contient bien 3 candidats triés par distance croissante

### Étape 5.2 — Tests de base
- [ ] **5.2.1** Tester l'identification d'une personne enregistrée → doit retourner son nom
- [ ] **5.2.2** Tester avec un vecteur inconnu (personne non enregistrée) → doit retourner "Inconnu"
- [ ] **5.2.3** Tester avec un dataset vide → doit lever une ValueError claire

### Étape 5.3 — Calibration du seuil
- [ ] **5.3.1** Lancer la calibration automatique
  ```python
  from identification import calibrer_seuil
  from capture import charger_dataset
  personnes, vecteurs = charger_dataset()
  seuil_optimal, rapport = calibrer_seuil(personnes, vecteurs)
  print(f"Seuil optimal : {seuil_optimal}")
  ```
- [ ] **5.3.2** Mettre à jour `SEUIL_DECISION` dans `main.py` avec la valeur trouvée
- [ ] **5.3.3** Tracer la courbe accuracy vs seuil (pour le rapport)

### Étape 5.4 — Test du pipeline intégré
- [ ] **5.4.1** Se placer devant la webcam et appuyer sur I dans `main.py`
- [ ] **5.4.2** Vérifier que le nom affiché correspond à la personne devant la caméra
- [ ] **5.4.3** Tester avec une personne non enregistrée → "Inconnu" doit s'afficher

---

## 🚪 PHASE 6 — Simulation de Porte
*Durée estimée : 2 heures | Semaine 3*

### Étape 6.1 — Implémentation
- [ ] **6.1.1** Le fichier `porte.py` est complété sans erreur
- [ ] **6.1.2** La fenêtre Tkinter s'ouvre correctement

### Étape 6.2 — Tests visuels
- [ ] **6.2.1** Tester `porte.ouvrir_porte("Alice")` → porte s'anime + voyant VERT
- [ ] **6.2.2** Tester `porte.fermer_porte()` → flash rouge × 3
- [ ] **6.2.3** Tester `porte.en_attente()` → voyant orange, porte fermée
- [ ] **6.2.4** L'animation est fluide (pas de saccades)
- [ ] **6.2.5** La fenêtre ne bloque pas la fenêtre webcam OpenCV

### Étape 6.3 — Intégration avec main.py
- [ ] **6.3.1** La porte répond bien à l'appui sur I dans la fenêtre webcam
- [ ] **6.3.2** L'overlay OpenCV affiche correctement le nom et la distance
- [ ] **6.3.3** Le nom affiché dans la porte = le nom affiché dans l'overlay webcam

---

## 📊 PHASE 7 — Évaluation des Performances
*Durée estimée : 3 heures | Semaine 4*

### Étape 7.1 — Matrice de confusion
- [ ] **7.1.1** Lancer l'évaluation complète (touche V ou script direct)
- [ ] **7.1.2** La matrice de confusion s'affiche correctement
- [ ] **7.1.3** Sauvegarder `matrice_confusion.png` pour le rapport

### Étape 7.2 — Métriques globales
- [ ] **7.2.1** Calculer et noter les métriques suivantes :
  | Métrique | Valeur obtenue | Objectif |
  |---|---|---|
  | Accuracy | ___% | > 80% |
  | Macro-Précision | ___% | > 75% |
  | Macro-Rappel | ___% | > 75% |
  | Macro-F1 | ___% | > 75% |
  | Taux de rejet | ___% | < 20% |

### Étape 7.3 — Analyse des erreurs
- [ ] **7.3.1** Identifier les personnes les plus souvent confondues
  ```python
  from identification import analyser_confusions
  from capture import charger_dataset
  personnes, vecteurs = charger_dataset()
  erreurs = analyser_confusions(personnes, vecteurs, seuil=0.5)
  ```
- [ ] **7.3.2** Pour chaque paire confondue, analyser :
  - [ ] Les deux visages ont-ils des proportions similaires ?
  - [ ] Le Snake produit-il des contours similaires ?
  - [ ] Quelle feature du vecteur 30D est-elle la plus différente ?
- [ ] **7.3.3** Proposer une correction (modifier seuil, ajouter landmarks, ajuster Snake)
- [ ] **7.3.4** Après correction, relancer l'évaluation et comparer les métriques

### Étape 7.4 — Rapport sur les performances
- [ ] **7.4.1** Rédiger l'analyse des erreurs dans le rapport
- [ ] **7.4.2** Inclure la matrice de confusion annotée
- [ ] **7.4.3** Inclure la courbe accuracy vs seuil
- [ ] **7.4.4** Comparer les métriques avant/après ajustement des paramètres

---

## 🚀 PHASE 8 — Intégration finale et démo
*Durée estimée : 2 heures | Semaine 4*

### Étape 8.1 — Test du système complet
- [ ] **8.1.1** Lancer `python main.py` sur une machine "vierge" (sans le dataset)
- [ ] **8.1.2** Enregistrer 3 personnes en live (touche E × 3)
- [ ] **8.1.3** Identifier chacune des 3 personnes (touche I)
- [ ] **8.1.4** Tenter d'accéder avec une personne non enregistrée → refusé
- [ ] **8.1.5** Vérifier que toutes les touches (I, E, V, Q) fonctionnent

### Étape 8.2 — Robustesse
- [ ] **8.2.1** Tester avec un éclairage différent de celui de l'enregistrement
- [ ] **8.2.2** Tester avec une expression différente de celle de l'enregistrement
- [ ] **8.2.3** Tester avec une légère rotation de la tête

### Étape 8.3 — Nettoyage du code
- [ ] **8.3.1** Supprimer tous les `print()` de débogage inutiles
- [ ] **8.3.2** Vérifier que tous les fichiers ont leurs en-têtes de documentation
- [ ] **8.3.3** Tester que `python main.py` fonctionne sans erreur du premier coup

---

## 📄 PHASE 9 — Livrables
*Durée estimée : 4 heures | Semaine 4*

### Étape 9.1 — Code
- [ ] **9.1.1** Tous les fichiers `.py` sont présents et fonctionnels
- [ ] **9.1.2** Le code est commenté (chaque fonction a une docstring)
- [ ] **9.1.3** Un fichier `requirements.txt` liste les dépendances
  ```bash
  pip freeze > requirements.txt
  ```
- [ ] **9.1.4** Le dossier projet est zippé proprement (sans `__pycache__`)

### Étape 9.2 — Dataset
- [ ] **9.2.1** `dataset.csv` contient ≥ 200 lignes (10 personnes × 20 images)
- [ ] **9.2.2** Le fichier CSV est inclus dans le livrable

### Étape 9.3 — Rapport
- [ ] **9.3.1** Page de garde (nom équipe, module, date)
- [ ] **9.3.2** Introduction (contexte, objectifs)
- [ ] **9.3.3** Description de chaque module avec schémas
- [ ] **9.3.4** Paramètres retenus et justification
- [ ] **9.3.5** Résultats d'évaluation (matrice de confusion + métriques)
- [ ] **9.3.6** Analyse des erreurs et pistes d'amélioration
- [ ] **9.3.7** Conclusion
- [ ] **9.3.8** Références bibliographiques (Viola-Jones, Kass et al.)

### Étape 9.4 — Présentation
- [ ] **9.4.1** Slides avec le pipeline illustré (schéma blocs)
- [ ] **9.4.2** Capture d'écran de la détection Haar
- [ ] **9.4.3** Capture d'écran du Snake sur un visage
- [ ] **9.4.4** Capture d'écran de la porte (ouverte et fermée)
- [ ] **9.4.5** Matrice de confusion commentée
- [ ] **9.4.6** Démo live prévue (tester avant la présentation !)

---

## 📚 RÉFÉRENCES UTILES

| Concept | Référence |
|---|---|
| Algoritme Viola-Jones | Viola & Jones, "Rapid object detection using a boosted cascade of simple features", CVPR 2001 |
| Modèle Snake | Kass, Witkin & Terzopoulos, "Snakes: Active contour models", IJCV 1988 |
| Équivalisation histogramme | Gonzalez & Woods, "Digital Image Processing", Chapitre 3 |
| Distance euclidienne | Cours de Mme Boukerram — Vision Artificielle |
| scikit-image Snake | https://scikit-image.org/docs/stable/api/skimage.segmentation.html |
| OpenCV Haar | https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html |

---

## 🏁 RÉCAPITULATIF DES PHASES

| Phase | Sujet | Semaine | Responsable | Statut |
|---|---|---|---|---|
| 0 | Environnement | S1 | Équipe | `[ ]` |
| 1 | Dataset | S1 | Équipe | `[ ]` |
| 2 | Détection Haar | S1-S2 | Membre 1 | `[ ]` |
| 3 | Snake | S2 | Membre 2 | `[ ]` |
| 4 | Descripteurs 30D | S2 | Membre 2 | `[ ]` |
| 5 | Identification 1-NN | S3 | Membre 3 | `[ ]` |
| 6 | Porte Tkinter | S3 | Membre 4 | `[ ]` |
| 7 | Évaluation | S4 | Équipe | `[ ]` |
| 8 | Intégration | S4 | Équipe | `[ ]` |
| 9 | Livrables | S4 | Équipe | `[ ]` |

---

*Bonne chance à toute l'équipe ! 🎓*
