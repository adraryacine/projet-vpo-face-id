"""
==============================================================================
FICHIER : main.py
ROLE    : Point d'entrée principal du système d'identification faciale.
          Ce fichier orchestre tous les modules du projet.
          Il gère la boucle principale en temps réel via la webcam,
          les touches clavier (I = identifier, E = enregistrer, Q = quitter),
          et coordonne détection → Snake → descripteurs → identification → porte.

AUTEURS : Yanis, Khadidja, Sarah, Mehdi — Master 1 IA, Université de Béjaia
MODULE  : Vision Artificielle — Mme S. Boukerram
ANNÉE   : 2025/2026
==============================================================================
"""

import time

try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2          # OpenCV : capture webcam, affichage, overlays
import numpy as np  # NumPy  : calculs matriciels sur les images/vecteurs
import threading    # Thread : lancer la fenêtre Tkinter en parallèle d'OpenCV

# --- Import de tous les modules du projet ---
from detection       import detecter_et_aligner      # Étape 1 : Haar + alignement
from snake           import appliquer_snake           # Étape 2 : Contour actif
from descripteurs    import calculer_vecteur_30D      # Étape 3 : Vecteur géométrique 30D
from capture         import enregistrer_personne      # Gestion CSV (enregistrement)
from capture         import charger_dataset           # Gestion CSV (chargement)
from identification  import identifier_knn            # Vote K-NN plus stable
from porte           import SimulateurPorte           # Animation Tkinter
from evaluation      import evaluer_depuis_dataset    # Métriques de performance


# ==============================================================================
# CONSTANTE GLOBALE : seuil de décision
# Si distance_euclidienne(inconnu, candidat) <= SEUIL → accès autorisé
# Valeur initiale empirique : à ajuster selon vos tests
# ==============================================================================
SEUIL_DECISION = 0.75
NOM_FENETRE = "Systeme d'identification faciale"
NB_FRAMES_SCAN = 5
MAX_FRAMES_SCAN = 20
NB_ENREGISTREMENTS_PAR_PERSONNE = 12


def afficher_overlay(frame, nom, distance, confiance, top3, acces):
    """
    Dessine les informations d'identification directement sur le flux webcam.

    Paramètres
    ----------
    frame     : image BGR provenant de la webcam (numpy array)
    nom       : nom identifié (ou "Inconnu")
    distance  : distance euclidienne minimale trouvée
    confiance : score de confiance en % (0 à 100)
    top3      : liste des 3 meilleurs candidats [(nom, dist), ...]
    acces     : booléen — True = accès autorisé, False = refusé

    Retourne
    --------
    frame modifiée avec les textes/rectangles dessinés dessus
    """
    couleur = (0, 255, 0) if acces else (0, 0, 255)   # Vert si OK, Rouge si refus
    label   = f"{'AUTORISE' if acces else 'REFUSE'} — {nom}"

    # Rectangle coloré en haut à gauche
    cv2.rectangle(frame, (10, 10), (420, 130), couleur, -1)
    cv2.rectangle(frame, (10, 10), (420, 130), (255,255,255), 2)

    # Nom identifié (grande police)
    cv2.putText(frame, label, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Distance et confiance
    cv2.putText(frame, f"Distance  : {distance:.4f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Confiance : {confiance:.1f}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Top 3 candidats (débogage)
    cv2.putText(frame, "Top 3 candidats :", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    for i, (cnom, cdist) in enumerate(top3):
        cv2.putText(frame, f"  {i+1}. {cnom} ({cdist:.3f})",
                    (20, 145 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    # Icône porte miniature (rectangle symbolique)
    px, py = frame.shape[1]-80, 20
    cv2.rectangle(frame, (px, py), (px+55, py+75), couleur, 2)
    cv2.circle(frame, (px+42, py+40), 4, couleur, -1)   # poignée

    return frame


def _est_touche_quitter(touche):
    if touche == -1:
        return False
    code = touche & 0xFF
    return code in (ord('q'), ord('Q'), 27)


def _touche_terminal():
    if msvcrt is None or not msvcrt.kbhit():
        return -1

    touche = msvcrt.getwch()
    if touche in ("\x00", "\xe0"):
        msvcrt.getwch()
        return -1
    return ord(touche)


def lire_touche(delai=30):
    touche = cv2.waitKeyEx(delai)
    if touche != -1:
        return touche
    return _touche_terminal()


def _fenetre_ouverte(nom_fenetre):
    try:
        return cv2.getWindowProperty(nom_fenetre, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def extraire_vecteur_depuis_frame(frame):
    visage = detecter_et_aligner(frame)
    if visage is None:
        return None

    contour = appliquer_snake(visage)
    return calculer_vecteur_30D(visage, contour)


def identifier_vecteur(vecteur, personnes, vecteurs):
    k = min(7, len(personnes))
    return identifier_knn(vecteur, personnes, vecteurs, seuil=SEUIL_DECISION, k=k)


def scanner_identification(cap, personnes, vecteurs):
    if len(personnes) == 0:
        print("[INFO] Dataset vide: enregistrez d'abord une personne avec E.")
        return None, False

    vecteurs_scan = []

    for tentative in range(MAX_FRAMES_SCAN):
        ret, frame = cap.read()
        if not ret:
            continue

        frame_affichage = frame.copy()
        cv2.putText(frame_affichage,
                    f"Scan visage... {len(vecteurs_scan)}/{NB_FRAMES_SCAN}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow(NOM_FENETRE, frame_affichage)

        touche = lire_touche(20)
        if _est_touche_quitter(touche):
            return None, True

        try:
            vecteur = extraire_vecteur_depuis_frame(frame)
        except Exception as exc:
            print(f"[SCAN] Frame ignoree: {exc}")
            continue

        if vecteur is not None:
            vecteurs_scan.append(vecteur)
            print(f"[SCAN] Visage detecte ({len(vecteurs_scan)}/{NB_FRAMES_SCAN})")

        if len(vecteurs_scan) >= NB_FRAMES_SCAN:
            break

    if not vecteurs_scan:
        print("[INFO] Aucun visage detecte pendant le scan.")
        return None, False

    vecteur_median = np.median(np.vstack(vecteurs_scan), axis=0)
    norme = np.linalg.norm(vecteur_median)
    if norme > 1e-8:
        vecteur_median = vecteur_median / norme

    return identifier_vecteur(vecteur_median, personnes, vecteurs), False


def pipeline_complet(frame, personnes, vecteurs):
    vecteur = extraire_vecteur_depuis_frame(frame)
    if vecteur is None:
        return None

    if len(personnes) == 0:
        print("[INFO] Dataset vide: enregistrez d'abord des personnes (touche E).")
        return None

    return identifier_vecteur(vecteur, personnes, vecteurs)

    """
    Applique le pipeline complet sur une frame webcam :
        1. Détection + alignement (Haar)
        2. Snake (contour ovale facial)
        3. Extraction vecteur 30D
        4. Identification 1-NN
        5. Calcul confiance

    Paramètres
    ----------
    frame    : image BGR brute de la webcam
    personnes: liste des noms dans le dataset CSV
    vecteurs : liste des vecteurs 30D correspondants

    Retourne
    --------
    tuple (nom, distance, confiance, top3) ou None si aucun visage détecté
    """
    # --- Étape 1 : Détecter et aligner le visage ---
    visage = detecter_et_aligner(frame)
    if visage is None:
        return None   # Aucun visage dans la frame → on ignore

    # --- Étape 2 : Contour actif Snake ---
    # Le Snake s'adapte à la forme réelle du visage normalisé 128×128
    contour = appliquer_snake(visage)

    # --- Étape 3 : Calcul du vecteur caractéristique 30D ---
    vecteur = calculer_vecteur_30D(visage, contour)

    # --- Étape 4 : Identification par plus proche voisin ---
    if len(personnes) == 0:
        print("[INFO] Dataset vide — enregistrez d'abord des personnes (touche E).")
        return None

    nom, distance, confiance, top3 = identifier(
        vecteur, personnes, vecteurs, seuil=SEUIL_DECISION
    )
    return nom, distance, confiance, top3


def mode_enregistrement(cap, personnes, vecteurs):
    print("\n[MODE ENREGISTREMENT]")
    print("Le terminal attend le nom. La fenetre webcam peut rester ouverte.")
    nom = input("Entrez le nom de la personne : ").strip()
    if not nom:
        print("[ANNULE] Nom vide.")
        return

    print(f"[INFO] Capture de {NB_ENREGISTREMENTS_PAR_PERSONNE} exemples pour '{nom}'.")
    print("Gardez le visage visible, puis variez legerement angle/expression/lumiere.")
    input("Appuyez sur ENTREE pour commencer la capture...")

    captures = 0
    tentatives = 0
    dernier_temps = 0.0
    max_tentatives = NB_ENREGISTREMENTS_PAR_PERSONNE * 8

    while captures < NB_ENREGISTREMENTS_PAR_PERSONNE and tentatives < max_tentatives:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_affichage = frame.copy()
        cv2.putText(frame_affichage,
                    f"Enregistrement {nom}: {captures}/{NB_ENREGISTREMENTS_PAR_PERSONNE}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_affichage, "Q/Esc annule",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        cv2.imshow(NOM_FENETRE, frame_affichage)

        touche = lire_touche(30)
        if _est_touche_quitter(touche):
            print("[ANNULE] Capture interrompue.")
            return

        maintenant = time.time()
        if maintenant - dernier_temps < 0.15:
            continue
        dernier_temps = maintenant
        tentatives += 1

        try:
            vecteur = extraire_vecteur_depuis_frame(frame)
        except Exception as exc:
            print(f"[ENREGISTREMENT] Frame ignoree: {exc}")
            continue

        if vecteur is None:
            continue

        enregistrer_personne(nom, vecteur)
        personnes.append(nom)
        vecteurs.append(vecteur)
        captures += 1
        print(f"[OK] Exemple {captures}/{NB_ENREGISTREMENTS_PAR_PERSONNE} enregistre.")

    if captures == 0:
        print("[ERREUR] Aucun visage detecte. Essayez avec plus de lumiere.")
    else:
        print(f"[OK] {captures} exemples ajoutes pour '{nom}'.")
    return

    """
    Mode E : capture le visage actuel depuis la webcam,
    calcule son vecteur 30D, demande le nom en console,
    puis l'ajoute au fichier dataset.csv.

    Paramètres
    ----------
    cap       : objet VideoCapture OpenCV (flux webcam ouvert)
    personnes : liste des noms (modifiée en place)
    vecteurs  : liste des vecteurs (modifiée en place)
    """
    print("\n[MODE ENREGISTREMENT]")
    print("Placez le visage devant la webcam. Appuyez sur ENTRÉE pour capturer...")
    input()   # Pause — l'utilisateur positionne son visage

    ret, frame = cap.read()
    if not ret:
        print("[ERREUR] Impossible de lire la webcam.")
        return

    # Pipeline de détection
    visage = detecter_et_aligner(frame)
    if visage is None:
        print("[ERREUR] Aucun visage détecté. Réessayez.")
        return

    contour = appliquer_snake(visage)
    vecteur = calculer_vecteur_30D(visage, contour)

    # Demander le nom de la personne
    nom = input("Entrez le nom de la personne : ").strip()
    if not nom:
        print("[ANNULÉ] Nom vide.")
        return

    # Sauvegarder dans le CSV
    enregistrer_personne(nom, vecteur)

    # Mettre à jour les listes en mémoire (sans relire tout le CSV)
    personnes.append(nom)
    vecteurs.append(vecteur)
    print(f"[OK] {nom} enregistré(e) avec succès.")


def main():
    """
    Fonction principale — boucle temps réel.

    Flux d'exécution
    ----------------
    1. Charger le dataset existant depuis dataset.csv
    2. Ouvrir la webcam
    3. Lancer la fenêtre Tkinter (porte) dans un thread séparé
    4. Boucle : afficher le flux webcam en continu
       - Touche I → identifier le visage actuel
       - Touche E → enregistrer une nouvelle personne
       - Touche Q → quitter proprement
    """
    print("=" * 60)
    print("  SYSTÈME D'IDENTIFICATION FACIALE — Vision Artificielle")
    print("=" * 60)
    print("  Touches : I = Identifier | E = Enregistrer | Q/Esc = Quitter")
    print("=" * 60)

    # --- Chargement du dataset CSV ---
    personnes, vecteurs = charger_dataset()
    print(f"[INFO] Dataset chargé : {len(set(personnes))} personnes, "
          f"{len(personnes)} enregistrements.")

    # --- Ouverture de la webcam (index 0 = webcam principale) ---
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(NOM_FENETRE, cv2.WINDOW_NORMAL)
    if not cap.isOpened():
        print("[ERREUR FATALE] Webcam introuvable. Vérifiez la connexion.")
        cv2.destroyAllWindows()
        return

    # --- Lancer la simulation de porte dans un thread parallèle ---
    # Tkinter ne peut pas tourner dans le même thread qu'OpenCV
    porte = SimulateurPorte()
    thread_porte = threading.Thread(target=porte.lancer, daemon=True)
    thread_porte.start()

    # --- Variables d'état ---
    frame_affichage = None   # Frame courante (potentiellement avec overlay)
    resultat_courant = None  # Dernier résultat d'identification

    print("\n[DÉMARRAGE] Flux webcam actif...")

    # ==============================================================================
    # BOUCLE PRINCIPALE TEMPS RÉEL
    # ==============================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Perte du flux webcam.")
            break

        frame_affichage = frame.copy()

        # Si un résultat d'identification est disponible, afficher l'overlay
        if resultat_courant is not None:
            nom, distance, confiance, top3 = resultat_courant
            acces = (distance <= SEUIL_DECISION)
            frame_affichage = afficher_overlay(
                frame_affichage, nom, distance, confiance, top3, acces
            )
        else:
            # Instructions de base quand aucune identification n'est en cours
            cv2.putText(frame_affichage,
                        "I: identifier | E: enregistrer | Q/Esc: quitter",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        cv2.imshow(NOM_FENETRE, frame_affichage)
        if not _fenetre_ouverte(NOM_FENETRE):
            print("\n[FERMETURE] Fenetre fermee.")
            break

        # --- Lecture des touches clavier (attente 1ms) ---
        touche = lire_touche(30)

        # TOUCHE I : Lancer l'identification
        if touche == ord('i') or touche == ord('I'):
            print("\n[IDENTIFICATION] Analyse en cours...")
            resultat, quitter = scanner_identification(cap, personnes, vecteurs)
            if quitter:
                print("\n[FERMETURE] Au revoir.")
                break

            if resultat is None:
                print("[INFO] Aucun visage détecté ou dataset vide.")
                resultat_courant = None
            else:
                nom, distance, confiance, top3 = resultat
                resultat_courant = resultat
                acces = (distance <= SEUIL_DECISION)

                print(f"  → Identifié  : {nom}")
                print(f"  → Distance   : {distance:.4f}")
                print(f"  → Confiance  : {confiance:.1f}%")
                print(f"  → Décision   : {'ACCÈS AUTORISÉ' if acces else 'ACCÈS REFUSÉ'}")
                print(f"  → Top 3      : {[(n, f'{d:.3f}') for n, d in top3]}")

                # Mettre à jour la simulation de porte
                if acces:
                    porte.ouvrir_porte(nom)
                else:
                    porte.fermer_porte()

        # TOUCHE E : Enregistrer une nouvelle personne
        elif touche == ord('e') or touche == ord('E'):
            porte.en_attente()   # Porte en attente pendant l'enregistrement
            mode_enregistrement(cap, personnes, vecteurs)
            resultat_courant = None   # Réinitialiser l'affichage

        # TOUCHE V : Évaluer les performances (bonus)
        elif touche == ord('v') or touche == ord('V'):
            print("\n[ÉVALUATION] Calcul des métriques sur le dataset...")
            evaluer_depuis_dataset(personnes, vecteurs, seuil=SEUIL_DECISION)

        # TOUCHE Q : Quitter
        elif _est_touche_quitter(touche):
            print("\n[FERMETURE] Au revoir.")
            break

    # --- Nettoyage des ressources ---
    cap.release()
    cv2.destroyAllWindows()
    try:
        porte.arreter()
    except Exception:
        pass
    # Le thread Tkinter se termine automatiquement (daemon=True)


# ==============================================================================
# Point d'entrée Python standard
# ==============================================================================
if __name__ == "__main__":
    main()
