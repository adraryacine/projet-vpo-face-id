"""
==============================================================================
FICHIER : porte.py
ROLE    : Simulation graphique d'une porte sécurisée via Tkinter.
          Représente visuellement la décision d'identification :
            - Accès autorisé → porte s'ouvre, voyant VERT
            - Accès refusé  → porte reste fermée, flash ROUGE

ARCHITECTURE
------------
La fenêtre Tkinter tourne dans un thread séparé de la boucle OpenCV
(voir main.py). Pour communiquer entre les threads, on utilise des méthodes
thread-safe : after() de Tkinter qui planifie l'exécution dans le thread GUI.

ANIMATION
---------
L'ouverture de la porte est animée : on modifie progressivement la largeur
visible de la porte (effet de "swing" en perspective) à l'aide d'after()
qui rappelle une fonction toutes les N millisecondes.
==============================================================================
"""

import tkinter as tk
import math


# ==============================================================================
# CONSTANTES DE STYLE
# ==============================================================================
COULEUR_FOND         = '#1a1a2e'   # Bleu nuit (fond de la fenêtre)
COULEUR_PORTE_FERMEE = '#8B7355'   # Brun bois
COULEUR_PORTE_OUVERTE= '#5a3e28'   # Brun foncé (côté visible en perspective)
COULEUR_MUR          = '#2d2d2d'   # Gris foncé (mur autour de la porte)
COULEUR_INTERIEUR    = '#1a3a1a'   # Vert sombre (intérieur visible quand ouvert)
COULEUR_POIGNEE      = '#FFD700'   # Or (poignée de porte)
COULEUR_VOYANT_VERT  = '#00FF00'
COULEUR_VOYANT_ROUGE = '#FF2200'
COULEUR_VOYANT_ATTENTE = '#FF8800'  # Orange = en attente


class SimulateurPorte:
    """
    Fenêtre Tkinter simulant une porte de sécurité biométrique.

    Méthodes publiques
    ------------------
    ouvrir_porte(nom) : afficher "accès autorisé" + animer l'ouverture
    fermer_porte()    : afficher "accès refusé" + flash rouge
    en_attente()      : état neutre (attente d'une identification)
    lancer()          : démarrer la boucle Tkinter (bloquant — à appeler dans un thread)
    """

    def __init__(self):
        """Initialise la fenêtre et tous les widgets Tkinter."""
        self.root = tk.Tk()
        self.root.title("Porte Sécurisée — Système d'Identification Faciale")
        self.root.configure(bg=COULEUR_FOND)
        self.root.resizable(False, False)

        # --- Dimensions de la fenêtre ---
        self.largeur_fenetre = 480
        self.hauteur_fenetre = 560
        self.root.geometry(f"{self.largeur_fenetre}x{self.hauteur_fenetre}")

        # --- État interne ---
        self._etat           = 'attente'   # 'attente' | 'ouvert' | 'ferme'
        self._angle_porte    = 0           # Angle d'ouverture (0=fermé, 90=ouvert)
        self._animation_active = False

        # --- Construction de l'interface ---
        self._creer_interface()

        # --- État initial ---
        self.en_attente()

    # ==========================================================================
    # CONSTRUCTION DE L'INTERFACE
    # ==========================================================================

    def _creer_interface(self):
        """Crée tous les widgets de la fenêtre."""
        # Bandeau d'état (haut)
        self.frame_etat = tk.Frame(self.root, bg=COULEUR_FOND)
        self.frame_etat.pack(fill='x', pady=(10, 0))

        self.label_etat = tk.Label(
            self.frame_etat,
            text="En attente...",
            font=('Consolas', 14, 'bold'),
            fg='white',
            bg=COULEUR_FOND
        )
        self.label_etat.pack()

        self.label_sous_etat = tk.Label(
            self.frame_etat,
            text="Lancez l'identification (touche I).",
            font=('Consolas', 10),
            fg='#888888',
            bg=COULEUR_FOND
        )
        self.label_sous_etat.pack()

        # Canvas principal (dessin de la porte)
        self.canvas = tk.Canvas(
            self.root,
            width=self.largeur_fenetre,
            height=380,
            bg=COULEUR_FOND,
            highlightthickness=0   # Pas de bordure autour du canvas
        )
        self.canvas.pack(pady=10)

        # Bandeau du bas avec les noms des membres
        frame_bas = tk.Frame(self.root, bg=COULEUR_FOND)
        frame_bas.pack(fill='x', side='bottom', pady=5)

        for nom in ["Yanis", "Khadidja", "Sarah", "Mehdi"]:
            tk.Label(
                frame_bas,
                text=nom,
                font=('Consolas', 10, 'bold'),
                fg='white',
                bg='#333355',
                width=10,
                relief='raised',
                pady=4
            ).pack(side='left', padx=5, expand=True)

    # ==========================================================================
    # DESSIN DE LA PORTE
    # ==========================================================================

    def _dessiner_porte(self, angle_ouverture=0, couleur_voyant=COULEUR_VOYANT_ATTENTE):
        """
        Redessine la porte sur le canvas selon l'angle d'ouverture.

        L'effet de perspective 3D est simulé en réduisant la largeur
        de la porte selon l'angle d'ouverture (projection orthogonale).

        angle_ouverture : float [0, 90]
          - 0  → porte fermée (vue de face, largeur maximale)
          - 90 → porte ouverte (vue de côté, largeur minimale)
        """
        self.canvas.delete("all")   # Effacer le dessin précédent

        W, H = self.largeur_fenetre, 370

        # --- Mur ---
        self.canvas.create_rectangle(0, 0, W, H, fill=COULEUR_MUR, outline='')

        # --- Cadre de porte (encadrement dans le mur) ---
        # Position du cadre dans la fenêtre
        cadre_x1, cadre_y1 = W//2 - 80, 40
        cadre_x2, cadre_y2 = W//2 + 80, 340
        cadre_ep = 8   # Épaisseur du cadre

        self.canvas.create_rectangle(
            cadre_x1 - cadre_ep, cadre_y1 - cadre_ep,
            cadre_x2 + cadre_ep, cadre_y2 + cadre_ep,
            fill='#555555', outline='#333333', width=2
        )

        # --- Intérieur (visible quand la porte est ouverte) ---
        self.canvas.create_rectangle(
            cadre_x1, cadre_y1, cadre_x2, cadre_y2,
            fill=COULEUR_INTERIEUR, outline=''
        )

        # --- Porte (avec effet de perspective selon l'angle) ---
        # La largeur visible de la porte diminue avec l'angle (cos de l'angle)
        # angle=0 → cos(0)=1 → largeur maximale (porte de face)
        # angle=90 → cos(90)=0 → largeur nulle (porte vue de côté)
        largeur_max = cadre_x2 - cadre_x1   # 160 pixels
        facteur_perspective = math.cos(math.radians(angle_ouverture))
        largeur_visible = max(4, int(largeur_max * facteur_perspective))

        # La porte s'ouvre vers la droite : elle commence à cadre_x1
        px1 = cadre_x1
        px2 = cadre_x1 + largeur_visible
        py1 = cadre_y1
        py2 = cadre_y2

        # Couleur de la porte : s'assombrit légèrement en s'ouvrant
        if angle_ouverture < 45:
            coul_porte = COULEUR_PORTE_FERMEE
        else:
            coul_porte = COULEUR_PORTE_OUVERTE

        self.canvas.create_rectangle(
            px1, py1, px2, py2,
            fill=coul_porte,
            outline='#3a2a1a',
            width=2
        )

        # Lignes de bois (texture simulée)
        if largeur_visible > 15:
            for yy in range(py1 + 30, py2, 50):
                self.canvas.create_line(
                    px1 + 5, yy, px2 - 5, yy,
                    fill='#7a6345', width=1
                )

        # Poignée de porte (uniquement si la porte est assez large)
        if largeur_visible > 30:
            poignee_x = px1 + int(largeur_visible * 0.75)
            poignee_y = (py1 + py2) // 2
            self.canvas.create_oval(
                poignee_x - 8, poignee_y - 8,
                poignee_x + 8, poignee_y + 8,
                fill=COULEUR_POIGNEE, outline='#cc9900'
            )

        # Sol (ligne de bas)
        self.canvas.create_line(
            cadre_x1 - cadre_ep, cadre_y2 + cadre_ep,
            cadre_x2 + cadre_ep, cadre_y2 + cadre_ep,
            fill='#888888', width=3
        )

        # --- Voyant lumineux ---
        voyant_x = W // 2
        voyant_y = 15
        rayon    = 12

        # Halo (cercle plus grand, semi-transparent simulé par superposition)
        self.canvas.create_oval(
            voyant_x - rayon - 5, voyant_y - rayon - 5,
            voyant_x + rayon + 5, voyant_y + rayon + 5,
            fill='', outline=couleur_voyant, width=2
        )
        # Voyant principal
        self.canvas.create_oval(
            voyant_x - rayon, voyant_y - rayon,
            voyant_x + rayon, voyant_y + rayon,
            fill=couleur_voyant, outline='#222222', width=2
        )

    # ==========================================================================
    # ANIMATION D'OUVERTURE
    # ==========================================================================

    def _animer_ouverture(self, angle_actuel=0, nom=""):
        """
        Fonction récursive d'animation appelée par Tkinter via after().

        Incrémente l'angle d'ouverture de 5° à chaque appel (toutes les 30ms)
        jusqu'à 85° (porte complètement ouverte).
        """
        if angle_actuel >= 85:
            # Animation terminée
            self._dessiner_porte(85, COULEUR_VOYANT_VERT)
            self._animation_active = False
            return

        # Dessiner la porte à l'angle courant
        self._dessiner_porte(angle_actuel, COULEUR_VOYANT_VERT)

        # Planifier la prochaine frame (30ms plus tard)
        # after() est thread-safe dans Tkinter
        self.root.after(
            30,
            lambda: self._animer_ouverture(angle_actuel + 5, nom)
        )

    def _animer_flash_rouge(self, nb_flashs=3, etat_allume=True):
        """Animation de flash rouge pour le refus d'accès."""
        if nb_flashs <= 0:
            self._dessiner_porte(0, COULEUR_VOYANT_ROUGE)
            return

        couleur = COULEUR_VOYANT_ROUGE if etat_allume else COULEUR_FOND
        self._dessiner_porte(0, couleur)
        self.root.after(
            200,
            lambda: self._animer_flash_rouge(nb_flashs - 1, not etat_allume)
        )

    # ==========================================================================
    # MÉTHODES PUBLIQUES (appelées depuis main.py dans un autre thread)
    # ==========================================================================

    def ouvrir_porte(self, nom=""):
        """
        Déclenche l'animation d'ouverture et met à jour les labels.
        Thread-safe grâce à after().
        """
        def _action():
            self._etat = 'ouvert'
            self.label_etat.config(
                text=f"✓ ACCÈS AUTORISÉ",
                fg=COULEUR_VOYANT_VERT
            )
            self.label_sous_etat.config(
                text=f"Bienvenue, {nom} !",
                fg=COULEUR_VOYANT_VERT
            )
            if not self._animation_active:
                self._animation_active = True
                self._animer_ouverture(angle_actuel=0, nom=nom)

        self.root.after(0, _action)

    def fermer_porte(self):
        """
        Déclenche le flash rouge et met à jour les labels.
        Thread-safe grâce à after().
        """
        def _action():
            self._etat = 'ferme'
            self.label_etat.config(
                text="✗ ACCÈS REFUSÉ",
                fg=COULEUR_VOYANT_ROUGE
            )
            self.label_sous_etat.config(
                text="Personne non reconnue.",
                fg=COULEUR_VOYANT_ROUGE
            )
            self._animer_flash_rouge(nb_flashs=6)

        self.root.after(0, _action)

    def en_attente(self):
        """
        Remet la porte en état neutre (fermée, voyant orange).
        Thread-safe grâce à after().
        """
        def _action():
            self._etat = 'attente'
            self.label_etat.config(
                text="En attente...",
                fg='white'
            )
            self.label_sous_etat.config(
                text="Lancez l'identification (touche I).",
                fg='#888888'
            )
            self._dessiner_porte(0, COULEUR_VOYANT_ATTENTE)

        self.root.after(0, _action)

    def arreter(self):
        """Ferme proprement la fenetre Tkinter depuis main.py."""
        try:
            self.root.after(0, self.root.destroy)
        except tk.TclError:
            pass

    def lancer(self):
        """
        Démarre la boucle principale Tkinter (bloquante).
        À appeler dans un thread séparé depuis main.py.
        """
        self.root.mainloop()
