#Iarina NISTOR 21210925
#Explications et Interprétation des Résultats dans le fichier .ipynb

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from scipy.spatial import KDTree 
import time


def multi_intercrop(N, L, rmin, rmax):
    """
    Génère N plantes (cercles) placées aléatoirement sur un carré de longueur L,
    en s'assurant que les cercles ne se chevauchent pas.
    
    Args:
    N (int): Nombre de plantes à générer.
    L (float): Longueur du côté du carré.
    rmin (float): Rayon minimum des plantes.
    rmax (float): Rayon maximum des plantes.
    
    Returns:
    list of dict: Liste de dictionnaires représentant les plantes, chaque dictionnaire contenant:
                  - 'pos': [posx, posy] (coordonnées du centre du cercle),
                  - 'r': rayon du cercle.
    """
    plantes = []
    
    for _ in range(N):
        position_valide = False
        while not position_valide:
            # Générer un rayon aléatoire pour la plante
            rayon = np.random.random() * (rmax - rmin) + rmin
            
            # Définir les bornes pour la position du centre du cercle
            borne_basse = rayon
            borne_haute = L - rayon
            
            # Générer une position aléatoire (posx, posy) dans la zone valide
            posx = np.random.random() * (borne_haute - borne_basse) + borne_basse
            posy = np.random.random() * (borne_haute - borne_basse) + borne_basse
            
            # Vérifier s'il y a un chevauchement avec les plantes existantes
            position_valide = True
            for plante in plantes:
                # Calculer la distance entre les centres des deux cercles
                distance = np.sqrt((posx - plante['pos'][0]) ** 2 + (posy - plante['pos'][1]) ** 2)
                
                # Si la distance est inférieure à la somme des rayons, il y a chevauchement
                if distance < (rayon + plante['r']):
                    position_valide = False
                    break
            
            # Si la position est valide (pas de chevauchement), ajouter la plante à la liste
            if position_valide:
                plante_dict = {'pos': [posx, posy], 'r': rayon}
                plantes.append(plante_dict)
    
    return plantes

def fig_field(plants, L):
    """
    Affiche la représentation graphique du champ avec les plantes (disques) sur un carré de taille L.
    
    Args:
    plants (list of dict): Liste de plantes, chaque plante est un dictionnaire contenant:
                           - 'pos': [x, y] (coordonnées de la plante),
                           - 'r': rayon du disque.
    L (float): La longueur du côté du carré représentant le champ.
    """
    # Créer une figure et un axe
    fig, ax = plt.subplots()

    # Définir les limites de l'axe (le carré)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')  # Assurez-vous que l'échelle des axes est la même pour x et y

    # Modifier la couleur de fond en beige
    ax.set_facecolor('#F5F5DC')  # Code hexadécimal pour beige

    # Ajouter chaque plante comme un cercle
    for plant in plants:
        pos = plant['pos']
        r = plant['r']
        
        # Générer une couleur aléatoire pour chaque cercle (composants RGB)
        color = np.random.rand(3,)  # Trois valeurs entre 0 et 1 pour la couleur RGB

        # Créer un cercle pour la plante à la position pos avec le rayon r
        circle = patches.Circle(pos, r, edgecolor='black', facecolor=color, alpha=0.9)  # Alpha réduit pour plus d'opacité

        # Ajouter le cercle à l'axe
        ax.add_patch(circle)

    # Ajouter des labels pour l'axe
    ax.set_title("Représentation graphique des plantes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Supprimer la grille
    ax.grid(False)

    # Afficher la figure
    plt.show()

def monocrop(N,L,r):
    """
    Génère N plantes (cercles) placées aléatoirement sur un carré de longueur L,
    en s'assurant que les cercles ne se chevauchent pas.

    Args:
    N (int) : Nombre de plantes à générer.
    L (float) : Longueur du côté du carré représentant le champ.
    r (float) : Rayon des plantes.

    Returns:
    tuple : Temps d'exécution de l'algorithme et liste des plantes générées, où chaque plante est
            représentée par un dictionnaire avec la position 'pos' et le rayon 'r'.
    """
    temps_debut = time.time()
    plantes = multi_intercrop(N,L,r,r)
    temps_fin = time.time()
    return temps_fin-temps_debut,plantes

def monocrop_KD(N, L, r, max_attempts=100):
    """
    Génère N plantes avec un rayon r, placées de manière aléatoire sur un carré de taille L,
    en utilisant un KD-Tree pour accélérer la vérification des chevauchements.
    
    Args:
    N (int) : Nombre de plantes à générer.
    L (float) : Longueur du côté du carré représentant le champ.
    r (float) : Rayon des plantes.
    max_attempts (int) : Nombre maximum de tentatives pour trouver une position valide pour une plante.

    Returns:
    tuple : Temps d'exécution de l'algorithme et liste des plantes générées, où chaque plante est
            représentée par un dictionnaire avec la position 'pos' et le rayon 'r'.
    """
    plantes = []  # Liste pour stocker les plantes
    positions = []  # Liste pour stocker les positions des plantes
    start_time = time.time()  # Enregistrement du temps de début de l'exécution

    # Créer un KDTree vide au départ
    tree = KDTree(np.empty((0, 2)))  # Un arbre vide, il sera mis à jour au fur et à mesure

    # Générer N plantes
    for _ in range(N):
        position_valide = False  # Variable pour vérifier si la position est valide
        attempts = 0
        
        while not position_valide and attempts < max_attempts:
            # Générer une position aléatoire pour la plante
            rayon = r
            borne_basse = rayon
            borne_haute = L - rayon
            posx = np.random.random() * (borne_haute - borne_basse) + borne_basse
            posy = np.random.random() * (borne_haute - borne_basse) + borne_basse

            # Vérifier les voisins dans le KD-Tree
            if len(positions) > 0:
                # Chercher les voisins dans un rayon de 2 * r
                voisins = tree.query([[posx, posy]], k=5, distance_upper_bound=2*r)

                # Si on trouve des voisins, on vérifie la distance avec chaque voisin
                distances, indices = voisins

                # Vérifier qu'aucun voisin n'est trop proche
                position_valide = all(d >= 2 * rayon for d in distances[0])
            
            else:
                # Si aucun voisin, la position est valide
                position_valide = True

            # Si la position est valide, ajouter la plante
            if position_valide:
                plantes.append({'pos': [posx, posy], 'r': rayon})  # Ajouter la plante à la liste
                positions.append([posx, posy])  # Ajouter la position dans la liste des positions
                
                # Mettre à jour le KDTree avec la nouvelle position
                tree = KDTree(np.array(positions))  # Recréer l'arbre avec la nouvelle liste de positions

            attempts += 1

        if attempts >= max_attempts:
            print(f"Attention : Impossible de placer une plante après {max_attempts} tentatives.")

    end_time = time.time()  # Enregistrement du temps de fin de l'exécution
    return end_time - start_time, plantes  # Retourne le temps d'exécution et la liste des plantes



def distance(x1, y1, x2, y2):
    """
    Calcule la distance euclidienne entre deux points dans un plan 2D.

    Args:
    x1 (float): Coordonnée x du premier point.
    y1 (float): Coordonnée y du premier point.
    x2 (float): Coordonnée x du deuxième point.
    y2 (float): Coordonnée y du deuxième point.

    Returns:
    float: La distance euclidienne entre les deux points.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def placer_disques_aleatoires(L, r, max_tentatives=10000):
    """
    Place autant de disques que possible dans un carré de côté L, chacun de rayon r, 
    sans chevauchement entre les disques. La fonction limite le nombre de tentatives 
    pour éviter les boucles infinies.

    Args:
    L (float): Longueur du côté du carré dans lequel placer les disques.
    r (float): Rayon de chaque disque.
    max_tentatives (int, optional): Nombre maximal de tentatives pour placer 
                                     un disque sans chevauchement. Par défaut à 10,000.

    Returns:
    list of tuples: Liste des positions des centres des disques placés, 
                    chaque position étant un tuple (x, y) des coordonnées.
    """
    positions = []  # Liste des positions des centres des disques
    tentatives = 0
    while tentatives < max_tentatives:
        # Générer des coordonnées aléatoires pour le centre d'un nouveau disque
        x, y = np.random.uniform(r, L - r), np.random.uniform(r, L - r)
        
        # Vérifier si le disque chevauche un autre disque déjà placé
        chevauchement = any(distance(x, y, px, py) < 2 * r for px, py in positions)
        
        if not chevauchement:
            positions.append((x, y))  # Ajouter la position si pas de chevauchement
        
        tentatives += 1
    
    return positions


def calculer_densite(positions, r, L):
    """
    Calcule la densité d'une configuration de disques dans un carré de côté L.
    La densité est définie comme le rapport de l'aire totale occupée par les disques
    à l'aire du carré.

    Args:
    positions (list of tuples): Liste des positions des centres des disques, chaque position
                                étant un tuple (x, y) de coordonnées.
    r (float): Rayon de chaque disque.
    L (float): Longueur du côté du carré contenant les disques.

    Returns:
    float: La densité de la configuration des disques, c'est-à-dire le rapport
           entre l'aire totale des disques et l'aire du carré.
    """
    nb_disques = len(positions)
    aire_disques = nb_disques * np.pi * r ** 2  # Aire occupée par les disques
    aire_carre = L ** 2  # Aire du carré
    return aire_disques / aire_carre


def simuler_plantation_aleatoire(L, r, nb_simulations=100):
    """
    Simule plusieurs configurations aléatoires de disques dans un carré de côté L,
    et calcule la densité moyenne de placement ainsi que l'écart-type associé.

    Args:
    L (float): Longueur du côté du carré où les disques sont placés.
    r (float): Rayon de chaque disque.
    nb_simulations (int, optional): Nombre de simulations à effectuer pour estimer
                                     la densité moyenne. Par défaut à 100.

    Returns:
    tuple: Un tuple contenant trois éléments :
        - densite_moyenne (float) : l'estimation ponctuelle de la densité moyenne des disques dans le carré.
        - ecart_type_densite (float) :l'estimation ponctuelle de l'écart-type des densités calculées sur les simulations.
        - densites (list of float) : Liste des densités obtenues pour chaque simulation.
    """
    densites = []
    
    for _ in range(nb_simulations):
        positions = placer_disques_aleatoires(L, r)
        densite = calculer_densite(positions, r, L)
        densites.append(densite)
        #print("simulation")
    
    # Calcul de l'estimation de moyenne  et de l'écart-type
    densite_moyenne = np.mean(densites)
    ecart_type_densite = np.std(densites)*nb_simulations/(nb_simulations-1)
    
    return densite_moyenne, ecart_type_densite, densites

def dynamic_random_planting(planting_rate, Rmax, th, L=100):
    """
    Simule la plantation dynamique de plantes avec un taux de plantation Poisson
    et une croissance linéaire des rayons.
    
    Args:
    planting_rate (float): Taux de plantation en plantes par unité de temps.
    Rmax (float): Rayon maximum que les plantes peuvent atteindre.
    th (float): Temps après lequel les plantes sont enlevées.
    
    Returns:
    - plants (list): Liste de dictionnaires, chaque dictionnaire contient:
                      - 'pos': Position de la plante,
                      - 't_plant': Temps de plantation de la plante,
                      - 'r_max': Rayon maximal de la plante,
                      - 'alpha': Coefficient de croissance de la plante.
    - ps (list): Liste des temps de plantation des plantes.
    """
    plants = []
    ps = []  # Temps de plantation
    alpha = Rmax / th  # Croissance linéaire du rayon
    
    # Simulation du processus de plantation
    t = 0  # Temps initial
    while t < th:
        t_next = np.random.exponential(1/planting_rate)
        t += t_next
        
        # Générer une position aléatoire
        pos = np.random.uniform(0, L, 2)
        
        # Vérification de la collision avec les plantes déjà plantées
        collision = False
        for plant in plants:
            dist = np.linalg.norm(np.array(plant['pos']) - np.array(pos))
            r_growth_max = alpha * (t - plant['t_plant'])
            if dist < r_growth_max:
                collision = True
                break
        
        if not collision:
            # Ajouter la nouvelle plante
            plants.append({
                'pos': pos,
                't_plant': t,
                'r_max': Rmax,
                'alpha': alpha
            })
            ps.append(t)
    
    return plants, ps

def fig_dynamic(plants, snapshot_filename, animation_filename, L=100, th=30):
    """
    Visualise l'état du champ au temps t (snapshot) et génère une animation.
    
    Args:
    plants (list): Liste de plantes avec leurs propriétés (position, temps de plantation, etc.).
    snapshot_filename (str): Nom du fichier pour enregistrer un snapshot.
    animation_filename (str): Nom du fichier pour enregistrer l'animation.
    L (float): Taille du champ.
    th (float): Temps maximal de plantation.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    
    # Fonction pour dessiner l'état à un instant t
    def draw_snapshot(t):
        ax.clear()
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.set_title(f"Snapshot à t = {t:.2f}")
        
        for plant in plants:
            t_plant = plant['t_plant']
            if t >= t_plant:
                r_growth = plant['alpha'] * (t - t_plant)
                if r_growth > plant['r_max']:
                    r_growth = plant['r_max']
                ax.add_patch(plt.Circle(plant['pos'], r_growth, color='green', alpha=0.5))
        
        plt.pause(0.1)
    
    # Affichage du snapshot initial
    draw_snapshot(th)
    plt.savefig(snapshot_filename)
    
    # Animation
    def animate(i):
        t = i * th / 100  # Évolution du temps pour l'animation
        draw_snapshot(t)
    
    ani = animation.FuncAnimation(fig, animate, frames=100, interval=100, repeat=False)
    
    # Sauvegarde de l'animation en .mp4
    ani.save(animation_filename, writer='ffmpeg', fps=10)



