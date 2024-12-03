# Iarina NISTOR 21210925
#Explications et Interprétation des Résultats dans le fichier .ipynb

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns


def get_cropping_cycles_database(csv_file):
    """
    Charge une base de données de cycles de culture à partir d'un fichier CSV et renvoie un DataFrame.

    Args:
    csv_file (str): Le chemin du fichier CSV contenant les données des cycles de culture.

    Returns:
    pandas.DataFrame: Un DataFrame contenant les données extraites du fichier CSV.
    """
    return pd.read_csv(csv_file)

def get_criteria(csv_file):
    """
    Charge une base de données de criterias de culture à partir d'un fichier CSV et renvoie un DataFrame.

    Args:
    csv_file (str): Le chemin du fichier CSV contenant les données des criterias de culture.

    Returns:
    pandas.DataFrame: Un DataFrame contenant les données extraites du fichier CSV.
    """
    return pd.read_csv(csv_file)

def creation_nom_colonne(month, debut, nom=1):
    """
    Crée un nom de colonne basé sur un mois et un préfixe donné.

    Args:
    month (int): Le mois sous forme d'entier (0 pour janvier, 1 pour février, etc.).
    debut (str): Le préfixe ou la première partie du nom de la colonne (ex : '2024_').
    nom (int, optionnel): Si égal à 1, retourne le nom du mois sous forme de texte (ex: 'Jan'). Si égal à autre chose, retourne le mois sous forme numérique (ex: '1'). Par défaut, égal à 1.

    Returns:
    str: Un nom de colonne formé du préfixe `debut` et du mois sous forme de texte ou numérique selon la valeur de `nom`.
    """
    # Dictionnaire des mois
    dico = {0:'Jan', 1:'Feb', 2:'Mar', 3:'Apr', 4:'May', 5:'Jun', 6:'Jul', 7:'Aug', 8:'Sep', 9:'Oct', 10:'Nov', 11:'Dec'}
    
    # Déterminer le mois sous forme textuelle ou numérique
    if nom == 1:
        m = dico[month]  # Nom du mois sous forme de texte
    else:
        m = str(month + 1)  # Mois sous forme numérique
    
    # Retourner le nom de la colonne
    return debut + m



def choose_cycle(category, month, df_cycles, df_criteria):
    """
    Choisit un cycle de culture pour une catégorie donnée en fonction des critères spécifiés et des cycles de culture disponibles.

    Args:
    category (str): La catégorie de culture pour laquelle un cycle doit être choisi (par exemple, 'Legume').
    month (int): Le mois sous forme d'entier (0 pour janvier, 1 pour février, etc.), utilisé pour identifier la colonne correspondante.
    df_cycles (DataFrame): Un DataFrame contenant les cycles de culture, incluant les informations sur les catégories et les cycles de culture disponibles.
    df_criteria (DataFrame): Un DataFrame contenant les critères de culture, incluant les informations sur le nombre minimal de cultures et d'autres conditions.

    Returns:
    int or bool: L'ID du cycle de culture sélectionné si les critères sont remplis, sinon retourne `False` si aucune culture n'est possible.
    """
    # Générer le nom de la colonne correspondant aux ventes pour le mois spécifié
    nom_colonne_sale = creation_nom_colonne(month, 'Sale_')

    # Filtrer les cycles de culture en fonction de la catégorie et des ventes disponibles pour le mois
    crop_cycles = df_cycles[(df_cycles['Crop category'] == category) & (df_cycles[nom_colonne_sale] == 1)]
    longueur = crop_cycles.shape[0]

    # Si aucun cycle de culture disponible, retourner False
    if longueur == 0:
        return False
    
    # Choisir un cycle de culture au hasard parmi ceux disponibles
    row = np.random.randint(0, longueur)

    # Générer le nom de la colonne pour le nombre minimal de cultures
    nom_colonne_crops = creation_nom_colonne(month, 'Minimal number of crops_', 0)

    # Filtrer les critères pour la catégorie de culture, le climat 'Mild' et le marketing '12M'
    crop_criteria = df_criteria[((df_criteria['Crop category'] == category) & (df_criteria['Climate'] == 'Mild')) & (df_criteria['Marketing'] == '12M')]

    # Vérifier si le nombre minimal de cultures est supérieur à zéro
    if crop_criteria.iloc[0][nom_colonne_crops] == 0:
        return False  # Aucun cycle de culture possible

    # Retourner l'ID du cycle de culture choisi
    return crop_cycles.iloc[row]['ID']


def update_shares(cycle, cycles, cropping_cycles):
    """
    Met à jour les parts de culture (shares) dans un dictionnaire en fonction de l'ID de cycle et des informations des cycles de culture.

    Args:
    cycle (DataFrame): Un DataFrame contenant les informations sur le cycle de culture actuel. Doit inclure les colonnes 'ID', 'Shmax', 'Shmin', et 'Harvest_last'.
    cycles (dict): Un dictionnaire où les clés sont les ID de cycles de culture et les valeurs sont les parts (shares) actuelles pour chaque cycle.
    cropping_cycles (DataFrame): Un DataFrame des cycles de culture, utilisé pour vérifier l'ID et d'autres informations nécessaires. (Non utilisé directement dans la fonction mais peut être utile dans des versions futures)

    Returns:
    dict: Le dictionnaire mis à jour avec les nouvelles valeurs de parts (shares) pour chaque cycle de culture.
    """
    cycle_id = cycle.iloc[0]['ID']  # Récupérer l'ID du cycle de culture actuel

    if cycle_id in cycles:  # Si l'ID du cycle existe déjà dans le dictionnaire
        shmax = cycle.iloc[0]['Shmax']  # Récupérer la valeur maximale des parts
        if cycles[cycle_id] < shmax:  # Si les parts actuelles sont inférieures à la valeur maximale
            cycles[cycle_id] += 1  # Incrémenter les parts
    else:  # Si l'ID du cycle n'existe pas dans le dictionnaire
        shmin = cycle.iloc[0]['Shmin']  # Récupérer la valeur minimale des parts
        harvest_last = cycle.iloc[0]['Harvest_last']  # Récupérer le dernier coefficient de récolte
        cycles[cycle_id] = shmin * harvest_last  # Calculer et affecter les parts initiales en fonction du minimum et du coefficient de récolte

    return cycles  # Retourner le dictionnaire mis à jour des parts


def spread_shares(cycle, cycles, cropping_cycles):
    """
    Répartit les parts de culture (shares) sur 12 mois en fonction des mois où la culture est vendue.

    Args:
    cycle (DataFrame): Un DataFrame contenant les informations sur le cycle de culture actuel. Ce DataFrame doit inclure les colonnes 'ID' et les colonnes 'Sale_<month>' indiquant si la culture est vendue ce mois-là (1 pour vendu, 0 pour non vendu).
    cycles (dict): Un dictionnaire où les clés sont les ID de cycles de culture et les valeurs sont les parts (shares) actuelles pour chaque cycle.
    cropping_cycles (DataFrame): Un DataFrame des cycles de culture. Bien que ce paramètre ne soit pas directement utilisé ici, il peut être utile pour ajouter des logiques supplémentaires dans une version future.

    Returns:
    list: Une liste de 12 éléments représentant les parts de culture réparties sur les 12 mois, où chaque élément est la part mensuelle attribuée à chaque mois en fonction des mois de vente.
    """
    cycle_id = cycle.iloc[0]['ID']  # Récupérer l'ID du cycle de culture actuel
    lots = cycles[cycle_id]  # Obtenir la part actuelle de culture pour ce cycle
    
    count = 0
    # Comptage du nombre de mois où la culture est vendue
    for i in range(12):
        nom_colonne_sale = creation_nom_colonne(i, "Sale_", 1)  # Création du nom de la colonne pour chaque mois
        sale = cycle.iloc[0][nom_colonne_sale]  # Vérifier si la culture est vendue ce mois-là
        if sale == 1:  # Si vendue, incrémenter le compteur
            count += 1
    
    # Calcul de la part mensuelle
    val = np.float32(lots / count) if count != 0 else 0  # Éviter la division par zéro si aucun mois n'est vendu

    l = []
    # Répartir la part mensuelle sur les mois de vente
    for i in range(12):
        nom_colonne_sale = creation_nom_colonne(i, "Sale_", 1)  # Création du nom de la colonne pour chaque mois
        sale = cycle.iloc[0][nom_colonne_sale]
        if sale == 1:  # Si le mois est un mois de vente, ajouter la part mensuelle
            l.append(val)
        else:  # Sinon, ajouter 0 pour ce mois
            l.append(np.float32(0))
    
    return l  # Retourner la liste des parts réparties sur les 12 mois



def check_quant(monthly_shares, criteria, month):
    """
    Vérifie si le critère de quantité est respecté pour un mois donné.

    Args:
    monthly_shares (dict): Un dictionnaire où les clés sont les IDs de cycle et les valeurs sont des listes représentant la quantité de parts (shares) allouées pour chaque mois.
    criteria (DataFrame): Un DataFrame filtré sur la catégorie de légume, contenant les critères de quantité minimale à respecter pour chaque mois. Ce DataFrame doit avoir des colonnes de type 'Minimal quantity of shares_<month>'.
    month (int): Le mois pour lequel vérifier le critère de quantité (0 pour janvier, 1 pour février, etc.).

    Returns:
    bool: `True` si le critère de quantité minimale est respecté pour le mois spécifié, sinon `False`.
    """
    
    # Filtrer le DataFrame des critères pour la catégorie de marketing '12M' et le climat 'Mild'
    criteria = criteria[(criteria['Marketing'] == '12M') & (criteria['Climate'] == 'Mild')]
    
    # Récupérer la colonne correspondant à la quantité minimale pour le mois spécifié
    min_quantity_col = creation_nom_colonne(month, 'Minimal quantity of shares_', 0)
    min_quantity = criteria[min_quantity_col].values[0]
    
    # Si le critère de quantité minimale est inférieur à 1, cela signifie qu'il n'y a pas de critère de quantité, donc retourner True
    if min_quantity < 1:
        return True
    
    # Calculer le total des parts pour le mois donné en sommant les parts de chaque cycle
    total_lots_for_month = sum([shares[month] for shares in monthly_shares.values()])
    
    # Vérifier si le total des parts respect le critère de quantité minimale
    return total_lots_for_month >= min_quantity




def check_div(monthly_shares, cropping_cycles, criteria):
    """
    Vérifie si le critère de diversité est respecté pour toute l'année, en considérant les filtres de climat et de marketing.
    
    Args:
    monthly_shares (dict): Un dictionnaire où les clés sont les IDs de cycle et les valeurs sont des listes représentant la quantité de lots alloués pour chaque mois.
    cropping_cycles (DataFrame): DataFrame contenant les informations sur chaque cycle, y compris la catégorie de culture et les autres critères associés.
    criteria (DataFrame): DataFrame contenant les critères de diversité pour chaque catégorie de légume, filtré par climat et type de marketing.

    Returns:
    bool: `True` si le critère de diversité est respecté pour tous les mois de l'année, `False` sinon.
    """
    
    # Sélectionner la catégorie de culture du premier cycle
    id = next(iter(monthly_shares))         
    crop_category = cropping_cycles[(cropping_cycles["ID"] == id)]["Crop category"].iloc[0]
    
    # Filtrer les critères de diversité en fonction de la catégorie de culture, du marketing et du climat
    criteria = criteria[(criteria['Marketing'] == '12M') & (criteria['Climate'] == 'Mild') & (criteria['Crop category'] == crop_category)]
    cropping_cycles = cropping_cycles[(cropping_cycles['Marketing'] == '12M') & (cropping_cycles['Climate'] == 'Mild')]
    
    # Boucle à travers les 12 mois de l'année
    months = range(12)
    for month in months:
        
        # Récupérer le critère de diversité minimale pour le mois
        min_diversity_col = creation_nom_colonne(month, 'Minimal number of crops_', 0)
        min_diversity = criteria[min_diversity_col].values[0]
        
        # Si le critère de diversité est 0, passer à l'autre mois (aucune exigence de diversité)
        if min_diversity == 0:
            continue
        
        # Initialiser un ensemble pour les catégories uniques de cultures présentes dans le mois
        categories_in_month = set()
        
        # Vérifier les cycles et ajouter les catégories de cultures présentes dans ce mois
        for cycle_id, shares in monthly_shares.items():
            if shares[month] > 0:  # Vérifier s'il y a des parts allouées pour ce cycle et ce mois
                cycle_info = cropping_cycles[(cropping_cycles['ID'] == cycle_id)]
                if not cycle_info.empty:
                    crop_category = cycle_info['Crop_french'].values[0]
                    categories_in_month.add(crop_category)
        
        # Vérifier si le nombre de catégories distinctes est suffisant pour respecter le critère de diversité
        if len(categories_in_month) < min_diversity:
            return False  # Le critère de diversité n'est pas respecté pour ce mois
    
    return True  # Le critère de diversité est respecté pour tous les mois




def get_box_cat(category, criteria, cropping_cycles):
    """
    Cette fonction choisit des cycles de culture pour une catégorie donnée en fonction des critères de quantité et de diversité.

    Args:
    category (str): La catégorie de culture (par exemple, légumes ou céréales) pour laquelle les cycles de culture sont choisis.
    criteria (DataFrame): DataFrame contenant les critères de culture, avec des informations sur la quantité minimale et la diversité par mois.
    cropping_cycles (DataFrame): DataFrame contenant les informations de chaque cycle de culture, y compris les conditions de marketing, de climat, et la catégorie de culture.

    Returns:
    tuple: Un tuple contenant deux éléments:
        - cycles (dict): Un dictionnaire avec les ID des cycles de culture comme clés et les valeurs correspondant à leurs parts de culture calculées.
        - monthly_shares (dict): Un dictionnaire avec les ID des cycles de culture comme clés et les valeurs correspondant aux parts mensuelles (une liste pour chaque cycle, représentant les parts allouées pour chaque mois).
    """
    
    # Filtrer les cycles de culture en fonction des conditions de climat (Mild) et de marketing (12M)
    filtered_cycles = cropping_cycles[(cropping_cycles['Climate'] == 'Mild') & (cropping_cycles['Marketing'] == '12M')]
    
    monthly_shares = {}  # Dictionnaire pour stocker les parts de culture par mois
    cycles = {}  # Dictionnaire pour stocker les cycles choisis

    while True:
        quantity_check = True  # Initialiser la vérification des quantités comme réussie
        
        # Vérification des quantités pour chaque mois
        for month in range(12):
            if not check_quant(monthly_shares, criteria[criteria['Crop category'] == category], month):
                quantity_check = False  # Si la quantité ne respecte pas les critères, arrêter la vérification
                cycle_id = choose_cycle(category, month, filtered_cycles, criteria)  # Choisir un cycle pour ce mois
                if cycle_id:
                    cycle_data = filtered_cycles[filtered_cycles['ID'] == cycle_id]  # Récupérer les données du cycle choisi
                    cycles = update_shares(cycle_data, cycles, filtered_cycles)  # Mettre à jour les parts du cycle choisi
        
        # Vérifier si les critères de quantité sont respectés
        if quantity_check:
            # Vérifier si les critères de diversité sont respectés
            if not check_div(monthly_shares, cropping_cycles, criteria):
                found = False
                while not found:
                    random_month = np.random.randint(0, 11)  # Choisir un mois aléatoire
                    cycle_id = choose_cycle(category, random_month, filtered_cycles, criteria)  # Choisir un cycle pour ce mois
                    if cycle_id and cycle_id not in monthly_shares:
                        cycle_data = filtered_cycles[filtered_cycles['ID'] == cycle_id]
                        cycles = update_shares(cycle_data, cycles, filtered_cycles)  # Mettre à jour les parts du cycle
                        found = True  # Si un cycle valide est trouvé, sortir de la boucle
            else:
                break  # Si les conditions de quantité et de diversité sont satisfaites, sortir de la boucle

        # Répartir les parts après avoir ajouté les cycles
        for cycle_id in cycles.keys():
            monthly_shares[cycle_id] = spread_shares(filtered_cycles[filtered_cycles['ID'] == cycle_id], cycles, filtered_cycles)

    return cycles, monthly_shares  # Retourner les cycles et les parts mensuelles


def get_N_boxes(N, criteria, cropping_cycles, categories):
    """
    Génère N paniers de cycles de culture, chacun correspondant à différentes catégories de culture,
    en fonction des critères fournis.

    Args:
    N (int): Le nombre de paniers (boîtes) à générer.
    criteria (DataFrame): DataFrame contenant les critères pour chaque catégorie de culture.
    cropping_cycles (DataFrame): DataFrame contenant les informations de chaque cycle de culture, incluant les critères de climat et de marketing.
    categories (list): Liste des catégories de culture pour lesquelles générer les paniers.

    Returns:
    list: Une liste contenant N paniers, où chaque panier est un dictionnaire des cycles de culture pour chaque catégorie.
    """
    boxes = []  # Liste pour stocker tous les paniers générés

    for _ in range(N):
        box = {}  # Dictionnaire pour stocker les cycles de culture pour chaque catégorie

        for category in categories:
            # Appel de la fonction get_box_cat pour chaque catégorie de culture
            box_cat, _ = get_box_cat(category, criteria, cropping_cycles)
            box[category] = box_cat  # Ajouter les cycles de culture pour cette catégorie au panier

        boxes.append(box)  # Ajouter le panier complet à la liste des paniers

    # Sauvegarde des paniers générés dans un fichier CSV
    save_boxes_to_csv(boxes)

    return boxes  # Retourner la liste des paniers générés

def save_boxes_to_csv(boxes):
    """
    Sauvegarde les paniers de cycles de culture dans un fichier CSV.

    Args:
    boxes (list): Liste des paniers à sauvegarder. Chaque panier est un dictionnaire
                  où les clés sont les catégories de culture et les valeurs sont des dictionnaires
                  avec des 'Cycle ID' comme clés et des 'Lots' comme valeurs.

    Returns:
    None
    """
    # Ouvrir le fichier en mode écriture (écrasera le fichier existant)
    with open('boxes.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Écriture de l'en-tête du fichier CSV
        writer.writerow(['Category', 'Cycle ID', 'Lots'])
        
        # Itération sur chaque panier dans la liste des paniers
        for box in boxes:
            for category, cycles in box.items():
                # Itération sur chaque cycle dans un panier
                for cycle_id, lots in cycles.items():
                    # Enregistrer les données dans le fichier CSV
                    writer.writerow([category, int(cycle_id), int(lots)])

    print("Les paniers ont été sauvegardés dans 'boxes.csv' avec succès !")

def calculate_Yc(ac):
    """
    Calcule le rendement Yc (en termes absolus) à partir des effets aléatoires sur le rendement, en utilisant 
    une distribution normale pour chaque effet (système, ferme et aléatoire).
    
    Args:
    ac (float): Effet sur le Log_Yield (rendement logarithmique) pour le légume donné.
    
    Returns:
    float: Le rendement Yc calculé en valeur absolue (non logarithmique).
    """
    # Paramètres du modèle pour les effets : moyennes et écarts-types des distributions normales
    as_mean, as_sigma = 0.74, 0.12  # Effet moyen et écart-type du système
    af_mean, af_sigma = 0.0, 0.42  # Effet moyen et écart-type de la ferme
    r_mean, r_sigma = 0.0, 0.14  # Effet moyen et écart-type aléatoire

    # Simulation des effets aléatoires tirés de distributions normales
    as_val = np.random.normal(as_mean, as_sigma)  # Effet du système
    af_val = np.random.normal(af_mean, af_sigma)  # Effet de la ferme
    r_val = np.random.normal(r_mean, r_sigma)  # Effet aléatoire

    # Calcul de log(Yc) en ajoutant les différents effets au Log_Yield donné (ac)
    log_Yc = as_val + af_val + ac + r_val

    # Retourner Yc en valeur absolue (exponentielle du log(Yc))
    return np.exp(log_Yc)


def calculate_Wc(bc):
    """
    Calcule la charge de travail Wc (en termes absolus) à partir des effets aléatoires sur la charge de travail, 
    en utilisant une distribution normale pour chaque effet (système, ferme et aléatoire).
    
    Args:
    bc (float): Effet sur le Log_Production_workload pour le légume donné.
    
    Returns:
    float: La charge de travail Wc calculée en valeur absolue (non logarithmique).
    """
    # Paramètres du modèle pour les effets : moyennes et écarts-types des distributions normales
    bs_mean, bs_sigma = 2.72, 0.19  # Effet moyen et écart-type du système
    bf_mean, bf_sigma = 0.0, 0.36  # Effet moyen et écart-type de la ferme
    s_mean, s_sigma = 0.0, 0.21   # Effet moyen et écart-type aléatoire

    # Simulation des effets aléatoires tirés de distributions normales
    bs_val = np.random.normal(bs_mean, bs_sigma)  # Effet du système
    bf_val = np.random.normal(bf_mean, bf_sigma)  # Effet de la ferme
    s_val = np.random.normal(s_mean, s_sigma)    # Effet aléatoire

    # Calcul de log(Wc) en ajoutant les différents effets au Log_Production_workload donné (bc)
    log_Wc = bs_val + bf_val + bc + s_val

    # Retourner Wc en valeur absolue (exponentielle du log(Wc))
    return np.exp(log_Wc)


def compute_CA_workload(box, crop_properties):
    """
    Calcule la charge de travail totale et le profit total pour un panier donné, en fonction des cycles de culture
    et des propriétés des cultures.

    Args:
    box (dict): Un dictionnaire représentant un panier de cultures. Chaque clé est une catégorie de culture, et chaque 
                valeur est un autre dictionnaire avec des IDs de culture et le nombre de lots correspondants pour chaque 
                culture dans la catégorie.
    crop_properties (DataFrame): Un DataFrame contenant les propriétés des cultures. Il doit inclure les colonnes suivantes :
                                 - 'Crop_french' : le nom en français de la culture
                                 - 'Effect_on Log_Yield' : l'effet sur le rendement logarithmique de la culture
                                 - 'Effect_on_Log_Production_workload' : l'effet sur la charge de travail logarithmique
                                 - 'Price' : le prix par kilogramme de la culture
                                 - 'Quantity_per_share' : la quantité de produit par part

    Returns:
    tuple: Un tuple contenant deux valeurs :
        - total_workload (float): La charge de travail totale calculée pour tous les lots dans le panier.
        - total_profit (float): Le profit total calculé pour tous les lots dans le panier.
    """
    
    total_profit = 0.0
    total_workload = 0.0

    # Chargement de la base de données des cycles de culture
    cropping_cycles = get_cropping_cycles_database("data/session2/cropping_cycles.csv")
    
    for cat in box.values():
        for crop_id, lots in cat.items():
            # Récupération des données de crop_cycles pour l'ID spécifié
            cycle_data = cropping_cycles[cropping_cycles['ID'] == crop_id]

            if not cycle_data.empty:
                crop_french = cycle_data['Crop_french'].iloc[0]
                # Récupération des données de crop_properties basées sur Crop_french
                crop_data = crop_properties[crop_properties['Crop_french'] == crop_french].iloc[0]
                
                # Extraction des paramètres nécessaires pour les calculs
                ac = crop_data['Effect_on Log_Yield']
                bc = crop_data['Effect_on_Log_Production_workload']
                
                Yc = calculate_Yc(ac)
                
                # Calcul du profit et de la charge de travail pour les lots donnés
                price_per_kg = crop_data['Price']
                quantity_per_share = crop_data['Quantity_per_share']
            
                # Surface occupé
                surface = lots / Yc
                
                # Profit €/m**2
                profit = (price_per_kg * quantity_per_share * Yc)
                
                
                total_profit += profit * surface
                total_workload += sum( calculate_Wc(bc) for _ in range(lots))/lots # Moyenne charge de travail

    return total_workload, total_profit




def compute_CA(ws, cas, workload_total):
    """
    Calcule le chiffre d'affaires correspondant à une charge de travail totale spécifique, en utilisant
    les charges de travail et chiffres d'affaires par mètre carré.

    Args:
    ws (np.array): Tableau des charges de travail en heures par mètre carré pour chaque boîte.
    cas (np.array): Tableau des chiffres d'affaires en euros par mètre carré pour chaque boîte.
    workload_total (float): Charge de travail totale en heures.

    Returns:
    np.array: Tableau des chiffres d'affaires calculés pour la charge de travail spécifiée.
    """
    
    # Calculer le chiffre d'affaires correspondant pour chaque surface calculée
    ca = workload_total * cas / ws
    
    return ca


def compute_probability_viable(boxes, CA_min, workload_max, crop_properties):
    """
    Calcule la probabilité qu'une configuration de panier soit viable en fonction des critères de charge de travail et 
    de revenu minimum. Une configuration est considérée comme viable si elle respecte les seuils spécifiés 
    pour la charge de travail et le revenu.

    Args:
    boxes (list of dict): Une liste de paniers où chaque panier est un dictionnaire. 
                           Chaque dictionnaire représente un ensemble de cultures pour une catégorie donnée, avec les IDs de cycle 
                           comme clés et les lots associés comme valeurs.
    CA_min (float): Le seuil minimum de revenu nécessaire pour qu'une configuration soit considérée comme viable.
    workload_max (float): Le seuil maximum de charge de travail autorisé pour qu'une configuration soit considérée comme viable.
    crop_properties (DataFrame): Un DataFrame contenant les propriétés des cultures. Il doit inclure les colonnes suivantes :
                                 - 'Crop_french' : le nom en français de la culture
                                 - 'Effect_on Log_Yield' : l'effet sur le rendement logarithmique de la culture
                                 - 'Effect_on_Log_Production_workload' : l'effet sur la charge de travail logarithmique
                                 - 'Price' : le prix par kilogramme de la culture
                                 - 'Quantity_per_share' : la quantité de produit par part

    Returns:
    float: La probabilité que les configurations de paniers soient viables, calculée comme le rapport des configurations viables 
           par rapport au nombre total de configurations.
    """
    
    # Compteurs pour les configurations viables
    viable_count = 0
    total_count = len(boxes)
    
    # Itérer sur chaque configuration de panier
    for box in boxes:
        # Calculer la charge de travail et le revenu pour la configuration de panier actuelle
        workload, CA = compute_CA_workload(box, crop_properties)
        
        # Vérifier si la condition est satisfaite
        profit = compute_CA(workload,CA,workload_max)
        if profit>CA_min:
            viable_count+=1
    
    # Calculer la probabilité comme le ratio des configurations viables sur le nombre total de configurations
    proba_viable = viable_count / total_count if total_count > 0 else 0
    
    return proba_viable

def figure_distribution(workloads, revenues):
    """
    Affiche la distribution conjointe des revenus et de la charge de travail par mètre carré à l'aide d'un histogramme 2D.
    
    Args :
    workloads (array) : Liste ou tableau des valeurs de la charge de travail (en heures/m²), représentant 
                          le travail requis pour chaque point de données (par exemple, le travail nécessaire pour une culture).
    revenues (array) : Liste ou tableau des valeurs des revenus (en €/m²), représentant le revenu associé 
                         à chaque point de données correspondant (par exemple, le profit gagné par mètre carré pour chaque culture).
    """
    
    sns.kdeplot(x=workloads,y=revenues,cmap="viridis",fill=True,levels=100,thresh=0)

    # Ajouter les labels et le titre
    plt.xlabel('Charge de travail (heures/m²)')
    plt.ylabel('Profit (€/m²)')
    plt.title('Distribution conjointe des revenus et de la charge de travail par m²')
    
    # Afficher le graphique
    plt.show()

def figure_CAtot(CA_total):
    """
    Affiche un histogramme de la distribution du chiffre d'affaires annuel.

    Args:
    CA_total (numpy.ndarray): Un tableau numpy contenant les valeurs du chiffre d'affaires annuel pour chaque microferme simulée.

    Returns:
    None: Cette fonction ne retourne rien. Elle affiche simplement un histogramme de la distribution du chiffre d'affaires annuel.
    """
    # Définir la taille de la figure
    plt.figure(figsize=(10, 6))
    
    # Créer un histogramme de CA_total avec 50 intervalles (bins) et une couleur noire pour les barres
    plt.hist(CA_total, bins=50, color="black")
    
    # Étiqueter l'axe x pour indiquer qu'il représente le chiffre d'affaires
    plt.xlabel("Chiffre d'affaire")
    
    # Étiqueter l'axe y pour indiquer la fréquence des valeurs de CA dans chaque intervalle
    plt.ylabel("Fréquence")
    
    # Ajouter un titre décrivant la distribution affichée
    plt.title("Distribution du Chiffre d'Affaire Annuel")
    
    # Afficher l'histogramme
    plt.show()
