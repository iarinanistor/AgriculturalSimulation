# Iarina NISTOR 21210925
#Explications et Interprétation des Résultats dans le fichier .ipynb

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.stats.weightstats as st
import random
import networkx as nx



def get_valid_indices_all_vars(li):
    """
    Renvoie les indices des colonnes qui ne contiennent aucune valeur manquante (NaN) 
    dans toutes les variables fournies dans une liste de tableaux.

    Args:
        li (list of np.array): Liste de tableaux NumPy

    Returns:
        list: Liste des indices des colonnes sans valeurs manquantes (NaN) .
    """
    res = []  # Initialise une liste pour stocker les indices des colonnes valides

    # Détermine le nombre de lignes et de colonnes à vérifier
    nb_var, nb_colonnes = np.array(li).shape

    # Parcours de chaque colonne pour vérifier la présence de NaN dans toutes les variables
    for i in range(nb_colonnes):
        add = True  # Flag indiquant si la colonne `i` est valide pour toutes les variables
        for j in range(nb_var):
            if np.isnan(li[j][i]):  # Vérifie si la variable `j` contient un NaN dans la colonne `i`
                add = False  # Si un NaN est trouvé, la colonne n'est pas valide
                break  # Sort de la boucle pour passer à la colonne suivante
        if add:
            res.append(i)  # Ajoute l'indice `i` à la liste si la colonne est valide
    
    return res  # Retourne la liste des indices de colonnes sans NaN



def compute_LER(df):
    """
    Calcule le Land Equivalent Ratio (LER) pour chaque paire de cultures en utilisant les rendements 
    en monoculture et en interculture. Le LER est une mesure de l'efficacité de l'interculture par 
    rapport aux monocultures.

    Args:
        df : Un DataFrame contenant les rendements pour les cultures en monoculture et en interculture.
            Il doit inclure les colonnes suivantes :
            - 'Crop_1_yield_sole' : Rendement de la première culture en monoculture.
            - 'Crop_2_yield_sole' : Rendement de la deuxième culture en monoculture.
            - 'Crop_1_yield_intercropped' : Rendement de la première culture en interculture.
            - 'Crop_2_yield_intercropped' : Rendement de la deuxième culture en interculture.
    
    Returns:
        numpy.array : Un tableau contenant les valeurs de LER calculées pour chaque ligne du df.
                      Le LER est calculé comme la somme des rendements relatifs des cultures en interculture 
                      par rapport à leurs rendements en monoculture.
    """
    # On obtient les indices valides des lignes où il n'y a pas de valeurs manquantes (NaN)
    list_index = get_valid_indices_all_vars([
        df['Crop_1_yield_sole'].to_numpy(), 
        df['Crop_2_yield_sole'].to_numpy(), 
        df['Crop_1_yield_intercropped'].to_numpy(), 
        df['Crop_2_yield_intercropped'].to_numpy()
    ])
    
    # Sélection des rendements pour les cultures en monoculture et en interculture, uniquement pour les lignes valides
    IY1 = df.loc[list_index, 'Crop_1_yield_intercropped']  # Rendement de la culture 1 en interculture
    IY2 = df.loc[list_index, 'Crop_2_yield_intercropped']  # Rendement de la culture 2 en interculture
    SY1 = df.loc[list_index, 'Crop_1_yield_sole']  # Rendement de la culture 1 en monoculture
    SY2 = df.loc[list_index, 'Crop_2_yield_sole']  # Rendement de la culture 2 en monoculture
    
    # Calcul du Land Equivalent Ratio (LER)
    # LER est la somme des rendements relatifs des cultures intercultivées par rapport à leurs monocultures
    return IY1 / SY1 + IY2 / SY2




def plot_LERs(LER_calcule, LER_tot):
    """
    Trace un graphique de régression linéaire entre les valeurs calculées du Land Equivalent Ratio (LER)
    et les valeurs observées du LER total. Affiche également les indicateurs de performance du modèle de régression.

    Args:
        LER_calcule (numpy.array): Série ou tableau contenant les valeurs calculées du LER pour chaque observation.
        LER_tot (numpy.array): Série ou tableau contenant les valeurs observées du LER total pour chaque observation.

    Returns:
        None : Cette fonction trace un graphique et affiche les indicateurs de performance.
    """
    # Filtrage des données valides (en utilisant les indices valides pour les deux séries)
    indexes = [LER_calcule.index[i] for i in range(len(LER_calcule))]
    LER_tot = LER_tot[indexes]  # Aligne les indices de LER_tot avec ceux de LER_calcule
    list_index = get_valid_indices_all_vars([LER_calcule.to_numpy(), LER_tot])  # Obtenez les indices sans NaN

    # Reshape des données pour la régression
    LER_calc = LER_calcule.to_numpy()[list_index].reshape(-1, 1)  # Les valeurs calculées de LER (reshape pour la régression)
    LER = LER_tot[list_index]  # Les valeurs observées de LER

    # Ajustement du modèle de régression linéaire
    model = LinearRegression()
    model.fit(LER_calc, LER)  # Entraîne le modèle avec les données
    y_pred = model.predict(LER_calc)  # Prédictions du modèle

    # Calcul des indicateurs de performance du modèle
    rmse = mean_squared_error(LER, y_pred, squared=True)  # Erreur quadratique moyenne (RMSE)
    r2 = r2_score(LER, y_pred)  # Coefficient de correlation R²

    # Affichage des résultats de la performance du modèle
    print("RMSE: " + str(rmse) + "\nR²: " + str(r2))

    # Tracé du graphique : nuage de points pour les données réelles et la ligne de régression
    plt.scatter(LER_calc, LER, color='blue', label='Données réelles')  # Trace les points réels
    plt.plot(LER_calc, y_pred, color='red', linewidth=2, label='Ligne de régression')  # Trace la droite de régression
    plt.xlabel("LER calculé")  # Légende de l'axe X
    plt.ylabel("LER observé")  # Légende de l'axe Y
    plt.title("Régression linéaire entre LER calculé et LER observé")  # Titre du graphique
    plt.legend()  # Affiche la légende du graphique
    plt.show()  # Affiche le graphique
    
    
    

def compute_mean_std_inter(LER):
    """
    Calcule la moyenne, l'écart-type et l'intervalle de confiance à 95% des valeurs de Land Equivalent Ratio (LER).

    Args:
        LER (numpy.array) : Tableau ou série contenant les valeurs du Land Equivalent Ratio (LER).

    Returns:
        tuple : Un tuple contenant trois éléments :
            - moyenne (float) : La moyenne des valeurs de LER.
            - std (float) : L'écart-type des valeurs de LER.
            - intervalle (tuple) : L'intervalle de confiance à 95% pour la moyenne des valeurs de LER.
    """
    # Filtrage des valeurs valides pour s'assurer qu'il n'y a pas de NaN
    LER = LER[get_valid_indices_all_vars([LER])]  # On sélectionne les indices valides
    
    # Calcul de la moyenne et de l'écart-type
    moyenne = np.mean(LER)
    std = np.std(LER)
    
    # Calcul de l'intervalle de confiance à 95% (utilisation de la distribution normale)
    err = 1.96 * std / np.sqrt(len(LER)) # Marge d'erreur
    deb = moyenne - err  # Limite inférieure de l'intervalle
    fin = moyenne + err    # Limite supérieure de l'intervalle
    
    # Retour des résultats sous forme de tuple : moyenne, écart-type et intervalle de confiance
    return moyenne, std, [deb, fin]


def testH0(LER):
    """
    Effectue un test statistique unilatéral pour déterminer si la moyenne des valeurs de LER est significativement supérieure à 1.

    Args :
    LER (numpy.array ou pandas.Series) : Tableau ou série contenant les valeurs du Land Equivalent Ratio (LER).
                                        Ces valeurs seront utilisées pour effectuer le test statistique unilatéral.

    Returns :
    None : Affiche un message indiquant si l'hypothèse nulle (que la moyenne de LER est inférieure ou égale à 1) est rejetée
           ou non au niveau de confiance de 95%. La fonction n'a pas de valeur de retour directe.
    """
    # Définir la valeur de l'hypothèse nulle (ici, 1)
    valeur_hypothese = 1

    # Effectuer le test z unilatéral pour vérifier si la moyenne de LER est supérieure à 1
    test_stat, p_value = st.ztest(LER, value=valeur_hypothese, alternative='larger')

    # Afficher les résultats du test
    print("Statistique du test Z :", test_stat)
    print("p-value :", p_value)

    # Interprétation des résultats
    if p_value < 0.05:
        print("Nous rejetons l'hypothèse nulle : la moyenne de LER est significativement supérieure à 1 au niveau de confiance de 95%.")
    else:
        print("Nous ne pouvons pas rejeter l'hypothèse nulle : la moyenne de LER n'est pas significativement supérieure à 1.")


def plot_dist_mean(LER, num_samples=100000):
    """
    Trace la distribution des moyennes d'échantillons tirés aléatoirement à partir des valeurs de LER.

    Args :
    LER (numpy.array ou pandas.Series) : Tableau ou série contenant les valeurs du Land Equivalent Ratio (LER). Ces valeurs sont utilisées pour générer des échantillons aléatoires et calculer les moyennes.
    
    num_samples (int, optionnel) : Nombre d'échantillons à tirer pour construire la distribution des moyennes (par défaut 100000). Cela définit combien d'échantillons aléatoires seront générés et utilisés pour estimer la distribution des moyennes.

    Returns :
    None : La fonction n'a pas de valeur de retour. Elle affiche un graphique représentant l'histogramme de la distribution des moyennes des échantillons générés.
    """
    # Filtrage des valeurs valides
    LER = LER[get_valid_indices_all_vars([LER])]
    sample_means = []
    
    # Génération des échantillons aléatoires et calcul des moyennes
    for _ in range(num_samples):
        size = random.randint(int(0.5*len(LER)), int(0.8*len(LER)))  # Taille de l'échantillon aléatoire entre 50% et 80% de la taille totale
        sample = np.random.choice(LER, size=size, replace=True)  # Tirage aléatoire avec remplacement
        sample_means.append(np.mean(sample))  # Calcul de la moyenne de cet échantillon
    
    # Tracer l'histogramme des moyennes
    sample_means = np.array(sample_means)
    plt.hist(sample_means, bins=200, color='black', alpha=1, edgecolor='black')
    plt.xlabel('Moyenne des échantillons')
    plt.ylabel('Fréquence')
    plt.title('Distribution des moyennes pour différents échantillons')
    plt.show()
    
    


def get_sorted_crops(df):
    """
    Calcule et renvoie les cultures triées par probabilité de rendement accru, basées sur les résultats des expériences.

    Args :
    df : DataFrame contenant les résultats des expériences, avec les colonnes suivantes :
        - 'Crop_1_Common_Name' : Nom commun de la première culture.
        - 'Crop_2_Common_Name' : Nom commun de la deuxième culture.
        - 'LER_crop1' : Ratio d'Équivalent de Terre (LER) pour la première culture.
        - 'LER_crop2' : Ratio d'Équivalent de Terre (LER) pour la deuxième culture.

    Returns :
    list : Une liste de tuples contenant (nom de culture, probabilité), triée par probabilité décroissante de rendement accru.
           Chaque tuple représente une culture et la probabilité d'un rendement accru (probabilité que son LER soit supérieur à 0.5 dans les expériences).
    """
    # Créer un ensemble contenant tous les noms de cultures uniques
    all_culture_names = set(df['Crop_1_Common_Name'].unique()) | set(df['Crop_2_Common_Name'].unique())
    cultures_to_keep = []
    
    # Filtrer les cultures apparaissant dans au moins 10 expériences
    for culture in all_culture_names:
        count = df[(df['Crop_1_Common_Name'] == culture) | (df['Crop_2_Common_Name'] == culture)].shape[0]
        if count > 10:
            cultures_to_keep.append(culture)
    
    probabilities = []
    # Calculer la probabilité d'un rendement accru pour chaque culture
    for culture in cultures_to_keep:
        total_experiment = 0
        success_count = 0
        for i in range(len(df)):
            if df['Crop_1_Common_Name'].iloc[i] == culture:
                total_experiment += 1
                if df['LER_crop1'].iloc[i] > 0.5:
                    success_count += 1
            if df['Crop_2_Common_Name'].iloc[i] == culture:
                total_experiment += 1
                if df['LER_crop2'].iloc[i] > 0.5:
                    success_count += 1
        # Calcul de la probabilité
        probability = success_count / total_experiment
        probabilities.append((culture, probability))
    
    # Retourner la liste triée par probabilité décroissante
    return sorted(probabilities, key=lambda x: x[1], reverse=True)


def list_clusters(df, th=1.8):
    """
    Identifie et affiche les clusters de cultures ayant un Land Equivalent Ratio (LER) total moyen supérieur ou égal à un seuil donné.
    Chaque cluster est constitué de paires de cultures qui présentent un rendement accru lorsqu'intercultivées, selon les résultats d'expériences.

    Args :
    df : DataFrame contenant les colonnes suivantes :
        - 'Crop_1_Common_Name' : Nom de la première culture.
        - 'Crop_2_Common_Name' : Nom de la deuxième culture.
        - 'LER_tot' : Ratio d'Équivalent de Terre (LER) total pour la paire de cultures.
    th (float) : Seuil de LER total moyen pour la création d'une arête entre deux cultures. 
                Par défaut, 1.8. Si la moyenne du LER entre deux cultures est supérieure ou égale à ce seuil, elles sont connectées par une arête.

    returns :
    None : La fonction affiche directement les clusters identifiés sous forme de listes de cultures connectées.
    """
    
    # Création d'un graphe vide pour représenter les relations entre les cultures
    G = nx.Graph()
    
    # Parcours des paires de cultures dans le DataFrame
    for crop1, crop2 in zip(df['Crop_1_Common_Name'], df['Crop_2_Common_Name']):
        
        # Application du masque pour identifier les lignes où les paires de cultures correspondent (coupées dans les deux sens)
        mask = (
            ((df['Crop_1_Common_Name'] == crop1) & (df['Crop_2_Common_Name'] == crop2)) |
            ((df['Crop_1_Common_Name'] == crop2) & (df['Crop_2_Common_Name'] == crop1))
        )
        
        # Calcul de la moyenne du LER total pour la paire de cultures en question
        mean_ler = df[mask]['LER_tot'].mean()
        
        # Si la moyenne du LER est supérieure ou égale au seuil, on ajoute une arête entre ces deux cultures dans le graphe
        if mean_ler >= th:
            G.add_edge(crop1, crop2)
    
    # Identification des composantes connexes dans le graphe (c'est-à-dire les clusters de cultures)
    clusters = list(nx.connected_components(G))

    # Trouver les composantes connexes
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)  # Position des nœuds
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Graphe des associations de cultures (LER > 1.8)")
    

    # Affichage des clusters de cultures
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}\n-----------")
        # Affiche chaque culture du cluster, une par ligne
        print(*cluster, sep="\n")
        print("-----------\n")

    plt.show()


