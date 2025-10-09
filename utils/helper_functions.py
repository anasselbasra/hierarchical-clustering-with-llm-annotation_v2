from typing import Optional, Union, List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from plotly.subplots import make_subplots
from collections import defaultdict
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import pandas as pd
import numpy as np
import openai 
import emoji
import time
import re





def clean_text(text: str) -> str:
    """
    Supprime les emojis et normalise les espaces dans une chaîne de caractères.

    INPUTS:
    text : str   Texte brut à nettoyer.

    OUTPUTS:
    str Texte nettoyé (sans emoji, avec espaces normalisés).
    """
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text




def farthest_sampling_high_proba(
                    embeddings: np.ndarray,
                    indices: List[int],
                    proba_list: np.ndarray,
                    proba_threshold: float,
                    k: int
                ) -> List[int]:
    """
    Sélectionne jusqu'à k points les plus éloignés parmi ceux ayant une probabilité ≥ seuil.

    INPUTS:
    ----------
    embeddings : np.ndarray
        Matrice des embeddings (shape : n_samples, n_dims).
    indices : list[int]
        Indices des points appartenant à un cluster donné.
    proba_list : np.ndarray
        Vecteur des probabilités d'appartenance pour chaque point.
    proba_threshold : float
        Seuil minimal à respecter pour être considéré.
    k : int
        Nombre maximal de points à sélectionner.

    Retour
    ------
    list[int]
        Indices des k points choisis (par éloignement mutuel).
    """
    high_proba_indices = [i for i in indices if proba_list[i] >= proba_threshold]
    # on ne sélectionne que les points ayant une probabilité ≥ seuil
    if len(high_proba_indices) == 0:
        return [] # cela s'applique principalement au cluster est -1, on ne sélectionne rien

    cluster_coords = embeddings[indices]
    cluster_center = cluster_coords.mean(axis=0) # centre du cluster
    # trouve le centre puis le point le plus éloigné du centre
    # parmi ceux ayant une probabilité ≥ seuil
    # puis on sélectionne les points les plus éloignés mutuellement
    # jusqu'à k points ou jusqu'à ce qu'il n'y ait plus de points à sélectionner

    high_coords = embeddings[high_proba_indices]
    distances_to_center = np.linalg.norm(high_coords - cluster_center, axis=1)
    first_idx = np.argmax(distances_to_center)
    selected = [high_proba_indices[first_idx]]

    while len(selected) < min(k, len(high_proba_indices)):
        remaining = list(set(high_proba_indices) - set(selected))
        rem_coords = embeddings[remaining]
        sel_coords = embeddings[selected]

        dists = np.min(
            np.linalg.norm(rem_coords[:, None, :] - sel_coords[None, :, :], axis=2),
            axis=1
        )
        next_idx = remaining[np.argmax(dists)]
        selected.append(next_idx)

    return selected



def select_examples(
    reduced_embeddings: np.ndarray,
    label2docs: Dict[int, List[int]],
    probas: np.ndarray,
    min_size: int = 20,
    partition: float = 0.02,
    proba_threshold: float = 0.8,
    method_selection: str = "farthest_high_proba",
    proba_min_low: float = 0.35
) -> Dict[int, List[int]]:
    """
    Sélectionne des exemples représentatifs par cluster avec sampling intelligent.
    
    Returns:
        dict: {label → [doc_ids sélectionnés]}
    """
    selected_examples = {}

    for label in label2docs.keys():
        indices = label2docs[label]
        len_cluster = len(indices)

        examples_size = max(min(min_size, len_cluster), int(len_cluster * partition))
        # Ajustement de la taille d'échantillon: on choisit un nombre d'exemples proportionnel à la taille du cluster
        # mais pas inférieur à min_size
        # si le cluster est trop petit, on prend tout
        # si le cluster est trop grand, on prend un pourcentage
        # il faut justifier pourquoi "partition" est fixé à 0.02
        k_high = int(0.75 * examples_size) #max(15, int(0.75 * examples_size))
        # on prend au moins 27 exemples pour la sélection haute proba
        k_low  = examples_size - k_high #max(5, examples_size - k_high)

        if method_selection == "farthest_high_proba":
            # Sélection haute proba
            selected_high = farthest_sampling_high_proba(
                embeddings=reduced_embeddings,
                indices=indices,
                proba_list=probas,
                proba_threshold=proba_threshold,
                k=k_high
            )
            valid_low_indices = [
                i for i in indices if proba_min_low <= probas[i] < proba_threshold
            ]

            selected_low = farthest_sampling_high_proba(
                embeddings=reduced_embeddings,
                indices=valid_low_indices,
                proba_list=np.ones_like(probas),  # toutes passent le filtre
                proba_threshold=0,  # aucun filtrage supplémentaire ici
                k=k_low
            ) if len(valid_low_indices) > 0 else []

            # Union sans doublons
            selected_all = list(set(selected_high + selected_low))
            selected_examples[label] = selected_all

        else:
            raise ValueError(f"Unsupported method_selection: {method_selection}")

    return selected_examples
#test

# reduced_embeddings = df[["x", "y" ]].to_numpy()

# label2docs = defaultdict(list)
# for i, label in enumerate(cluster_labels):
#     label2docs[label].append(i)

# selected = select_examples(
#     reduced_embeddings=reduced_embeddings,
#     label2docs=label2docs,
#     probas=df.proba.to_numpy(),
#     min_size=30,
#     partition=0.025,
#     proba_threshold=0.8,
#     method_selection="farthest_high_proba"
# )
# print(f"Selected examples: {selected}")



def extract_numeric_key(label: str) -> int:
    """
    ["-1", "1",  "3","2_21"] -> ['-1', '1', '2_21', '8']
    """
    try:
        parts = label.split("_")
        nums = [int(p) for p in parts]
        return min(nums)
    except:
        return float('inf')  # Pour que les labels non numériques aillent à la fin

# Nice one
def call_openai_api(
                df: pd.DataFrame, # df doit contenir 'text', 'annotation', 'x', 'y', une colonne pour les clusters et une pour les probabilités

                col_topic: str = "topic",
                col_proba: str = "proba",
                api_key: Optional[str] = None,
                instruction : Optional[str] = None,
                model: str = "gpt-4o-mini", # à ne pas changer !!!!!
                summary_chunk_size: int = 1000,
                return_inputs_outputs: bool = False,
                existing_labels: List[str] = [],
                user_assistant: Optional[List[Dict[str, str]]] = [], 
                sleep_time: int = 5,
                sleep_every: int = 50
            ) -> Union[Dict[int, str], Dict[str, Union[Dict[int, str], List, List]]]:
    
    """
    Génère automatiquement des étiquettes de sujets pour chaque cluster à l'aide de l'API OpenAI.
    Cette fonction sélectionne des exemples pertinents dans chaque cluster, construit un prompt pour chaque cluster, 
    puis interroge un LLM (via OpenAI) pour générer un résumé textuel représentant le sujet du cluster.

    INPUTS:
    ----------
    df : pd.DataFrame
        Un DataFrame contenant les colonnes suivantes :
        - 'text' : les textes bruts du document
        - 'col_topic' : le label de cluster associé à chaque document
        - 'col_proba' : la probabilité d'appartenance au cluster (issue d'un modèle type HDBSCAN ou autre)
        - 'x', 'y' : coordonnées réduites (par ex. via UMAP) utilisées pour la sélection d'exemples.

    api_key : str, optionnel
        Clé d'API OpenAI. Si non fournie, les clusters seront annotés de manière générique sans appel LLM.

    instruction : str, optionnel
        Instruction à injecter dans le prompt pour guider le modèle. Si non spécifiée (non recommandée si existance de api_key), une instruction par défaut est utilisée.

    model : str, optionnel
        Nom du modèle OpenAI à utiliser (ex : "gpt-4o-mini", "gpt-3.5-turbo", etc.), la valeur par défaut est gpt-4o-mini, à ne pas changer.

    summary_chunk_size : int, optionnel
        Nombre maximal de caractères à conserver par exemple. Par défaut : 1000.

    return_inputs_outputs : bool, optionnel
        Si `True`, retourne également les prompts construits et les réponses brutes du modèle, nécessaire si on a besoin de calculer le cout de chaque appel.

    OUTPUTS:
    -------
    Union[Dict[int, str], Dict[str, Union[Dict[int, str], List, List]]]
        - Si `return_inputs_outputs=False` : retourne un dictionnaire `cluster_summaries` mappant chaque label de cluster à son résumé.
        - Si `return_inputs_outputs=True` : retourne un dictionnaire contenant :
            - 'summaries' : les résumés textuels
            - 'messages' : la liste des prompts envoyés
            - 'outputs' : les réponses brutes reçues
        - Si `api_key` n'est pas fourni, retourne un dictionnaire avec des étiquettes génériques pour chaque cluster.
    """
    # Copie défensive pour éviter les effets de bord
    df = df.copy()
    # On crée deux colonnes internes utilisées par la suite
    df["topic"] = df[col_topic]
    df["proba"] = df[col_proba]

    texts = (df.text.apply(lambda x: clean_text(x))).to_list()

    # cluster_labels = df.topic.apply(lambda x: int(x)).to_numpy() ####ici on transforme les labels en int
    cluster_labels = df.topic.to_numpy() # str

    #unique_labels = [int(label) for label in set(cluster_labels) if label != -1]  ## int ??
    unique_labels = [(label) for label in set(cluster_labels) if label != "-1"]  ## str

    # cluster_summaries = {-1: "None"} ## int
    cluster_summaries = {'-1': "None"} ## str

    if api_key:
        client = openai.OpenAI(api_key=api_key)
        if instruction is None:
                instruction = "Summarize the following examples into a concise topic label."

        reduced_embeddings = df[["x", "y" ]].to_numpy()

        # ⚠️ Cluster -1 ignoré
        # unique_labels = [int(label) for label in set(cluster_labels) if label != -1] ## int
        unique_labels = [(label) for label in set(cluster_labels) if label != '-1'] ## str  

        # Regroupement des documents par label
        label2docs = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            label2docs[label].append(i)

        # Sélection des exemples
        selected_ids_per_cluster = select_examples(
            reduced_embeddings=reduced_embeddings,
            label2docs=label2docs,
            probas=df.proba.to_numpy(),
            min_size=20,
            partition=0.02,
            proba_threshold=0.77,
            proba_min_low=0.35,
            method_selection="farthest_high_proba"
        )
        all_messages = [] 
        all_outputs = []
        
        existing_labels = existing_labels
        instruction_template = instruction  # on garde le modèle intact
        ii = 0
        for label in tqdm(unique_labels, desc="Processing clusters"): # unique_labels
            instruction = instruction_template.format(existing_labels=existing_labels)
            selected_ids = selected_ids_per_cluster.get(label, [])

            if not selected_ids:
                cluster_summaries[label] = "No valid examples"
                continue

            examples = "\n\n".join([
                f"Example {i+1}:\n{texts[_id][:summary_chunk_size]}"
                for i, _id in enumerate(selected_ids)
            ])
            user_assistant = user_assistant

            messages = [{"role": "system", "content": instruction}]
            if user_assistant:
                messages.extend(user_assistant)
            messages += [{"role": "user", "content": examples}]

            all_messages.append(messages)  
            ii += 1
            if ii % sleep_every == 0:
                print(f"Waiting for {sleep_time} seconds to respect rate limits after {ii} requests...")
                time.sleep(sleep_time)
            


            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
                summary = response.choices[0].message.content.strip()
                
            except Exception as e:
                summary = f"Error: {e}"

            if "delete" in summary.lower():
                summary = "DELETE"  # Si le modèle suggère de supprimer le cluster, on le remplace par "DELETE"


            cluster_summaries[label] = summary
            existing_labels.append(summary) if summary.lower() != "delete" else None   # Ajouter le nouveau label à la liste des labels existants
            all_outputs.append(summary) 


        
        return {
                "summaries": cluster_summaries,
                "messages": all_messages,
                "outputs": all_outputs
            } if return_inputs_outputs else cluster_summaries
    
    # Si pas d'API key, on retourne des étiquettes génériques {"-1": "None", "0": "cluster 0", ...}

    cluster_summaries = cluster_summaries | {label: ("cluster " + label) for label in sorted(unique_labels, key=extract_numeric_key)}  #(Python 3.9+)
    return cluster_summaries
# test 
# call_openai_api(df)   
  







def cluster_centers(
                embeddings: np.ndarray,                 # Il faut seulement changer la ligne avec la unique_labels_
                cluster_labels: np.ndarray              # la fonction est pas impactée par le type de cluster_labels, int ou str
            ) -> Tuple[np.ndarray, List[int]]:          
    """
    Calcule les centres des clusters à partir des embeddings et des étiquettes de cluster.

    INPUTS:
    embeddings : Tableau des vecteurs d'embedding; 
                 shape : np.ndarray, shape = (n_samples, embedding_dim)
                
    cluster_labels : Étiquettes de cluster associées à chaque vecteur. L'étiquette -1 est ignorée dans le calcul des centres.
                     array-like, shape = (n_samples,)
                     attention cluster_labels doit contenir les labels de clusters en strings

    OUTPUTS:
    cluster_centers_ : Chaque ligne est un vecteur représentant  Le centre d'un cluster.
                       np.ndarray, shape = (n_clusters, embedding_dim)
    unique_labels_ : list of int
        Liste triée des labels de clusters (hors -1). Le i-ème vecteur de `cluster_centers_` correspond au i-ème label dans `unique_labels_`.

    """
    
    # Ignorer le cluster -1 (outliers)
    # unique_labels_ = sorted([label for label in set(cluster_labels) if label != -1]) # si est unique_labels_
    unique_labels_ = sorted([label for label in set(cluster_labels) if label != "-1"], key=extract_numeric_key) # si on unique_labels_ commente ici only
    
    # Regrouper les index par label
    label2indices_ = {
        label: np.where(np.array(cluster_labels) == label)[0]
        for label in unique_labels_
    }
    # Calculer les centres
    cluster_centers_ = np.array([
        embeddings[indices].mean(axis=0)
        for label, indices in label2indices_.items()
    ])
    
    return cluster_centers_, unique_labels_

# test
# test = cluster_centers(
#                 embeddings,
#                 cluster_labels
#             )

# cluster_centers_dict = {l : test[0][i] for  l,i in zip(test[1],range(len(test[0])))}
# test





def similarity_matrix(
                    embeddings: np.ndarray,
                    cluster_labels: Union[np.ndarray, list],    # la fonction est pas impactée par le type de cluster_labels, int ou str
                    cluster_summaries: Dict[int, str] = {}      # s'il ya un probleme voir cluster_centers
                ) -> pd.DataFrame: # indices: nom des clusters, colomns: nom des clusters
    """
    Calcule la matrice de similarité cosinus entre les centres des clusters (repose sur `cosine_similarity`).

    INPUTS: 

    embeddings : Tableau des vecteurs d'embedding; 
                 shape : np.ndarray, shape = (n_samples, embedding_dim)
    cluster_labels : Étiquettes de cluster associées à chaque vecteur. L'étiquette "-1" est ignorée dans le calcul des centres.
                     array-like, shape = (n_samples,), 
                     attention cluster_labels doit contenir les labels de clusters en strings

    cluster_summaries : dict, optionnel (par défaut: {})
                        Dictionnaire associant à chaque label de cluster un résumé textuel. Si fourni, ces résumés seront
                        utilisés comme index et colonnes dans la matrice de similarité.

    OUTPUTS:
    sim_matrix_ : DataFrame; Matrice de similarité cosinus entre les centres des clusters.
                    Les lignes et colonnes sont étiquetées par les noms des clusters : soit `Cluster i`,

    """
    
    # Calculer les centres et récupérer les labels
    centers_matrix_, labels_ = cluster_centers(embeddings, cluster_labels)
    sim_matrix_ = cosine_similarity(centers_matrix_)

    # Génération des noms de lignes et colonnes
    if cluster_summaries:
        label_names_ = [f"{label}: {cluster_summaries.get(label, f'Cluster {label}')}" for label in labels_]
    else: 
        label_names_ = [f"{label}" for label in labels_]

    # Construire la matrice DataFrame
    sim_matrix_ = pd.DataFrame(sim_matrix_, index=label_names_, columns=label_names_)
    return sim_matrix_

def max_similarity(
                embeddings: np.ndarray,
                cluster_labels: Union[np.ndarray, list],
                return_argmax: bool = False
            ) -> Union[
                Tuple[float, np.ndarray],
                Tuple[float, np.ndarray, Tuple[int, int]]
            ]:
    """
    Fusionne les deux clusters les plus similaires en mettant à jour cluster_labels.
    
    INPUTS:
        embeddings : comme les fonctions cluster_centers et similarity_matrix,
        cluster_labels : comme les fonctions cluster_centers et similarity_matrix, attention il doit contenir les labels de clusters en strings
        return_argmax : bool, si True retourne aussi les indices de fusion
    
    OUTPUTS:
        max_similarity : valeur de similarité maximale
        new_cluster_labels : labels mis à jour après fusion
        (optionnel) max_labels : labels ("i", "jj") qui ont été fusionnés, (ceux les plus similaires)
    """

    # Obtenir les indices (i, j) du max hors diagonale
    sim_matrix_ = similarity_matrix(embeddings, cluster_labels) # nice, sim_matrix_ est un dataframe indices: nom des clusters, colomns: nom des clusters
    triu_sim = np.triu(sim_matrix_, k=1) # k=1 pour ignorer la diagonale et extract upper triangle
    max_similarity = triu_sim.max()
    max_indices = np.unravel_index(np.argmax(triu_sim), sim_matrix_.shape)

    row_name = sim_matrix_.index[max_indices[0]]   # le nom du cluster i
    col_name = sim_matrix_.columns[max_indices[1]] # le nom du cluster j
    new_label = f"{row_name}_{col_name}"  # i_j

    # Mettre à jour les labels de cluster
    new_cluster_labels = cluster_labels.copy()
    new_cluster_labels = np.where(
                                  np.isin(new_cluster_labels, [row_name, col_name]),
                                  new_label,
                                  new_cluster_labels)

    if return_argmax:
        return max_similarity, new_cluster_labels, (row_name, col_name)
    else:
        return max_similarity, new_cluster_labels
    
# test
# max_similarity(embeddings, cluster_labels, return_argmax=True)

def simplify_cluster_labels_reduced(cluster_labels_reduced):
    simplify2real_labels = {}
    for label in cluster_labels_reduced:
        parts = [int(p) for p in str(label).split("_")]
        new_label = str(min(parts))
        if label not in simplify2real_labels:
            simplify2real_labels[label] = new_label
    return simplify2real_labels


def get_number_clusters_given_seuil(

    embeddings: np.ndarray,
    cluster_labels: Union[np.ndarray, list],
    seuil: float,
    ) -> int:
    """
    Reçoit un  seuil de similarité (distance cosinus) et renvoie le nombre de clusters
    """
    seuil = 1- seuil # Convertir le seuil en similarité (1 - distance cosinus)

    k = len(np.unique(cluster_labels)) - (1 if "-1" in cluster_labels else 0)
    n = k

    cluster_labels_= cluster_labels.copy()
    max_sim, _= max_similarity(embeddings, cluster_labels_)
    temp = cluster_labels_.copy()
    while max_sim >= seuil:
        temp = cluster_labels_.copy()
        max_sim, cluster_labels_ = max_similarity(embeddings, cluster_labels_)
        n -= 1
    return n+1, temp



def compute_inertia(embeddings, cluster_labels):
    """
    Calcule l'inertie des clusters en utilisant la similarité cosinus (la somme de la somme de chaque point à son centre de cluster).

    INPUTS:
    embeddings : np.ndarray
    cluster_labels : np.ndarray

    OUTPUTS:
    inertia : float
        Inertie totale des clusters, mesurée par la somme des distances cosinus entre les points et leurs centres de cluster.
    """
    centroides, label__ = cluster_centers(embeddings, cluster_labels)

    inertia = 0.0
    for i, label in enumerate(label__):
        mask = (cluster_labels == label)
        if mask.sum() <= 1:
            continue

        points = embeddings[mask]
        centroid = centroides[i].reshape(1, -1)
        sims = cosine_similarity(points, centroid)
        inertia += np.sum(1 - sims)

    return inertia




def hirarchical_clustering(
    embeddings: np.ndarray,
    cluster_labels: Union[np.ndarray, list], ### str 
    number_of_clusters: int = 50, 
    return_grouped_labels_per_iteration: bool = False, # veux-tu les labels regroupés par itération ? (information inutile)
    seuil: float = None):
    """
    Regroupe hiérarchiquement les clusters existants jusqu'à atteindre un nombre cible.

    Cette fonction applique une méthode de fusion itérative des clusters existants
    en mesurant leur similarité moyenne dans l'espace des embeddings. À chaque itération, 
    les deux clusters les plus proches sont fusionnés.
    Un seuil de similarité peut être spécifié pour générer une suggestion optimale du nombre
    de clusters, en plus du nombre de clusters demandé.

    INPUTS:
    ----------
    embeddings : np.ndarray
        Matrice des représentations vectorielles des documents (shape : [n_samples, embedding_dim]).

    cluster_labels : array-like
        Labels de clusters initiaux à regrouper. Peut provenir de HDBSCAN, KMeans, etc.

    number_of_clusters : int, default=20
        Nombre cible de clusters après regroupement. La fusion s'arrête dès que ce nombre est atteint.

    return_grouped_labels_per_iteration : bool, default=False
        Si True, retourne aussi la liste des regroupements effectués à chaque étape sous forme de
        paires (cluster_1, cluster_2) fusionnés et le nombre de clusters restants.

    seuil : float, default=0.2
        Seuil de similarité maximale (1-seuil car seuil c'est pour la distance cosine) en dessous duquel une suggestion de regroupement
        est émise. Cela permet d'alerter l'utilisateur si le nombre de clusters souhaité est
        peut-être trop élevé par rapport à la structure des données.

    OUTPUTS:
    -------
    cluster_labels_ : np.ndarray
        Nouveau vecteur de labels après fusion hiérarchique des clusters jusqu'au `number_of_clusters`.

    grouped_labels : list, optional
        Si `return_grouped_labels_per_iteration=True`, retourne aussi une liste de tuples :
        [ ((cluster1, cluster2), n_clusters_restants), ... ] indiquant les fusions successives.

    Notes
    -----
    - Cette méthode ne modifie pas les embeddings, uniquement les `cluster_labels`.
    - Le seuil de similarité est utilisé uniquement pour générer des recommandations,
      la fusion continue même en-dessous du seuil si `number_of_clusters` n'est pas encore atteint.
    - Utile pour réduire des clusters très fragmentés en groupes plus interprétables (ex: post-HDBSCAN).
    """
    
    seuil = 1 - seuil if seuil is not None else seuil   # Convertir le seuil en similarité (1 - distance cosinus)
    k = len(np.unique(cluster_labels)) - (1 if "-1" in cluster_labels else 0)
    n = k
    cluster_labels_ = cluster_labels.copy()
    grouped_labels = [] # ----------------------------------------------------
    are_we_under_seuil = 0
    suggestion = ""
    max_sim, _= max_similarity(embeddings, cluster_labels_)
    if number_of_clusters > k:
        print(f"Le nombre de clusters demandé ({number_of_clusters}) est supérieur au nombre de clusters initiaux ({k}), aucune fusion n'est effectuée.")
        return cluster_labels_
    while n > number_of_clusters:
        max_sim, cluster_labels_, _ = max_similarity(embeddings, cluster_labels_, return_argmax=True) # ----------------------------------------------------
        if seuil and max_sim < seuil and are_we_under_seuil == 0:
            are_we_under_seuil = 1
            if n != number_of_clusters and seuil is not None:
                suggestion = (
                            f"Il est suggéré de réduire le nombre de clusters à {n} pour une meilleure interprétation des données.\n"
                            f"Si c'est le cas, vous regrouperez seulement les clusters qui se ressemblent à plus de {seuil*100:.0f} %.\n"
                        )
                print(suggestion)
            
        n = n - 1
        grouped_labels.append((_, n))

    if seuil is None:
        if return_grouped_labels_per_iteration:
            return cluster_labels_, grouped_labels
    
        return cluster_labels_

    recommended_number_of_clusters = n
    temp = cluster_labels_.copy()
    max_sim, _= max_similarity(embeddings, temp)
    while max_sim >= seuil:
        max_sim, temp= max_similarity(embeddings, temp)
        recommended_number_of_clusters -=1


    if recommended_number_of_clusters+1 != number_of_clusters and are_we_under_seuil==0:
        suggestion = f"Il est suggéré de choisir le nombre de clusters à {recommended_number_of_clusters + (1 if recommended_number_of_clusters < k  else  0)} pour une meilleure interprétation des données." if recommended_number_of_clusters < number_of_clusters else ""
        print(suggestion)

    if return_grouped_labels_per_iteration:
        return cluster_labels_, grouped_labels
    
    return cluster_labels_
# test
# hirarchical_clustering( embeddings, cluster_labels, 24, seuil = 0.25)



def llm_instruction(subject):
    """
    Génère une instruction pour l'API OpenAI en fonction du sujet donné.
    
    INPUTS:
    subject : str
        Sujet ou thème pour lequel l'instruction est générée.

    OUTPUTS:
    str
        Instruction formatée pour guider le modèle LLM dans la génération de résumés thématiques.
    """
    if subject.lower() == "cloud":
        existing_labels = ["Cloud Computing", "Cloud", "Cybersecurity",  "Artificial Intelligence", "Cloud seeding"]
        instruction = """####################  SYSTEM  #####################
You are an expert Topic-Labeling Agent inside an automated topic-modeling pipeline.  
Your mission: deliver one concise label in english (1-3 words) that best represents a single cluster of short texts.
UNIQUENESS: **Do not reuse** a label already present in {existing_labels}; this is strictly PROHIBITED.
We know that the subject is about Cloud Computing, SO DO NOT USE “Cloud” IN THE LABEL.
#####################  CONTEXT  ####################
Labels already used for other clusters (MUST NOT be reused):
{existing_labels}
CRITICALLY forbidden words (must NOT appear anywhere in the label, whole or substring, case-insensitive) IT'S FORBIDDEN:
IT'S 100% FORBIDDEN to label as :"Artificial Intelligence" | "Cybersecurity" | "Cloud Computing" | "Cloud"

###################  off-topic DETECTION  ##################
If the cluster is primarily off-topic (Cloud computing is not addressed in any of the examples provided.), return exactly: DELETE
Examples of OFF-TOPIC include (examples can be in any language):
- HR/promotion posts (DON'T give a label EVEN IF it's about cloud computing): job offers, CERTIFICATIONS, celebrations, “open to work”, networking/self-promo, do
- Weather “clouds”: I DO NOT TOLERATE ANY LABEL ABOUT WEATHER OR CLIMATE RELATED CONTENT, such as cloud cover, cloud seeding, or natural phenomena.
- e.g. of  Weather “clouds”: forecasts, water, meteorology, CLOUD SEEDING, or nuclear clouds
- Pop culture / products named “Cloud” but non-IT: Cloud Strife (FF7), perfumes (Ariana Grande Cloud), sneakers (Cloudfoam), “cloud bread”, “cloud wash” apparel
- Causal discussions or random comments"
It is NOT recommanded TO LABEL AS DELETE, EXCEPT UNDER ONE CONDITION: you are sure that we don't talk about cloud computing (else give a label).
If any of these apply, do not derive a topic → RETURN DELETE.


#####################  RULES  ######################
1. LENGTH: 1 to 2 words (MAXIMUM 3 WORDS), **Title Case**, no punctuation, no hashtags, no quotation marks.
2) CLOUD SPECIFICITY — Include at least **one concrete Cloud mechanism/lever** (service, practice, metric, policy, compliance...).
Examples (indicative):
• Cost/Operations: FinOps, Egress Fees, Savings Plans, Spot Instances, Rightsizing, Autoscaling, Capacity Reservation, Cost Anomaly.....
• Infra/Network: VPC Peering, Private Link, Transit Gateway, NAT Gateway, L7 WAF, CDN Invalidation, Edge Caching, Load Balancing.....
• Data: Data Residency, Sovereign Cloud, Object Lock, S3 Lifecycle, Cross-Region Replication, Backup Rotation.....
• Security: KMS Rotation, BYOK, Zero Trust, OIDC Federation, IAM Boundaries, Posture Management, Confidential Computing.....
• Kubernetes: Cluster Autoscaler, Pod Security, Service Mesh, Sidecar Injection, Nodegroups, Autopilot.....
• Regulatory: GDPR, NIS2, DORA, SecNumCloud, HDS, CNIL Sanction.....
3) DOMINANT ENTITY — If a provider/regulator/event dominates, use canonical name **+ qualifier** (action/mechanism/incident/region):
e.g., “AWS Outage”, “Azure Pricing”, “GCP Egress”, “CNIL Sanction”, “EU NIS2”, “SecNumCloud Certification”, “OVHcloud Outage Gravelines”.
4) UNIQUENESS — The label **must not** appear in {existing_labels}. On conflict, append a **concrete differentiator** (service/region/period):
e.g., “Egress Fees EU”, “S3 Outage eu-west-3”, “FinOps Tagging”.
5) BANS — Disallow “Cloud” or “Cloud Computing” (alone or as generic prefix). Disallow single generic words like “AWS”, “Azure”, “Security”.
Disallow vague/meta labels (“Update”, “News”, “Trends”, “Innovation”), symbols (+ / :), emojis, hashtags, quotes.
6) OUTPUT FORMAT — Respond with **plain text containing only the label** — nothing else.
7° NO SYMBOLS → No “+”, “/”, “:”. Only a single space if two words.  
8) SELF-CHECK — Before returning, verify: length preference, Cloud specificity, uniqueness, forbidden terms, and off-topic→DELETE. If ANY rule fails, revise.
(Think step-by-step internally, but reveal only the final label.)
"""
        user_assistant =  [{
                                "role": "user",
                                "content": """(HR / Self-Promotion / Networking)
                        1) I'm excited to announce I passed my AWS Solutions Architect exam! #certified....
                        1) Happy to share I just completed AWS cloud academy foundation course, also a cloud coputing training!
                        2) Open to work as a cloud engineer — happy to network and connect.
                        4) Sharing my updated CV as a cloud security agent — referrals appreciated, thank you.
                        5) Honored to speak at ACME Summit next week — join my session!
                        6) Celebrating my new certification in Cloud Security — proud to be part of the community!
                        7) Another milestone unlocked! Attended a game-changing talk on serverless technologies and cloud scalability. Let's build the future! #CloudGeek #ProfessionalDevelopment
                        8) THRILLED to announce My new AWS certification!, next step: Azure Fundamentals certification, and then GCP Associate Cloud Engineer certification!
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 2) Hiring / Recruiting
                            {
                                "role": "user",
                                "content": """(Hiring / Recruiting)
                        1) We're hiring a Senior Backend Engineer — apply on our careers page.
                        2) Looking for a Cloud Architect contractor for a 6-month mission.
                        3) Recruiting DevOps and cloud interns for summer 2025 — DM for details.
                        4) Join our team as a Cloud Security Engineer — remote work available.
                        5) Hiring a Cloud Solutions Architect with expertise in AWS and Azure."""
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 3) Pop Culture / Products named “Cloud/Cyber/AI” (Non-IT)
                            {
                                "role": "user",
                                "content": """(Pop Culture / Products named “Cloud/Cyber/AI” — Non-IT)
                        1) Cloud Strife is my favorite Final Fantasy VII character.
                        2) Ariana Grande's “Cloud” perfume is back in stock this weekend.
                        3) These Cloudfoam sneakers are incredibly comfortable for running.
                        4) Cloud bread recipe with only three ingredients is trending again.
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 4) Weather / Natural Phenomena / Non-IT “clouds”
                            {
                                "role": "user",
                                "content": """(Weather / Cloud seeding/ Natural Phenomena / Non-IT “clouds”)
                        1) Low clouds and swell along the Brittany coast this morning.
                        2) The nuclear cloud was visible for miles in archival footage.
                        3) Ophiuchi cloud images from the observatory look stunning.
                        4) Cloud seeding can stimulate rainfall using silver iodide.
                        5) Satellite imagery shows dense cloud cover over the Alps today.
                        6) Cloud seeding is harmful to the environment and has sparked significant controversy in society.
                        6) The cloud seeding process involves injecting particles into the atmosphere to encourage precipitation.
                        7) The cloud of polluted smoke disrupts the environment, which leads to a decrease in the formation of rain clouds. 
                        8) flooded streets, that are caused by CLOUD SEEDING!! STOP CLOUD SEEDING NOW!"""
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 5) Generic chit-chat / Off-topic
                            {
                                "role": "user",
                                "content": """(Generic chit-chat / Off-topic)
                        1) hi, hello
                        2) It's amazing how much we can learn from the past.
                        3) lol, you crack me up
                        4) Weekend plans: reading a book of cloud computing.
                        5) My travel diary from Rome is finally online — photos on cloud drive."""
                            },
                            {"role": "assistant", "content": "DELETE"}]
        
    if subject.lower() == "ai":
        
        existing_labels = ["Artificial Intelligence", "Cloud", "Cybersecurity",  "AI"]
        instruction = """
####################  SYSTEM  #####################
You are an expert Topic-Labeling Agent inside an automated topic-modeling pipeline for Artificial Intelligence content.  
Your mission: deliver one concise label in english (1-3 words) that best represents a single cluster of short texts.
UNIQUENESS: **Do not reuse** a label already present in {existing_labels}; this is strictly PROHIBITED.
We know that the subject is about Artificial Intelligence, SO DO NOT USE "Artificial Intelligence" or "AI" IN THE LABEL (or as a label).
#####################  CONTEXT  ####################
Labels already used for other clusters (MUST NOT be reused):
{existing_labels}
CRITICALLY forbidden words (must NOT appear anywhere in the label, whole or substring, case-insensitive) IT'S 100% FORBIDDEN that the label contains, or to label as:
"Artificial Intelligence" OR "AI" 

###################  off-topic DETECTION  ##################
If the cluster is primarily off-topic (Artificial Intelligence is not addressed in any of the examples provided.), return exactly: DELETE
Examples of OFF-TOPIC include (examples can be in any language):
- HR/promotion posts (DON'T give a label EVEN IF it's about Artificial Intelligence): job offers, CERTIFICATIONS, celebrations, “open to work”, networking/self-promo
- Pop culture / products using "AI" in a non-IT sense: games (AI Dungeon, AI War), perfumes, shoes, recipes...
- Misleading names/acronyms: the Japanese artist "Ai", “Institut Agricole (IA)”, lifestyle, weather, or unrelated news...
- Causal discussions or random comments
It is NOT recommanded TO LABEL AS DELETE, EXCEPT UNDER ONE CONDITION: you are sure that we don't talk about Artificial Intelligence (else give a label).
If any of these apply, do not derive a topic → RETURN DELETE.


#####################  RULES  ######################
1. LENGTH: 1 to 2 words (MAXIMUM 3 WORDS), **Title Case**, no punctuation, no hashtags, no quotation marks.
2) AI SPECIFICITY — The label must reflect a concrete AI mechanism, task, method, safety/regulatory concept, evaluation, or deployment lever.
   Examples (indicative):
   - Learning/Optimization: RLHF, DPO, LoRA, QLoRA, Distillation, Curriculum Learning...
   - Inference/Serving: vLLM Inference, KV Cache, Speculative Decoding, Flash Attention...
   - Retrieval/Knowledge: RAG Evaluation, Vector Search, Data Provenance, Deduplication....
   - Safety/Security: Prompt Injection, Jailbreaks, Red Teaming, Alignment, Toxicity Mitigation, Watermarking...
   - Evaluation/Quality: Hallucination, MMLU Benchmarks, Truthfulness, Robustness...
   - Multimodality: Vision Language, Speech Recognition, Audio Transcription, OCR...
   - Privacy/Policy: Differential Privacy, Federated Learning, Data Minimization...
   - Regulation/Governance: Model Governance, Risk Management, Impact Assessment...
   - LLMOps/MLOps: Observability, Guardrails, Canary Rollout, Feature Store...
3) DOMINANT ENTITY — If a provider/regulator/event dominates, use canonical name **+ qualifier** (action/mechanism/incident/region):
   Examples: "OpenAI Lawsuit", "Anthropic Alignment", "Meta Distillation", "CNIL Sanction", "EU Regulation", "NVIDIA Inference".
4) UNIQUENESS — The label **must not** appear in {existing_labels}. On conflict, append a **concrete differentiator** (service/region/period):
   Examples: "Hallucination Search", "RAG Evaluation Legal", "Alignment Healthcare".
5) BANS — Disallow the standalone token "AI" and the phrase "Artificial Intelligence".
   Disallow vague/meta labels ("Update", "News", "Trends", "Innovation"), single generic words ("Models", "Security"), symbols (+ / :), emojis, hashtags, quotes.
6) OUTPUT FORMAT — Respond with **plain text containing only the label** — nothing else.
7) NO SYMBOLS → No “+”, “/”, “:”. Only spaces between words.  
8) SELF-CHECK — Before returning, verify: length preference, AI specificity, uniqueness, forbidden terms, and off-topic→DELETE. If ANY rule fails, revise.
(Think step-by-step internally, but reveal only the final label.)
"""
        user_assistant = [{
                                "role": "user",
                                "content": """(HR / Self-Promotion / Networking/ Personal Branding))
                        1) Thrilled to announce I completed the Generative AI specialization! #certificate
                        1) Sharing my updated CV for LLM research internships — referrals appreciated.
                        2) Proud to add a new AI Safety certification to my profile (RLHF & red teaming).
                        4) I just published a Medium article on prompt engineering — please like and share!
                        5) Passed the MLOps Foundations exam — next up: LLMOps Professional!
                        6) Another milestone unlocked: attended an inspiring talk on prompt engineering — let's build the future!
                        7) Excited to showcase my GenAI chatbot demo — DM me for a walkthrough.
                        8) Honored to speak at Global AI Summit about fine-tuning — join my session!
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 2) Hiring / Recruiting
                            {
                                "role": "user",
                                "content": """(Hiring / Recruiting)
                        1) We're hiring a Senior LLM Engineer — apply on our careers page.
                        2) Seeking a Prompt Engineer for production chatbots (remote).
                        3) Recruiting MLOps / LLMOps interns for summer 2025 — DM for details.
                        4) Join our team as an AI Product Manager (GenAI) — EU time zones preferred.
                        5) Hiring a Research Scientist (NLP/Deep Learning) with instruction-tuning experience.
                        6) Looking for a Data Scientist with expertise in RAG and vector search.
                        7) FOR MACHINE LEARNING ENGINEERS!!: for a Lead ML Engineer to own our RAG platform in healthcare.
                        8) Open roles: Data Scientist (computer vision), LLM Evaluator, and MLOps Architect.
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 3) Pop Culture / Products named “AI” (Non-IT)
                            {
                                "role": "user",
                                "content": """(Pop Culture / Non-AI uses of “AI” — NOT artificial intelligence)
                        1) Playing AI Dungeon with friends all weekend.
                        2) Loved the movie “A.I. Artificial Intelligence” by Spielberg.
                        3) Concert tonight by the Japanese artist Ai — so excited!
                        4) My Adobe Illustr ator (.ai) logo file is corrupted — need help!
                        5) Bought a premium .ai domain from Anguilla for my portfolio.
                        6) Air India flight AI-101 was delayed by two hours.

                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 5) Generic chit-chat / Off-topic
                            {
                                "role": "user",
                                "content": """(Generic chit-chat / Off-topic)
                        1) hi, hello
                        2) It's amazing how much we can learn from the past.
                        3) lol, you crack me up
                        4) Weekend plans: reading a book of computer vision.
                        5) My travel diary from Rome is finally online — photos on insta."""
                            },
                            {"role": "assistant", "content": "DELETE"},]
    
    if subject.lower() == "cyber":
        existing_labels = ["Artificial Intelligence", "Cybersecurity", "Cyber"]
        instruction = """
####################  SYSTEM  #####################
You are an expert Topic-Labeling Agent inside an automated topic-modeling pipeline for Cyber security content.  
Your mission: deliver one concise label in english (1-3 words) that best represents a single cluster of short texts.
UNIQUENESS: **Do not reuse** a label already present in {existing_labels}; this is strictly PROHIBITED.
We know that the subject is about Cyber security, SO DO NOT USE "Cyber security" AS A LABEL (or IN a label).
#####################  CONTEXT  ####################
Labels already used for other clusters (MUST NOT be reused):
{existing_labels}
CRITICALLY forbidden words (must NOT appear anywhere in the label, whole or substring, case-insensitive) IT'S 100% FORBIDDEN that the label contains, or to label as:
"Cyber Security"

###################  off-topic DETECTION  ##################
If the cluster is primarily off-topic (Cyber Security is not addressed in any of the examples provided.), return exactly: DELETE
Examples of OFF-TOPIC include (examples can be in any language):
- HR/promotion posts (DON'T give a label EVEN IF it's about Cyber Security): job offers, CERTIFICATIONS, celebrations, “open to work”, networking/self-promo
- Pop culture / products using "Cyber" in a non-IT sense: Cybertruck, Cyberpunk cosplay....
- Retail events or lifestyle: Cyber Monday deals, fashion, travel/weather...
- Causal discussions or random comments
It is NOT recommanded TO LABEL AS DELETE, EXCEPT UNDER ONE CONDITION: you are sure that we don't talk about Cyber Security (else give a label).
If any of these apply, do not derive a topic → RETURN DELETE.


#####################  RULES  ######################
1. LENGTH: 1 to 2 words (MAXIMUM 3 WORDS), **Title Case**, no punctuation, no hashtags, no quotation marks.
2) CYBER SPECIFICITY — The label must reflect a concrete cybersecurity mechanism, technique, asset, control, threat, incident, regulation, or metric.
   Indicative examples (not exhaustive):
   - Threats/Exploitation: Phishing, MFA Fatigue, Credential Stuffing, SQL Injection, XSS, CSRF, SSRF, RCE, BGP Hijack, DNS Poisoning....
   - Vulnerabilities/Advisories: CVE-2021-44228 (Log4Shell), CVE-2023-4966 (CitrixBleed), CISA KEV, Patch Tuesday....
   - Malware/Actors: Ransomware, LockBit, BlackCat, Cl0p, APT29, Lazarus, TA505....
   - Detection/Response: YARA Rules, Sigma Rules, IOC Hunting, Threat Intelligence, Incident Response, Forensics, EDR, XDR, SIEM, SOAR....
   - Identity/Access: MFA, Passkeys, FIDO2, SSO, SAML, OIDC, IAM, PAM, Zero Trust....
   - Network/Cloud Sec: WAF Bypass, TLS 1.3, PKI, Certificate Pinning, DDoS Mitigation, S3 Public Access, Secret Leakage....
   - Governance/Compliance: NIS2, DORA, SOC 2, ISO 27001, CNIL Sanction, SBOM, Supply Chain....
3) DOMINANT ENTITY — If a provider/regulator/event dominates, use canonical name **+ qualifier** (action/mechanism/incident/region):
   Examples: "CISA Advisory", "Microsoft Patch Tuesday", "CNIL Sanction", "LockBit Ransomware", "OpenSSL Vulnerability".
4) UNIQUENESS — The label **must not** appear in {existing_labels}. On conflict, append a **concrete differentiator** (service/region/period):
   Examples: "Phishing QR", "Ransomware Healthcare", "DDoS Mitigation EU".
5) BANS — Disallow the standalone phrase "Cyber Security".
   Disallow vague/meta labels ("Update", "News", "Trends", "Innovation"), single generic words ("Security", "Attack"), symbols (+ / :), emojis, hashtags, quotes.
6) OUTPUT FORMAT — Respond with **plain text containing only the label** — nothing else.
7) NO SYMBOLS → No “+”, “/”, “:”. Only spaces between words.  
8) SELF-CHECK — Before returning, verify: length preference, cybersecurity specificity, uniqueness, forbidden terms, and off-topic→DELETE. If any rule fails, revise.
(Think step-by-step internally, but reveal only the final label.)
"""
        user_assistant = [{
                                "role": "user",
                                "content": """(HR / Self-Promotion / Networking/ Personal Branding))
                        1) Thrilled to pass CISSP — thanks to my mentors!
                        1) Open to work as a SOC analyst — let's connect.
                        2) Proud to add OSCP and Security+ to my profile.
                        4) I just published a Medium article on SIEM detection use cases — please like and share
                        5) Passed Blue Team Level 1 — next up: GCIA!
                        6) Another milestone: attended an inspiring talk on Zero Trust — let's build the future!
                        7) Honored to speak at Global Cyber Summit about incident response — join my session!
                        8) Excited to showcase my phishing-awareness workshop — DM me for a walkthrough.
                        9) Sharing my updated CV for DFIR roles — referrals appreciated.
                        10) Celebrating 2 years at CompanyX's CERT — grateful to my team!
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 2) Hiring / Recruiting
                            {
                                "role": "user",
                                "content": """(Hiring / Recruiting)
                        1) We're hiring a Senior SOC Analyst for 24/7 operations — apply on our careers page.
                        2) Seeking a Threat Hunter with EDR/XDR and Sigma/YARA experience (remote).
                        3) Recruiting DFIR interns for summer 2025 — DM for details.
                        4) Join our team as a Cloud Security Engineer (AWS/Azure/GCP) — EU time zones preferred.
                        5) Hiring a Red Team Operator / Penetration Tester with OSCP/OSCE.
                        6) Looking for an AppSec Engineer to own SAST/DAST and SBOM pipeline.
                        7) Open roles: GRC Analyst (ISO 27001, SOC 2), NIS2 Program Manager, DORA Lead, and OT/ICS Security Engineer needed for industrial networks (IEC 62443).
                        8) Contract: IAM/PAM Architect (Okta, Azure AD, CyberArk) — 6 months.
                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 3) Pop Culture / Products named “cyber” (Non-IT)
                            {
                                "role": "user",
                                "content": """(Pop Culture / Non-Cyber uses of “Cyber” — NOT cybersecurity)
                        1) Cybertruck memes are everywhere this week.
                        2) Loved the Cyberpunk cosplay at last night's festival.
                        3) Cyber Monday deals start tonight — huge discounts!
                        4) Cybergoth outfits are back in style this season.
                        5) The Cyberdog fashion store launched a new collection.
                        6) New café named “Cybercafé District” opened downtown.
                        7) My e-bike is the “CyberX” model — amazing suspension.
                        8) Street art with the word “CYBER” just popped up in our neighborhood.

                        """
                            },
                            {"role": "assistant", "content": "DELETE"},

                            # 5) Generic chit-chat / Off-topic
                            {
                                "role": "user",
                                "content": """(Generic chit-chat / Off-topic)
                        1) hi, hello
                        2) It's amazing how much we can learn from the past.
                        3) lol, you crack me up
                        4) Weekend plans: reading a book of cyber attacks and cyber security.
                        5) My travel diary from Rome is finally online — photos on insta."""
                            },
                            {"role": "assistant", "content": "DELETE"}]
    

    return existing_labels, instruction, user_assistant
