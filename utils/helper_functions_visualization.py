from utils.helper_functions import *
# if there is a problem,  import other libraries




def visualize_clusters(
    df: pd.DataFrame,  # df doit contenir 'text', 'annotation', 'x', 'y', une colonne pour les clusters et une pour les probabilités
    reduced_cluster_centers_: Dict[int, Tuple[float, float]],
    cluster_summaries: Dict[int, str],
    max_text_showed_length: int = 80,
    title: str = "Clustering UMAP + HDBSCAN — Topic Label per Cluster",
    width: int = 1300,
    height: int = 750,
    size_points: int = 7,
    opacity_points: float = 0.8,
    size_text: int = 9,
    opacity_text: float = 0.9,
    col_topic: str = "topic",
    col_proba : str = "proba",
    col_annotation: str = None
) -> None:
    """
    Affiche une visualisation interactive 2D des clusters, annotés avec leurs résumés textuels.

    Cette fonction génère un graphique de dispersion (souvent basé sur une réduction type UMAP)
    en coloriant chaque document selon son cluster (`topic`) et en annotant chaque centre de cluster 
    avec un résumé (topic label) fourni via `cluster_summaries`. L'objectif est d'assister
    l'analyse sémantique et la validation visuelle de regroupements thématiques.

    INPUTS:
    -------
    df : pd.DataFrame
        DataFrame contenant les colonnes suivantes :
        - 'text' : texte original du document
        - 'annotation' : résumé textuel du document (souvent injecté après annotation LLM)
        - 'proba' : probabilité d'appartenance au cluster (ex: HDBSCAN)
        - 'x', 'y' : coordonnées 2D (souvent issues d'UMAP ou t-SNE)
        - 'topic' : identifiant du cluster

    reduced_cluster_centers_ : dict[int, Tuple[float, float]]
        Dictionnaire `{label: (x_center, y_center)}` des centres des clusters dans le plan réduit.

    cluster_summaries : dict[int, str]
        Dictionnaire `{label: résumé}` avec les étiquettes thématiques par cluster.

    max_text_showed_length : int, optional (default = 80)
        Longueur maximale (en caractères) à afficher pour le texte dans le `hover_data`.

    title : str, optional
        Titre principal affiché en haut du graphique (par défaut : UMAP + HDBSCAN).

    width : int, optional
        Largeur du graphique en pixels.

    height : int, optional
        Hauteur du graphique en pixels.

    size_points : int, optional
        Taille des points (documents) affichés.

    opacity_points : float, optional
        Opacité des points (entre 0.0 et 1.0).

    size_text : int, optional
        Taille de la police utilisée pour afficher les labels des clusters.

    opacity_text : float, optional
        Opacité des labels (entre 0.0 et 1.0).

    OUTPUT:
    -------
    None
        Affiche directement un graphique Plotly interactif dans le notebook ou l'environnement courant.
        Aucune valeur n'est retournée.
    """
    df = df.copy()
    # On crée deux colonnes internes utilisées par la suite
    df["topic"] = df[col_topic]
    df["proba"] = df[col_proba]


    def smart_truncate(text, max_text_showed_length=max_text_showed_length): # si un texte est trop long, on le tronque
        if len(text) <= max_text_showed_length:
            return text
        return text[:max_text_showed_length].rsplit(' ', 1)[0] + "..." 

    _ = df.copy()
    _["text"] = _["text"].apply(smart_truncate) 
    _["topic"] = _["topic"].astype(str)  # Assurez-vous que 'topic' est un entier
    fig = px.scatter(
                    _, x="x", y="y", color="topic",
                    hover_data={
                                "text": True,   
                                "annotation": True,
                                "proba": True,
                                "x": False,
                                "y": False
                                },
                    width=width,
                    height=height,
                    color_continuous_scale="HSV",
                    )

    # On améliore les points
    fig.update_traces(
        marker=dict(size=size_points, opacity=opacity_points), # size: taille des points en pixels, opacity: opacité
        selector=dict(mode="markers"),    # n'applique ces changements qu'aux traces qui sont des markers (points)
    )

    
    
    for label, (x_center, y_center) in reduced_cluster_centers_.items():
        if label == "-1":
            continue  # skip noise cluster
        
        summary = cluster_summaries[label]
        
        fig.add_annotation(
            x=x_center,
            y=y_center,  # décalage vertical
            text=summary,
            showarrow=False,
            yshift=0,
            font=dict(
                size=size_text,
                color="black",
                family="Arial Black"
            ),
            opacity=opacity_text,
        )
    # Layout final
    fig.update_layout(
        template="plotly_white",
        title=title,
        coloraxis_showscale=False 
    )

    fig.show()



def plot_heatmap_inter_cluster_cosine_similarity(
                    embeddings: np.ndarray,
                    cluster_labels: Union[np.ndarray, list],
                    cluster_summaries: Dict[int, str] = {},
                    title: str ="Inter-Cluster Cosine Similarity Heatmap (with Labels)",
                    width: int =1200,
                    height: int =1000
                ) -> None:
    """
    Affiche une heatmap interactive des similarités cosinus entre les centres de clusters.

    INPUTS:
    embeddings : Comme les fonctions cluster_centers et similarity_matrix,
    cluster_labels : Comme les fonctions cluster_centers et similarity_matrix,
    cluster_summaries : Comme la fonction similarity_matrix,

    OUTPUTS:
    Une figure Plotly interactive représentant la matrice de similarité cosinus entre les centres des clusters.
    Chaque cellule indique le degré de proximité entre deux clusters, avec un code couleur et des labels explicites.
    """
    
    sim_df = similarity_matrix(embeddings, cluster_labels, cluster_summaries)

    fig = px.imshow(
        sim_df,
        text_auto=False,
        aspect="auto",
        labels=dict(x="Cluster", y="Cluster", color="Cosine Similarity"),
        title=title
    )

    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.show()


def plot_dendrogram_with_labels(
                    embeddings: np.ndarray,
                    cluster_labels: Union[np.ndarray, list],
                    cluster_summaries: Dict[int, str] = {},
                    title: str = "Hierarchical Clustering of Cluster Centers (Named Labels)",
                    width: int = 1200,
                    height: int = 35
                ) -> None:
    """
    Affiche un dendrogramme interactif basé sur la similarité entre les centres de clusters.

    INPUTS:
    embeddings : Comme les fonctions cluster_centers et similarity_matrix,
    cluster_labels : Comme les fonctions cluster_centers et similarity_matrix,
    cluster_summaries : Comme la fonction similarity_matrix,

    OUTPUTS:
    Un dendrogramme hiérarchique interactif (via Plotly) représentant les relations
    entre les clusters selon la distance 1 - similarité cosinus entre les centres.

    NOTES:
    La distance entre deux clusters est définie comme `1 - cosine_similarity`.
    """

    # Calcul des centres
    centers_matrix_, labels_ = cluster_centers(embeddings, cluster_labels)

    # Calcul de la matrice de distance (1 - similarité cosinus)
    dist_matrix = 1 - similarity_matrix(embeddings, cluster_labels).to_numpy()
    np.fill_diagonal(dist_matrix, 0.0)  # sécurité pour squareform

    # Conversion en format condensé pour linkage
    condensed_dist = squareform(dist_matrix, checks=False)

    # Calcul de la matrice de linkage hiérarchique
    linkage_matrix = linkage(condensed_dist, method="ward")

    # Étiquettes du dendrogramme
    dendro_labels = [
        f"{label}  {cluster_summaries.get(label, f'Cluster {label}')}"
        for label in labels_
    ]

    # Création du dendrogramme interactif
    fig = ff.create_dendrogram(
        centers_matrix_,
        orientation='left',
        labels=dendro_labels,
        linkagefun=lambda _: linkage_matrix
    )

    fig.update_layout(
        width=width,
        height= height * len(labels_),
        title=title,
        margin=dict(l=240, r=40, t=60, b=40)
    )

    fig.show()
# test
# plot_dendrogram_with_labels(embeddings, cluster_labels, cluster_summaries)




def plot_elbow_and_clusters_vs_threshold(
    embeddings,
    cluster_labels,
    seuil=None,
    exemple_how_to_read_graph_given_that_seuil=4,
    width=1000,
    height=500
):
    """
    Trace :
    - À gauche : nombre de clusters en fonction du seuil
    - À droite : inertie en fonction du nombre de clusters
    """
    seuil = list(np.arange(0, 0.1, 0.04)) + list(np.arange(0.12, 0.26, 0.015)) + list(np.arange(0.26, 0.8, 0.03)) + [1] if seuil is None else seuil
    number_of_clusters = []
    inertias = []
    temp = cluster_labels.copy()

    for s in seuil[:-1]:
        n, temp = get_number_clusters_given_seuil(embeddings, temp, s)
        inertia = compute_inertia(embeddings, temp)
        number_of_clusters.append(n)
        inertias.append(inertia)
    number_of_clusters.append(1)  # Ajouter le dernier nombre de clusters
    inertias.append(compute_inertia(embeddings, np.array([temp[0]]*len(temp))))  # Ajouter l'inertie finale
    # Récupérer le point d’exemple
    x_target = seuil[exemple_how_to_read_graph_given_that_seuil]  
    index = seuil.index(x_target)
    y_target = number_of_clusters[index]

    # Création d'un graphe à deux colonnes
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Nombre de clusters en fonction du seuil de similarité",
            "Inertie en fonction du nombre de clusters"
        ],
        horizontal_spacing=0.15
    )

    # ➤ Graphe 1 : nombre de clusters vs seuil
    initial_cluster_count = number_of_clusters[0]
    reduction_percent = [
        f"{100 * (initial_cluster_count - n) / initial_cluster_count:.1f}%" for n in number_of_clusters
    ]

    fig.add_trace(go.Scatter(
        x=seuil,
        y=number_of_clusters,
        mode='lines+markers',
        name="Nombre de clusters",
        marker=dict(size=5),
        line=dict(shape='linear'),
        customdata=np.stack((number_of_clusters, reduction_percent), axis=-1),
        hovertemplate=(
            "Seuil = %{x:.2f}<br>" +
            "Clusters = %{customdata[0]}<br>" +
            "Réduction = %{customdata[1]}<extra></extra>"
        )
    ), row=1, col=1)


    # ➤ Ajouter annotation sur graphe 1
    fig.add_annotation(
        x=x_target,
        y=y_target,
        text=(
            f"<b>Exemple : x = {x_target:.2f}, y = {y_target}</b><br>"
            f"Si la distance entre deux clusters est < {x_target:.2f},<br>"
            f"on les regroupe. On obtient {y_target} clusters."
        ),
        showarrow=True,
        arrowhead=2,
        ax=200,
        ay=-30,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
        row=1,
        col=1
    )

    # ➤ Graphe 2 : inertie vs nombre de clusters
    fig.add_trace(go.Scatter(
        x=number_of_clusters[:-1],
        y=inertias[:-1],
        mode='lines+markers',
        name="Inertie intra-cluster",
        marker=dict(size=5),
        line=dict(shape='linear')
    ), row=1, col=2)

    # Mise en page finale
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False
    )

    fig.update_xaxes(title_text="Seuil (distance)", row=1, col=1)
    fig.update_yaxes(title_text="Nombre de clusters", row=1, col=1)

    fig.update_xaxes(title_text="Nombre de clusters", row=1, col=2)
    fig.update_yaxes(title_text="Inertie", row=1, col=2)

    fig.show()
# Test de la fonction plot_elbow_and_clusters_vs_threshold
# plot_elbow_and_clusters_vs_threshold(embeddings, cluster_labels)