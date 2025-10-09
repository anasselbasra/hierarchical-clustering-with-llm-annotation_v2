# Helper Functions – Documentation Guide

This document provides a detailed explanation of each function defined in the `helper_functions.py` module.

Its primary goal is to make the logic and purpose of these utilities accessible to users who wish to **leverage the pipeline without diving into the source code**.  
It also serves as a **starting point for contributors** looking to understand, modify, or extend the system.

Where relevant, additional context (e.g., modeling rationale, usage examples) will be provided directly in the notebooks where these functions are invoked.

All functions in `helper_functions.py` are designed to keep the main notebooks **clean**, **modular**, and **reproducible**.

# LLM annotation and hierarchichal clustering 

## Function: `clean_text`

**Description**  
Removes emojis and normalizes whitespace in a string.

This utility is meant to clean up raw input—typically from user-generated content—by stripping emojis and reducing redundant spaces. It ensures a minimal, standardized format for downstream text processing.

**Arguments**
- `text` (`str`): The raw input string that may contain emojis and irregular spacing.

**Returns**
- `str`: A cleaned version of the input string, without emojis and with normalized whitespace.

**Usage Context**
This function is helpful during early preprocessing, especially when dealing with noisy inputs from platforms like Twitter, YouTube comments, or TikTok captions.


## Function: `farthest_sampling_high_proba` & `select_examples`

### Context: Why Intelligent Sampling Matters for LLM Annotation

This function plays a central role in the automation of cluster annotation using Large Language Models (LLMs).  
The idea is intuitive: provide representative text samples from a cluster, and let the LLM generate a topic label.

However, the key challenge is: **how should we select these representative texts?**

The naive solution would be to send the entire cluster to the LLM. While this increases annotation accuracy by covering the full diversity of the cluster, it is **infeasible in practice** due to:
- High computational and financial cost,
- Increased latency.

We therefore designed a **smart sampling strategy** to choose a small subset of texts that captures the **semantic diversity** of the cluster.

### Baseline Methods (And Their Limitations)

#### 1. **Random Sampling**
- Fast and simple.
- But risks selecting:
  - Noisy texts (e.g., with low membership probability),
  - Redundant examples (texts too similar to each other).

#### 2. **Biased Random Sampling (Center-Focused)**
- Assumes that central points (close to the cluster centroid) are most representative.
- Selects 50% of samples near the center, 50% elsewhere.
- Limitations:
  - Not generalizable (especially with HDBSCAN which uses density rather than centroid-based logic).
  - May still select redundant or noisy samples due to random choice.

### Our Approach: Probabilistic and Dispersed Sampling

We propose a **hybrid strategy**:
- Select a subset of **high-confidence points** (membership probability ≥ threshold) that are also **maximally dispersed**.
- Add a few **low-confidence (but non-noise) points**, also dispersed, to enrich the LLM’s view of edge cases.

---

## Function: `farthest_sampling_high_proba`

**Description**  
Selects `k` points from a cluster:
- That have high membership probability (above a threshold),
- And are **as far apart as possible** in the original embedding space.

**How it works**:
1. Filter the input `indices` to keep only those with `proba ≥ threshold`.
2. Compute the geometric center (mean) of the full cluster.
3. Select the first point as the one **farthest from the center**.
4. Iteratively select the next point as the one that **maximizes the minimum distance** to all already selected points.

> We compute distances using the **original embeddings**, not UMAP-reduced ones.  
> This is because UMAP does not preserve pairwise distances—it only maintains **manifold structure**.

**Why not brute-force the best combination?**  
Because selecting the optimal combination of `k` maximally dispersed points is a **combinatorial problem**—exponential in `k`. Our greedy method is a scalable approximation.

**How is "farthest" defined?**  
At each step, for every remaining candidate point, we compute its distance to all previously selected points.  
We then select the point whose **minimum distance** to any selected point is the **largest**—this avoids selecting a point that's too close to any already-selected one.

**Arguments**
- `embeddings` (`np.ndarray`): The full embedding matrix, shape `(n_samples, n_features)`.
- `indices` (`List[int]`): Indices of a specific cluster.
- `proba_list` (`np.ndarray`): Membership probabilities for all points.
- `proba_threshold` (`float`): Minimum threshold to define high-confidence points.
- `k` (`int`): Number of points to select.

**Returns**
- `List[int]`: Selected point indices (within the cluster), high-confidence and dispersed.

> **Note:** If no point exceeds the threshold, the function returns an empty list.

---

## Function: `select_examples`

**Description**  
Wraps the sampling strategy for the **entire set of clusters**.

Its goal is to decide:
- **How many examples to select per cluster**, based on its size.
- **How many to take from high-confidence vs. low-confidence points**.

**Sampling policy**:

Let `cluster_size = len(cluster_points)`  
Let `min_size = 17` and `partition = 0.02` (2%)

We compute: examples_size = max(min(min_size, cluster_size), int(cluster_size * partition))


This gives:
- If cluster size ≤ 16: take all points.
- If 17 ≤ size ≤ ~900: take 17 examples.
- If size > 900: take 2% of the points.

Then we split:
- `high_proba_k = max(13, int(0.75 * examples_size))`
- `low_proba_k = max(4, int(0.25 * examples_size))`

Thresholds used:
- High proba: ≥ 0.8 → core topic
- Low proba: [0.3, 0.8) → transitional content
- Below 0.3 → noise, not considered

**Why this matters**:  
This mixed strategy helps the LLM capture:
- The **central theme** of the cluster (via high-proba samples),
- Peripheral or ambiguous content (via low-proba, non-noise samples).

**Arguments**
- `reduced_embeddings` (`np.ndarray`): 2D matrix of embeddings for all documents (e.g., UMAP or PCA). Shape: `(n_documents, n_features)`.
- `label2docs` (`Dict[int, List[int]]`): Dictionary mapping cluster labels to lists of document indices (i.e., `label → [doc_ids]`).
- `probas` (`np.ndarray`): Array of soft clustering probabilities for each document.
- `min_size` (`int`, default=`17`): Minimum number of samples to select per cluster.
- `partition` (`float`, default=`0.02`): Proportion of the cluster size to use as sample size. Controls how much to increase the number of examples for larger clusters.
- `proba_threshold` (`float`, default=`0.8`): Threshold above which a point is considered high-confidence.
- `method_selection` (`str`, default=`"farthest_high_proba"`): Sampling method to use.  
  Currently, only the `"farthest_high_proba"` method is implemented.  
  This parameter has been designed to make the function easily **extensible**—future contributors can add and plug in other sampling strategies here (e.g., uncertainty-based sampling, center-weighted sampling, etc.).
- `proba_min_low` (`float`, default=`0.3`): Minimum probability for a point to be considered low-confidence but non-noisy.

**Returns**
- `Dict[int, List[int]]`: A dictionary mapping each cluster label to the list of selected document indices for that cluster.

> **Note:** This function handles the entire clustering result. For each cluster, it adapts the number of examples based on its size and then selects representative samples with a **dispersion + confidence tradeoff strategy**, ensuring diverse and robust annotation candidates for LLMs.

---

## Function: `extract_numeric_key`

**Description**  
Extracts the smallest integer from a label that may contain one or more numeric parts separated by underscores.

Useful when sorting cluster labels such as `"2_21"` or `"1"` based on their primary numeric key.

**Arguments**
- `label` (`str`): A string label, possibly composed of multiple numeric parts (e.g., `"2_21"`).

**Returns**
- `int`: The smallest integer found in the label. If no integer can be extracted, returns `float('inf')` to push the label to the end in a sort.

> **Note:** This function is typically used to sort cluster labels in a meaningful order. Ex.:["-1", "1",  "3","2_21"] -> ['-1', '1', '2_21', '8']
---

##  Function: `call_openai_api`

**Description**  
This is the **core function** responsible for automatically generating topic labels for clusters using the OpenAI API.  

The function:
- Selects representative examples from each cluster (via `select_examples`),
- Constructs a prompt with those examples,
- Sends the prompt to a Large Language Model (LLM),
- Collects the response and stores it as the topic label for that cluster.

It also supports fallback behavior: if no API key is provided, generic cluster labels (`"cluster 0"`, `"cluster 1"`, etc.) are returned.

### Step 1 – Sampling Parameters for `select_examples`

Before calling the LLM, this function uses the `select_examples(...)` utility to pick informative, diverse texts from each cluster. These parameters are set internally:

- `min_size = 20` → ensures small clusters are still represented with sufficient examples.
- `partition = 0.02` → in large clusters, selects 2% of documents.
- `proba_threshold = 0.8` → defines which texts are considered "core" to the cluster.
- `proba_min_low = 0.3` → includes a few low-probability, non-noisy examples for diversity.
- `method_selection = "farthest_high_proba"` → uses a mutual-distance strategy to select diverse samples among confident texts.

> This step ensures the prompt sent to the LLM is compact, relevant, and semantically diverse.

### Step 2 – Dynamic Prompt Construction via `instruction_template`

To avoid assigning **duplicate labels** (e.g., two clusters named `"Intelligence Debate"`), a dynamic prompt template is used:
instruction = instruction_template.format(existing_labels=existing_labels) 

instruction_template is a string that may contain {existing_labels} at each iteration, this placeholder is replaced with the list of already-used topic labels.

### Step 3 – OpenAI Chat Message Construction

For each cluster, the function builds a two-turn message, system and user: The system message defines the summarization task and can include banned topics, the user message contains N selected examples from the cluster, formatted as :
Example 1:
<cleaned and truncated text>

Example 2:
<cleaned and truncated text>
...
The number of characters per example is capped at summary_chunk_size = 800 to fit the OpenAI token limit.

### Step 4 –LLM Interaction:

For each cluster label ≠ "-1", the function: sends the constructed message to the OpenAI model (gpt-4o-mini by default), then it parses the model’s response and stores it as the cluster summary , and finally adds the new label to the list of existing labels to avoid reuse in the next iterations.

In case of an exception, it stores: "Error: <exception message>"

**Arguments**
- `df` (`pd.DataFrame`): DataFrame containing the text data and clustering information. Must include:
  - `'text'`: raw textual content of documents,
  - `'x'`, `'y'`: 2D projection coordinates (e.g., UMAP),
  - `'col_topic'`: cluster label column,
  - `'col_proba'`: membership probability column.

- `col_topic` (`str`): Name of the column containing cluster labels. Default: `"topic"`.

- `col_proba` (`str`): Name of the column containing membership probabilities. Default: `"proba"`.

- `api_key` (`Optional[str]`): OpenAI API key. If not provided, the function returns generic fallback labels without calling the LLM.

- `instruction` (`Optional[str]`): Template instruction injected into the system prompt. May contain `{existing_labels}` for dynamic adaptation.

- `existing_labels` (`Optional[List[str]]`): List of topic labels already generated. Used to avoid duplication in LLM outputs. Default: empty list.

- `model` (`str`): OpenAI model name to use. Default: `"gpt-4o-mini"`. Should not be changed in production.

- `summary_chunk_size` (`int`): Maximum number of characters per text example passed in the prompt. Default: `800`.

- `return_inputs_outputs` (`bool`): If `True`, returns detailed prompts and raw LLM outputs along with summaries. Default: `False`.

**Returns**
- `Dict[str, str]`: If `return_inputs_outputs` is `False`, returns a dictionary mapping each cluster label to its LLM-generated topic summary. This is the default mode for direct usage.

- `Dict[str, Union[Dict[str, str], List, List]]`: If `return_inputs_outputs` is `True`, returns:
  - `'summaries'`: dictionary of topic labels per cluster,
  - `'messages'`: list of prompts sent to the LLM,
  - `'outputs'`: list of raw responses received from the model.

> **Note:** Enabling `return_inputs_outputs = True` is especially useful for:
> - Calculating the cost associated with each API call (based on input/output tokens),
> - Logging, debugging, or quality control of the generation pipeline,
> - Designing an alerting or retry mechanism in case of API failures or anomalous outputs.

> ⚠️ These post-processing layers (cost tracking, alerting, retry logic) are **not yet implemented**, but the structure is intentionally designed to support them. Future contributors are encouraged to extend this logic based on project needs.

---
## Function: `cluster_centers`

We now finish the first group of helper functions dedicated to **LLM-based cluster annotation**.  
We move on to the second part of the pipeline:  
**Hierarchical merging of semantically similar clusters** — to correct cases where a single topic is **scattered across multiple clusters**.

This next set of functions focuses on:
- Measuring similarity between cluster centroids,
- Iteratively merging the most semantically similar clusters,
- Structuring the clustering result into higher-level topics.


**Description**  
This function computes the centroid (average embedding vector) of each cluster, excluding outliers labeled `"-1"`.  
It returns:
- a matrix of cluster centers, and
- a sorted list of corresponding cluster labels.

The `i`-th row in the returned matrix corresponds to the `i`-th label in the returned list.

> ⚠️ Cluster labels must be provided as strings (e.g., `"0"`, `"1"`, `"2_1"`), not integers, to ensure correct behavior with `extract_numeric_key`.

**Arguments**
- `embeddings` (`np.ndarray`): Embedding matrix of all documents. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`np.ndarray`): Array of cluster labels, one per document. Shape: `(n_samples,)`.  
  The label `"-1"` (outliers) is automatically excluded.  
  Labels must be strings for proper numeric sorting (e.g., `"2_1"` before `"10"`).

**Returns**
- `np.ndarray`: Cluster center matrix. Each row is the average embedding of one cluster. Shape: `(n_clusters, embedding_dim)`.

- `List[str]`: Sorted list of cluster labels (excluding `"-1"`). The `i`-th vector in the matrix corresponds to the `i`-th label in this list.

> **Note:** The labels are sorted using `extract_numeric_key`, which allows intuitive ordering even with compound labels like `"2_1"`.

---

## Function: `similarity_matrix`

**Description**  
This function computes the **cosine similarity matrix** between cluster centers, using the output of `cluster_centers(...)`.  
Each cluster is represented by the **mean** of its document texts embeddings .  

> The **mean** is chosen over other metrics (like the median or mode) because it:
> - preserves the vector space structure,
> - aligns with most distance-based methods (e.g., cosine, Euclidean),
> - is differentiable and more stable under continuous updates.

The function can optionally label the matrix rows/columns using textual summaries (e.g., `"Cluster 3: Artificial Intelligence"`), if provided via `cluster_summaries`.

The resulting matrix is a square `DataFrame` of shape `(n_clusters, n_clusters)` where each entry `[i, j]` represents the cosine similarity between clusters `i` and `j`.

> ⚠️ Cluster `-1` (outliers) is ignored during the centroid computation.  
> Labels must be strings for correct ordering and summary resolution.


**Arguments**
- `embeddings` (`np.ndarray`): Matrix of document embeddings. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`): Array or list of cluster labels (one per document). Labels must be strings. Shape: `(n_samples,)`.

- `cluster_summaries` (`Dict[str, str]`, optional):  
  Optional dictionary mapping cluster labels to topic summaries. If provided, these summaries are used to annotate the matrix rows and columns (e.g., `"2: Healthcare"`).  
  If not provided, default labels like `"2"` or `"2_1"` are used.

**Returns**
- `pd.DataFrame`: Square cosine similarity matrix between cluster centers.  
  Both rows and columns are labeled either by:
  - the raw cluster labels, or  
  - `"{label}: {summary}"` if `cluster_summaries` is provided.

> **Note:** This matrix is used as input to hierarchical merging logic and for visualization (e.g., heatmaps or dendrograms).

---

## Function: `max_similarity`

**Description**  
This function performs one step of **hierarchical merging** between the two most similar clusters, based on cosine similarity between their centroids.  
The returned labels reflect the merge, and optionally the function tells you which clusters were fused.

### Key Idea: Label Merging as Traceable Strings

Instead of maintaining a tree-like structure to track merging paths (which would require expensive memory and traversal operations to later assign the same label to all descendant nodes),  
we encode **the entire fusion history directly into the new label itself**, using a string of the form:

```text
"1", "10"        → "1_10"  
"1_10", "5_3"    → "1_10_5_3"
```

At the end, a simple text-based flattening (e.g., `label.split("_")`) is enough to retrieve the full list of initial cluster labels involved in the merge.

This strategy keeps the process **stateless**, **transparent**, and **easy to use in downstream annotation steps** (e.g., applying the same LLM-generated label to all fused clusters).


**Arguments**
- `embeddings` (`np.ndarray`): Embedding matrix of all documents. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`): Current cluster labels, one per document. Must be strings. Shape: `(n_samples,)`.  
  The function is compatible with compound labels like `"1_3"` or `"3_10_4"`.

- `return_argmax` (`bool`, default = `False`):  
  If `True`, the function also returns the names of the two clusters that were merged.


**Returns**
- `float`: Cosine similarity between the two most similar clusters (i.e., the merge score).

- `np.ndarray`: Updated `cluster_labels` where the two merged clusters now share a new label of the form `"labelA_labelB"`.

- *(optional)* `Tuple[str, str]`: The names of the two merged clusters, returned only if `return_argmax = True`.

> **Note:** The similarity matrix is recomputed at each call, and the most similar pair is selected using the upper triangular part of the matrix (excluding the diagonal).

---

## Function: `simplify_cluster_labels_reduced`

**Description**  
This function simplifies compound cluster labels generated during hierarchical merging (e.g., `"1_3_10"`) into a **single representative label**, typically the smallest numeric ID in the group (e.g., `"1"`).

It serves two key purposes:
- **Visualization**: useful for clearer, shorter labels in plots or tables.
- **Post-processing**: helps assign a consistent ID (e.g., for evaluation, coloring, or reporting).

> This function is especially useful after using `max_similarity(...)` or `hirarchical_clustering(...)`, which accumulate label histories like `"1_5_10"`.

**Example**
```python
Input :  ["1_10", "3_5", "7_9_11_20"]
Output:  {"1_10": "1", "5_3": "3", "9_7_11_20": "7"}
```

**Arguments**
- `cluster_labels_reduced` (`List[str]`):  
  List of string labels after hierarchical merging.  
  Each label may be a compound string like `"2_7_19"`.

**Returns**
- `Dict[str, str]`:  
  A dictionary mapping each full (compound) label to its simplified version, based on the **minimum numeric part**.

---


##  Function: `get_number_clusters_given_seuil`

**Description**  
This function performs **iterative merging** of clusters based on a similarity threshold, and returns the number of clusters remaining once all highly similar clusters (above the threshold) have been fused.

> It helps quantify "how many clusters would remain" if we were to merge all clusters with cosine similarity ≥ `1 - seuil`.

The process:
- Starts from the current set of clusters.
- Repeatedly calls `max_similarity(...)` to merge the most similar pair.
- Stops once the highest similarity between any two clusters falls below the threshold.

This is used as input to the **cluster reduction curve** (number of clusters vs. similarity threshold), which is later plotted and analyzed.


**Arguments**
- `embeddings` (`np.ndarray`): Embedding matrix of all documents. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`): Initial cluster labels, one per document. Labels must be strings.

- `seuil` (`float`): Distance threshold (cosine distance, not similarity).  
  The function converts it internally to `similarity = 1 - seuil`.


**Returns**
- `int`: Number of remaining clusters after merging (i.e., `k_final`).

- `List[str]`: The updated cluster labels after applying all merges required by the threshold.

> **Note:** This function is used to generate the left plot in `plot_elbow_and_clusters_vs_threshold`.
---

##  Function: `compute_inertia`

**Description**  
This function computes the **cosine inertia** of the clustering: the sum of cosine distances between each document and its cluster center.

> This gives a quantitative evaluation of how compact the clusters are.  
> Lower inertia = more compact clusters = better merging quality (in general).

The function:
- Computes the center of each cluster with `cluster_centers(...)`.
- For each point, calculates `1 - cosine similarity` to its cluster centroid.
- Sums the total distance across all clusters.

Used to produce the **elbow plot of inertia vs. number of clusters**, guiding the user in choosing a **good merge level**.


**Arguments**
- `embeddings` (`np.ndarray`): Matrix of all document embeddings. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`np.ndarray`): Cluster labels, one per document. Labels must be strings.


**Returns**
- `float`: Total cosine-based inertia across all clusters.

> **Note:** This metric is used to generate the right plot in `plot_elbow_and_clusters_vs_threshold`, enabling trade-off decisions between semantic granularity and compactness.


---


## Function: `hirarchical_clustering`

**Description**  
This is the **core function** of the second phase of the pipeline: the **hierarchical merging of clusters**.  
It addresses a common issue in topic modeling pipelines: when two or more clusters generated by a model (e.g., HDBSCAN) actually refer to the **same underlying topic**.  

This function applies a classical hierarchical clustering approach — **not on the original data points**, but on the **output clusters** from a previous topic modeling step.  
Each cluster is treated as a unit and represented by its centroid in the embedding space.

###  Why not use hierarchical clustering directly from the beginning?

One might ask: _Why not apply hierarchical clustering directly on the documents, rather than chaining models?_  
That's a valid question. But it overlooks the strengths of **density-based models like HDBSCAN**, which:
- Detect **variable-density clusters**,
- Discard **noise and outliers** more effectively,
- Are often more robust in complex, high-similarity contexts (e.g., news articles on the same theme).

In this work, our documents are **not generic**: they often revolve around a known subject (e.g., "AI", "cybersecurity", or the reputation of a public figure).  
This makes **initial clustering via cosine similarity** too aggressive, potentially grouping everything into one cluster.

> By combining HDBSCAN with a hierarchical merge phase, we exploit **density-aware segmentation**, then refine the result by merging clusters that are **semantically close**.

### Behavior and Threshold Logic

You can either:
- Provide a **target number of clusters** (e.g., 50), and/or  
- Use a **similarity threshold** (e.g., `seuil = 0.1` → similarity ≥ 90%) to trigger **adaptive recommendations**.

Assume you start with 150 clusters, and you ask the algorithm to reduce them to 50. However, you set seuil = 0.1, meaning you only want to merge clusters that are more than 90% similar. If, at some point, the most similar clusters fall below that threshold, the algorithm will halt the similarity-based suggestions and output:

> _"It is suggested to reduce the number of clusters to 90 for better interpretability of the data. If so, you will only merge clusters with more than 90% similarity."_

This warning is **non-blocking**: the algorithm will still reduce to the target `number_of_clusters`, but you’ll be informed that **interpretability may degrade** past that point, if None, no recommandation is returned.


### When to use a low or high threshold?

- **Low thresholds** (e.g., `seuil = 0.1`) are preferred when:
  - The documents are thematically focused (like in our case we have collected data about IA, cloud, etc.),
  - Clusters are already well-formed and need fine-grained merging.

- **High thresholds** (e.g., `seuil = 0.3`) may apply when:
  - The corpus is diverse or multi-thematic,
  - You're working with time slices, or open-topic modeling.

This reinforces a key principle in data science:

> **Business knowledge matters** Domain context should guide modeling choices — including clustering granularity.
This function integrates all of this into a simple, controllable interface, with optional outputs for tracking the full merge process.

### Arguments

- `embeddings` (`np.ndarray`):  
  The full document embedding matrix. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`):  
  Initial cluster labels assigned to each document. Must be strings (especially if previous merges have occurred).  
  Shape: `(n_samples,)`.

- `number_of_clusters` (`int`, default = 50):  
  The target number of clusters to obtain after merging. The process stops once this number is reached.

- `return_grouped_labels_per_iteration` (`bool`, default = False):  
  If `True`, the function also returns a history of merge operations at each iteration.  
  Useful for tracking cluster evolution or debugging.

- `seuil` (`float`, default = None):  
  Cosine distance threshold to trigger **recommendations** (not strict constraints), if None, no recommandation is returned.  
  For example, `seuil = 0.2` means you will be notified when merging clusters that are **less than 80% similar**.

### Returns

- `cluster_labels_` (`np.ndarray`):  
  New array of cluster labels after hierarchical merging.  
  Same shape as the input (`(n_samples,)`), but with updated labels reflecting fused groups (e.g., `"3_7_12"`).

- `grouped_labels` (`List[Tuple[Tuple[str, str], int]]`, optional):  
  Only returned if `return_grouped_labels_per_iteration=True`.  
  A list of tuples describing each merge step:  
  Each element is `((label1, label2), n_remaining_clusters)`  
  — showing which clusters were merged and how many clusters remained at that stage.


---

# Visualization functions
---

## Function: `visualize_clusters`
---

**Description**

This function generates an interactive 2D scatter plot (typically based on UMAP reduction), where:

- **Each point** corresponds to a document (tweet, sentence, etc.) plotted at coordinates `(x, y)`.
- **Point color** encodes the assigned cluster label (`topic`).
- **Cluster centers** are annotated with their semantic summary (from `cluster_summaries`).

In the **hover tooltip**, the following fields are displayed for each point:
- A **truncated version of the raw text** (`text`, up to `max_text_showed_length` characters),
- The **LLM-generated annotation** (`annotation`),
- The **membership probability** (`proba`, often from HDBSCAN).

This visualization supports a semantic validation of clusters via their central summaries, also a visual inspection of topic separation in reduced space (e.g., UMAP), and a diagnosis of uncertainty via cluster probabilities.

**Arguments**

- `df` (`pd.DataFrame`): Input dataframe that must contain the following columns:
  - `'text'`: original document text
  - `'annotation'`: LLM annotation or human summary (can be `None`)
  - `'x', 'y'`: 2D coordinates from dimensionality reduction (e.g., UMAP)
  - `'topic'`: cluster label per document
  - `'proba'`: probability of membership to the assigned cluster

- `reduced_cluster_centers_` (`Dict[int, Tuple[float, float]]`):  
  Dictionary mapping each cluster label to its center in the 2D space.

- `cluster_summaries` (`Dict[int, str]`):  
  Dictionary mapping each cluster label to its semantic summary (e.g., via GPT).

- `max_text_showed_length` (`int`, default = 80):  
  Maximum character length to display in the hover tooltip for `text`.

- `title` (`str`, default = `"Clustering UMAP + HDBSCAN — Topic Label per Cluster"`):  
  Title of the visualization.

- `width` (`int`, default = 1300): Width of the figure in pixels.  
- `height` (`int`, default = 750): Height of the figure in pixels.  
- `size_points` (`int`, default = 7): Size of individual document points.  
- `opacity_points` (`float`, default = 0.8): Opacity of document points.  
- `size_text` (`int`, default = 9): Font size of topic labels.  
- `opacity_text` (`float`, default = 0.9): Opacity of topic labels.  
- `col_topic` (`str`, default = `"topic"`): Name of the column containing cluster assignments.  
- `col_proba` (`str`, default = `"proba"`): Name of the column containing cluster probabilities.  
- `col_annotation` (`str`, default = `None`): Name of the annotation column (LLM output or summaries).


**Returns**  
None.  
Directly displays a Plotly interactive figure with:
- Color-coded clusters
- Hover info for text, annotation, and proba
- Centered annotation labels

**Note**  
This function is display-oriented and returns no data structure. It is designed to be used in a notebook or dashboard as a final visual sanity check.
---

## Function: `plot_heatmap_inter_cluster_cosine_similarity`

**Description**

Displays an interactive cosine similarity heatmap between **cluster centers** in 2D.  
This visualization helps identify which clusters are semantically close, guiding decisions about merging, interpreting thematic overlaps, or assessing model quality.

Each cell in the matrix represents the **cosine similarity** between the centroids of two clusters. The matrix is symmetric, with diagonal values of 1 (perfect similarity with self).


**Arguments**

- `embeddings` (`np.ndarray`):  
  Embedding matrix of all documents.  
  Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`):  
  Array of cluster labels for each document.  
  Used to compute the average vector (centroid) per cluster.

- `cluster_summaries` (`Dict[int, str]`, optional):  
  Optional dictionary `{label: summary}` to rename rows/columns in the heatmap.  
  If empty, cluster IDs are used instead.

- `title` (`str`, optional):  
  Title of the heatmap figure. Default: `"Inter-Cluster Cosine Similarity Heatmap (with Labels)"`.

- `width` (`int`, optional):  
  Width of the figure in pixels. Default: `1200`.

- `height` (`int`, optional):  
  Height of the figure in pixels. Default: `1000`.

**Returns**

- `None`:  
  Displays a **Plotly heatmap** of cosine similarities between cluster centroids.  
  Each axis shows cluster IDs or labels, and color intensity indicates similarity (from 0 to 1).

**Notes**

- Internally uses the `similarity_matrix` function to compute pairwise similarities between cluster centroids.
- The plot is interactive: you can hover over each cell to see the exact similarity value.
- Useful to:
  - Inspect thematic overlaps
  - Suggest potential merges
  - Evaluate model fragmentation

  ---

## Function: `plot_dendrogram_with_labels`

### Description

This function displays an **interactive dendrogram** based on cosine similarity between **cluster centers**.  
Each node represents a **cluster centroid**, and the tree structure reveals how similar clusters are in the embedding space.

### Why is it useful?

This dendrogram is not applied to individual data points, but to **clusters already formed** (e.g., from HDBSCAN).  
It enables:

- **Semantic analysis of cluster proximity**  
  You can directly see which clusters talk about similar topics.
  
- **Guided meta-clustering decisions**  
  For example: Should I merge clusters 4, 7, and 12 into a broader theme?

- **Threshold discovery**  
  By reading distances on the dendrogram axis, one can infer **natural cut points** to reduce or compress cluster granularity.


### What do the axes represent?

- **Y-axis (vertical)**: Cluster labels (with optional summaries if provided).
- **X-axis (horizontal)**: Distance between clusters =  
  $$
  \text{distance}(C_i, C_j) = 1 - \cos(\theta) = 1 - \text{cosine\_similarity}
  $$
  A value close to 0 means **high similarity** (e.g., 0.1 → 90% similar), while a value near 1 indicates very different clusters.


### Example label:

Each cluster is labeled as: 5 Responsible AI in Healthcare

Where `5` is the cluster ID, and the label is its **LLM-generated annotation** (optional, but highly recommended).


**Ideas for improvement**: add **LLM-generated summaries** at each merge node.

### Arguments

- `embeddings` (`np.ndarray`):  
  Embedding matrix of documents. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`):  
  Array of cluster labels assigned to each document. These are used to compute centroids.

- `cluster_summaries` (`Dict[int, str]`, optional):  
  Mapping from cluster IDs to annotated topic labels. Displayed next to each cluster ID in the dendrogram.

- `title` (`str`, optional):  
  Title of the dendrogram. Default: `"Hierarchical Clustering of Cluster Centers (Named Labels)"`.

- `width` (`int`, optional):  
  Plot width in pixels. Default: `1200`.

- `height` (`int`, optional):  
  Height per cluster. The actual figure height is `height × number of clusters`. Default: `35`.

### Returns

- `None`  
  Directly renders a **Plotly dendrogram** showing semantic similarity between clusters.

---

## Function: `plot_elbow_and_clusters_vs_threshold`

### Description

This function generates **two complementary diagnostic plots** to guide the decision on **how many clusters** to retain during hierarchical merging based on cosine similarity between clusters.

It serves as a **quantitative tool** to balance between **semantic granularity** and **intra-cluster cohesion**.

### What does it display?

The function renders **two subplots** side-by-side:

#### 1. **Left Plot** – Number of Clusters vs Cosine Distance Threshold
- **X-axis**: Distance threshold ($\delta = 1 - \text{similarity}$)
- **Y-axis**: Number of clusters **remaining** after merging all clusters whose cosine similarity exceeds $(1 - \delta)$.
- This curve typically **decreases** as the threshold increases:
  - Low threshold → strict merging → many small clusters
  - High threshold → loose merging → few large clusters

> This graph answers: *“How many clusters remain if I merge those more than X% similar?”*

#### 2. **Right Plot** – Inertia vs Number of Clusters
- **X-axis**: Number of clusters (after each merge iteration)
- **Y-axis**: Inertia = total dissimilarity between points and their assigned cluster center.
- Lower inertia implies more **compact clusters**, but too few clusters can lead to **semantic dilution**.
- The curve typically **decreases** with fewer clusters, but may flatten after a point → **elbow point detection**.

> This graph helps find a sweet spot between **semantic precision** and **cluster compactness**.

### Example Insight (from annotation)

A point is annotated to show how to interpret the left graph:

> If the selected threshold is `x = 0.25`, then we merge all clusters that are **more than 75% similar**.  
> The resulting number of clusters is shown as `y = 42`.

This helps translate abstract similarity values into **concrete decisions**.

### Arguments

- `embeddings` (`np.ndarray`):  
  Full embedding matrix of all documents. Shape: `(n_samples, embedding_dim)`.

- `cluster_labels` (`Union[np.ndarray, list]`):  
  Cluster labels assigned to each document (initial labels before merging).

- `seuil` (`Optional[List[float]]`, default = `None`):  
  List of cosine distance thresholds to test. If `None`, a default range is generated.

- `exemple_how_to_read_graph_given_that_seuil` (`int`, default = `4`):  
  Index of the threshold to highlight on the left plot (used to display annotation).

- `width` (`int`, default = `1000`):  
  Total width of the final plot (in pixels).

- `height` (`int`, default = `500`):  
  Total height of the final plot (in pixels).

### Returns

- `None`  
  Displays a Plotly interactive subplot with:
  - Number of clusters vs threshold  
  - Inertia vs number of clusters  

Both are interactive and can guide **merge strategy tuning**.







