# Hierarchical Clustering with LLM Annotation – Overview

This repo (v2, as it improves upon the earlier version) builds a **scalable topic discovery pipeline** — turning thousands of raw texts into structured, interpretable themes.  
Developed on 140K LinkedIn posts about AI (June 2025), it generalises to any domain (Cybersecurity, Politics...).

---

##  Pipeline at a Glance

| Step | Core Idea | Tooling |
|------|------------|----------|
| 1- Preprocessing | Clean, normalize, remove noise (short or hashtag-heavy texts) | `clean_text`, regex |
| 2- Embedding | Encode semantics into dense vectors | `all-MiniLM-L6-v2`, SentenceTransformers |
| 3- Smart Sampling | Stratified KMeans sampling for representativity | `MiniBatchKMeans`, √n/2 rule |
| 4- Dimensionality Reduction | Project embeddings for structure | `UMAP` (cosine, 2D) |
| 5- Density Clustering | Find natural groups, tune parameters | `HDBSCAN` + DBCV / silhouette grid |
| 6- Meta-Clustering | Merge close clusters via cosine similarity | Hierarchical / Agglomerative |
| 7- LLM Annotation | Auto-label each theme in 1–3 words | OpenAI API (prompt templates per domain) |
| 8- Visualization | Explore clusters interactively | Plotly, heatmaps, dendrograms |

---

## Strategic Takeaways

- **UMAP + HDBSCAN** yields stable, shape-aware clusters with interpretable density.  
- **Meta-clustering** fuses micro-topics into macro-themes.  
- **LLM-based labeling** automates the naming step while keeping semantic diversity via farthest-point sampling.  
- **Thoughts/** notebook series explains:  
  - when to normalize embeddings,  
  - why duplicates break UMAP,  
  - how cosine similarity aligns with semantic distance.

---

## Quick Start

```bash
# install
pip install -r requirements.txt

# run main notebooks
1_semantic_stratification_&_topic_inference_.ipynb
2_meta_cluster_&_annotation_with_llm.ipynb
