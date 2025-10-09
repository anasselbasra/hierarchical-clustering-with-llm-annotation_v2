# Partitioned Clustering Results

This folder contains the partitioned `.parquet` files resulting from the full clustering pipeline.

The full dataset `clustering_results.parquet` exceeded GitHub's 100MB file size limit.  
To avoid using Git LFS and keep the repository lightweight and accessible, the dataset was split into 5 smaller parts.

Each file (`clustering_results_partX.parquet`) corresponds to a non-overlapping chunk of the full DataFrame.
