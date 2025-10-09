# Partitioned annotated final data

This folder contains the partitioned `.parquet` files resulting from the full clustering pipeline.

The full dataset `annotated_final_data.parquet` exceeded GitHub's 100MB file size limit.  
To avoid using Git LFS and keep the repository lightweight and accessible, the dataset was split into 5 smaller parts.

Each file (`annotated_final_data_partX.parquet`) corresponds to a non-overlapping chunk of the full DataFrame.
