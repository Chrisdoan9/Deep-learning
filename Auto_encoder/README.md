Project Summary

This project explores representation learning for single-cell RNA-seq data using a denoising autoencoder. Using PBMC data, I trained a neural network to compress gene expression profiles into a low-dimensional latent space and evaluated whether this latent representation preserves biologically meaningful cell-type information.

Specifically, I compared cell-type classification performance using autoencoder-derived latent embeddings versus raw highly variable gene expression, and visualized the learned latent structure using UMAP.

⸻

Key Takeaways
	•	A denoising autoencoder can learn compact representations of scRNA-seq data that retain major immune cell-type structure.
	•	Cell-type classification using latent embeddings achieves performance comparable to models trained on raw gene expression.
	•	Latent space visualization with UMAP shows clear separation of major PBMC cell populations.

⸻

Why this project

This project was completed as a learning exercise to better understand how deep learning models can be applied to high-dimensional biological data, and how learned representations can be evaluated for downstream biological tasks.
