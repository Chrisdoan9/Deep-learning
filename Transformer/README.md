This model uses a Transformer encoder to learn cell representations from gene expression, followed by a classification head. A decoder is unnecessary because the task is discriminative, not generative.

•	Trained a small Transformer encoder on PBMC scRNA-seq to predict cell types using top-K expressed genes as tokens.  
•	Evaluated accuracy and visualized learned cell embeddings with UMAP.
