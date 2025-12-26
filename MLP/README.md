## PBMC Cell Type Classification with Deep Learning

Overview

This project uses a multilayer perceptron (MLP) implemented in PyTorch to predict immune cell types from single-cell RNA-seq data. The goal is to build a clean deep-learning baseline and understand how neural networks perform on high-dimensional gene expression data.

Data  
	•	PBMC 3k dataset (loaded via Scanpy)  
	•	Input features: 2,000 highly variable genes  
	•	Labels: Louvain clusters used as proxy cell-type annotations

Model  
	•	Fully connected neural network (MLP)  
	•	ReLU activations with dropout  
	•	Cross-entropy loss, Adam optimizer

Results  
	•	Test accuracy: ~94%  
	•	Strong performance on distinct cell types   
	•	Expected confusion among transcriptionally similar populations (e.g. CD8 T vs NK)

Purpose

This project serves as a learning exercise and baseline implementation for applying deep learning to scRNA-seq data, bridging classical ML and neural network approaches.

Tools

Python, PyTorch, Scanpy, scikit-learn
