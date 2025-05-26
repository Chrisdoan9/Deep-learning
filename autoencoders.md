Both PCA and Autoencoders try to compress high-dimensional data (like 20,000 genes) into a lower-dimensional space (like 10 or 2 dimensions), while keeping important patterns. 
PCA learns linear relationship and autoencoders can learn nonlinear relationship.
Input = [GeneA, GeneB, GeneC, GeneD] = [0.8, 1.2, 0.9, 1.1]  
Original: [0.8, 1.2, 0.9, 1.1]  
↓
Compressed: [0.3, -1.0]  
↓
Rebuilt: [0.79, 1.18, 0.91, 1.12]  ← Close to original

The key idea: Patterns

In real-world data, most values aren’t random. They have structure or patterns.

Imagine:  
	•	GeneA and GeneB are always high together  
	•	GeneC is the average of GeneA and GeneB  
	•	GeneD is usually the opposite of GeneC

So even though there are 4 numbers, they really only vary along 2 key directions.
