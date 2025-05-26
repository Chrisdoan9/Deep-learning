RNNs are a type of neural network designed to handle data that comes in a sequence - like sentences, time series, or audio.
Hidden state is not the final prediction itself. It’s intermediate memory — which you can use to compute the prediction.
Cross-entropy measures how well one probability distribution (your prediction) matches another distribution (the truth). If your prediction matches the true distribution well, cross-entropy is low.

Say your model is predicting the next letter in “hello”:  
	•	True next letter: "e"   
	•	Vocabulary = ["a", "b", "e", "l", "o"]  
	•	Model prediction: [0.05, 0.05, 0.75, 0.1, 0.05]  
→ Cross-entropy loss = −log(0.75) ≈ 0.287  

But if it guessed: [0.6, 0.2, 0.1, 0.05, 0.05]  
→ Cross-entropy loss = −log(0.1) = 2.302
