CWS-PyTorch: BiLSTM/IDCNN + CRF

A high-performance Chinese Word Segmentation (CWS) system built with PyTorch. It implements and compares two deep learning architectures for sequence labeling.


🌟 Highlights
Manual CRF Layer: The Conditional Random Field (CRF) is implemented from scratch with tensor vectorization, achieving a 10x training speedup compared to standard loops.
Dual Encoders: Supports both BiLSTM (for context accuracy) and IDCNN (for parallel efficiency).
PKU Dataset: Trained and evaluated on the SIGHAN 2005 PKU corpus.


🛠 Technical Route
Preprocessing: Converts text to BMES (Begin, Middle, End, Single) labels.
Embedding: Maps characters to 128-dimensional dense vectors.
Encoding:
BiLSTM: Captures long-range dependencies.
IDCNN: Uses dilated convolutions for fast, parallel feature extraction.
Decoding: A vectorized CRF layer with Viterbi algorithm to find the optimal tag sequence.


📊 Performance (F1-Score)
Model	Precision	Recall	F1-Score
BiLSTM-CRF	0.9561	0.9551	0.9556
IDCNN-CRF	0.9218	0.9291	0.9254
