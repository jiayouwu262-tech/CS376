Replication and Cross-Domain Evaluation of Computational Politeness Classifiers
COMPSCI 376 · Natural Language Processing

Hellen Han, Jiayou Wu, Lisa Zhao, and Luoshen Wang
Overview
Can a model trained to recognize politeness in Wikipedia or Stack Exchange requests generalize to customer-service interactions? This project revisits the classifiers introduced by Danescu-Niculescu-Mizil et al. (2013) and compares two approaches to binary politeness classification under domain shift.
Datasets
Dataset	Role	Label preparation
Stanford Politeness Corpus	Training and replication on human-annotated requests from Wikipedia and Stack Exchange	Use the highest and lowest quartiles of politeness scores as the polite and impolite classes, respectively.
Intel Polite Guard	External evaluation on customer-service text	Keep examples labeled polite or impolite, map them to 1 and 0, respectively, and exclude the neutral and somewhat polite categories.


Models
- Bag-of-words SVM uses unigram features as the baseline.
- Linguistically informed SVM combines unigrams with signals such as gratitude, apologies, hedging, indirect requests, and syntactic patterns.
Experimental design
Training domain	Evaluation domain	Purpose
Wikipedia	Wikipedia	In-domain replication
Stack Exchange	Stack Exchange	In-domain replication
Wikipedia	Stack Exchange	Cross-domain replication
Stack Exchange	Wikipedia	Cross-domain replication
Wikipedia	Intel Polite Guard	External transfer
Stack Exchange	Intel Polite Guard	External transfer


The external evaluation applies each source-trained model directly to Intel Polite Guard without retraining or fine-tuning. Comparing the two SVMs under the same conditions tests whether linguistic features help when the domain changes.
Evaluation
Accuracy is the primary metric for comparison with the original study. The evaluation also includes precision, recall, and F1 for the polite class, plus ROC-AUC when classifier scores are available.

The Stanford and Intel datasets use different annotation schemes. Their binary labels are aligned for evaluation, so differences in performance can reflect both changes in the text domain and changes in how politeness is labeled.
