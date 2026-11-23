# Stanford Replication: Methods and Results

This documents the replication of the politeness classifiers from Danescu-Niculescu-Mizil et al. (2013), Table 4, on the Stanford Politeness Corpus. We are aiming to confirm that our setup reproduces the original results before applying the models to Intel Polite Guard.

**Summary.** The replication is close to the paper for Wikipedia in-domain (−2 to −4 points) and for Stack Exchange → Wikipedia transfer (−1 to −3 points). It is further off for Wikipedia → Stack Exchange (−5 to −8 points) and more off for Stack Exchange in-domain (−13 to −16 points). Inspecting the Stack Exchange model suggests it relies heavily on topic words rather than politeness cues.

---

## 1. What we did

### Script
`stanford_replicate.py` run with:

```
python stanford_replicate.py          # 10-fold cross-validation (what was used here)
python stanford_replicate.py --loo    # leave-one-out, as in the original paper
```

### Data
Wikipedia and Stack Exchange portions of the Stanford Politeness Corpus, downloaded through ConvoKit (`wikipedia-politeness-corpus`, `stack-exchange-politeness-corpus`).

Following the paper, only the top quartile (polite) and bottom quartile (impolite) of politeness scores were kept. ConvoKit's `Binary` label reproduced the paper's class sizes exactly:

| Corpus | Polite | Impolite | Paper |
|---|---|---|---|
| Wikipedia | 1,089 | 1,089 | 1,089 per class |
| Stack Exchange | 1,651 | 1,651 | 1,651 per class |

### Features and models
- **BoW:** linear SVM on unigram counts (words appearing in fewer than 10 requests removed).
- **Ling:** linear SVM on unigram counts plus 21 politeness-strategy features from ConvoKit's `PolitenessStrategies` (gratitude, apologizing, hedges, position of *please*, direct questions, etc.).
- SVM: scikit-learn `LinearSVC`, `C=1.0`, random seed 42.

### Evaluation
- **In-domain:** 10-fold stratified cross-validation within each corpus. The vectorizer is refit inside each fold, so no test vocabulary leaks into training.
- **Cross-domain:** train on all of one corpus, test on all of the other.
- **Metrics:** accuracy (the paper's metric), plus precision, recall, and F1 for the polite class, and ROC-AUC.

---

## 2. Results

Accuracy (%), compared with Table 4 of the original paper:

| Model | Train | Test | Evaluation | Ours | Paper | Difference | F1 (polite) | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| BoW | Wiki | Wiki | 10-fold | 77.64 | 79.84 | −2.20 | 0.77 | 0.85 |
| BoW | Wiki | SE | cross-domain | 59.72 | 64.23 | −4.51 | 0.56 | 0.64 |
| BoW | SE | SE | 10-fold | 61.45 | 74.47 | −13.02 | 0.61 | 0.65 |
| BoW | SE | Wiki | cross-domain | 71.07 | 72.17 | −1.10 | 0.71 | 0.77 |
| Ling | Wiki | Wiki | 10-fold | 79.61 | 83.79 | −4.18 | 0.79 | 0.87 |
| Ling | Wiki | SE | cross-domain | 59.96 | 67.53 | −7.57 | 0.55 | 0.64 |
| Ling | SE | SE | 10-fold | 62.14 | 78.19 | −16.05 | 0.62 | 0.66 |
| Ling | SE | Wiki | cross-domain | 72.64 | 75.43 | −2.79 | 0.73 | 0.79 |


### What are similar
- **Ling beats BoW in every setting**, as in the paper (by 0.2 to 2.0 points), confirming that the politeness features add information beyond words alone.
- **Wikipedia in-domain** and **Stack Exchange → Wikipedia** are within about 1–4 points of the paper.
- **Wikipedia → Stack Exchange shows the cross-domain drop** the paper reported (about 18–20 points below Wikipedia in-domain in our run, versus about 16 in the paper).

### What are different
**Stack Exchange in-domain is 13–16 points below the paper** (about 62% vs. 75–78%). This is surprising because a model *trained* on Stack Exchange transfers well to Wikipedia (71–73%), so the training data seems usable; it is specifically *predicting* Stack Exchange requests that fails.

---

## 3. Investigating the Stack Exchange gap

The 20 largest weights in each direction for the BoW model trained on Stack Exchange (`results/top_features_BoW_se.csv`):

| Leans impolite | Leans polite |
|---|---|
| comments, wikipedia, install, definition, hash, learn, homework, whether, wait, shouldn, dont, nothing, purpose, js, special, software, **why**, outside, apps, multiple | thinking, although, approach, **thanks**, heard, classes, mentioned, css, needed, **welcome**, specifically, curiosity, others, view, send, explanation, share, back, their, allow |

**Observations:**
- Many top features are **topic words** unrelated to politeness: *install, hash, js, software, apps* (impolite) and *css, classes, approach* (polite). Stack Exchange spans many subjects, and the model appears to pick up which topics happened to co-occur with polite or impolite requests.
- Some genuine politeness cues appear: *thanks, welcome* (polite) and *why, dont, shouldn* (impolite). *Why* matches the paper's "direct question" strategy.
- The weights are large (up to 1.6), suggesting the model **overfits** the training folds. Topic patterns that fit one subset of requests do not generalize to held-out requests.

---

## 4. Differences from the original setup

To report in the paper's Methods section:
1. **10-fold cross-validation** instead of leave-one-out for in-domain evaluation.
2. **ConvoKit's reimplementation** of the politeness features and its parser, instead of the original Stanford dependency parser.
3. **SVM hyperparameters:** `C=1.0` was chosen as a default; the paper does not report its settings.
4. **Unigram cutoff:** words appearing in fewer than 10 *requests* (document frequency) were removed; the paper removed words occurring fewer than 10 *times*.

---

## 5. Output files (`results/`)

| File | Contents |
|---|---|
| `replication_results.csv` | All 8 results with the paper's numbers and differences |
| `model_{BoW,Ling}_{wiki,se}.joblib` | Fitted models (trained on the full corpus) for the Intel transfer experiments |
| `strategy_columns.txt` | Exact order of the 21 politeness-strategy features |
| `top_features_{BoW,Ling}_{wiki,se}.csv` | 20 strongest polite and impolite features per model |

---