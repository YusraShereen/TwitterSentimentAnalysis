# DS5007 - Twitter Sentiment Classifier

This repository contains a Google Colab notebook for training and evaluating a **3‑class Twitter sentiment classifier**. The dataset consists of:

* `clean_text` — preprocessed tweet text
* `category` — sentiment labels: **−1 (Negative), 0 (Neutral), 1 (Positive)**

---

## How to Run on Google Colab

1. **Upload Dataset**
   Upload `Twitter_Data.csv` to your **Google Drive**.

2. **Open Notebook**
   Open the notebook file (`.ipynb`) in **Google Colab**.

3. **Mount Drive & Set Paths** — update inside notebook:

   ```python
   DATA_PATH = '/content/drive/MyDrive/path/to/Twitter_Data.csv'  # Update
   OUTDIR = '/content/drive/MyDrive/path/to/output_directory'     # Update
   ```

4. **Install Dependencies** — run the cell containing:

   ```bash
   !pip install pandas numpy scikit-learn matplotlib seaborn reportlab nbformat tensorflow nltk
   ```

5. **Download NLTK Data** — run:

   ```python
   import nltk
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('averaged_perceptron_tagger_eng')
   ```

6. **Run Full Pipeline** — execute all remaining cells for:

   * Data loading & preprocessing
   * Baseline, BiLSTM, GRU_MODEL_WITH_FEATURES training
   * Evaluation + error analysis

---

## How to Run Locally

```bash
git clone <repo_url>
cd TwitterSentimentAnalysis
pip install pandas numpy scikit-learn matplotlib seaborn reportlab nbformat tensorflow nltk
```

Then download NLTK data:

```python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

Place `Twitter_Data.csv` locally and update **DATA_PATH** and **OUTDIR** in the notebook accordingly. Run via **Jupyter Notebook / VS Code / PyCharm**.

---

## Project Structure

```
	TwitterSentimentAnalysis.ipynb  
	Twitter_Data.csv      
	twitter_sentiment_results/
```

---

##  Models Implemented

| Model              | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| **Baseline**       | TF‑IDF + Multinomial Naive Bayes + handcrafted features       |
| **BiLSTM**         | Deep model using Bidirectional LSTM for context understanding |
| **GRU + Features** | Combines GRU text encoder + handcrafted numeric features      |

---

##  Evaluation Metrics

* **Accuracy**
* **Macro F1‑Score** 
* **Precision & Recall (Macro)**
* **Error Rate**
* **Confusion Matrices**
* **ROC Curves**

 Final comparison table highlights **best performing model (Macro‑F1 on Test)**.

---

##  Error Analysis

Identifies & saves misclassified tweets — including cases of:

* Sarcasm / irony
* Polysemy / context ambiguity
* Negation handling issues

---

## Generated Output Files (`OUTDIR`)

* `baseline_pipeline.pkl`
* `tokenizer.pkl` / `tokenizer_gru.pkl`
* `bilstm_model.h5` / `gru_model_with_features.h5`
* Confusion matrices: `baseline_confusion.png`, `bilstm_confusion.png`, etc.
* Misclassification CSVs
---
