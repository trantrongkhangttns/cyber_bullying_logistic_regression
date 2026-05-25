# 🛡️ Cyber Bullying Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![sklearn](https://img.shields.io/badge/scikit--learn-1.0-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Overview
A text classification system that automatically detects cyberbullying in online conversations using Natural Language Processing (NLP) and Machine Learning. The project was trained and evaluated on the **Formspring dataset** — a real-world Q&A platform known for anonymous harassment.

---

## 📊 Dataset
| Property | Value |
|----------|-------|
| Source | Formspring.csv |
| Total samples | 13,147 |
| Features | `text` (Q&A conversation) |
| Labels | `No` (normal) / `Yes` (bullying) |
| Class distribution | ~12,295 normal / ~852 bullying |

Download file csv
📦 [Download from Google Drive](https://drive.google.com/drive/folders/1wMpf_xQA0_NH8-xDWAxu7Pxdw085LJgj?usp=drive_link)

---

## ⚙️ Pipeline

```
Raw Text → Preprocessing → TF-IDF Vectorization → Model Training → Evaluation
```

**Preprocessing steps:**
- HTML tag removal
- Punctuation removal
- Lowercasing
- Stopword removal (NLTK)
- Stemming (SnowballStemmer)

**Feature Extraction:**
- TF-IDF Vectorizer with 10,000 features, n-gram range (1,2)

---


## 🤖 Models Compared

### Part 1 — Classifier Group

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVC | 0.9224 | 0.4267 | 0.5614 | 0.4848 |
| MultinomialNB | 0.8806 | 0.3103 | 0.6842 | 0.4270 |
| Decision Tree | 0.8996 | 0.3147 | 0.4620 | 0.3744 |

### Part 2 — Advanced Models

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** ⭐ | **0.9228** | **0.4380** | **0.6608** | **0.5268** |
| XGBoost | 0.9449 | 0.7826 | 0.2105 | 0.3318 |
| Random Forest | 0.9414 | 0.7576 | 0.1462 | 0.2451 |

> ✅ **Logistic Regression** was selected as the final model — best balance between Recall and F1-Score, which matters most for detecting bullying cases.

---

## 🏆 Final Model — Logistic Regression (Threshold-tuned)

| Property | Value |
|----------|-------|
| Model type | Logistic Regression |
| TF-IDF features | 10,000 |
| N-gram range | (1, 2) |
| Threshold (tuned) | 0.4962 |
| F1-Score | 0.531 |
| Recall | 0.678 |
| Precision | 0.436 |
| Labels | `0` = normal, `1` = bullying |

> Threshold was tuned to prioritize **Recall** — missing a bullying case is more costly than a false positive.

---

## 📁 Project Structure

```
cyber_bullying_logistic_regression/
├── metadata/
│   └── metadata.json        # Model config & metrics
├── notebooks/
│   └── Project_Cyber_Bullying.ipynb   # Full EDA, training & evaluation
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/trantrongkhangttns/cyber_bullying_logistic_regression
cd cyber_bullying_logistic_regression
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model files**

The trained model files are stored on Google Drive:

📦 [Download from Google Drive](https://drive.google.com/drive/folders/1FoGMncHcccmejjwqTql-Aim617NDv5BR?usp=sharing) — includes:
- `lr_model.pkl` (79 KB)
- `tfidf_vectorizer.pkl` (370 KB)

Place them in a `models/` folder at the project root.

**4. Open the notebook**
```bash
jupyter notebook notebooks/Project_Cyber_Bullying.ipynb
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8 |
| ML | scikit-learn |
| NLP | NLTK, TF-IDF |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook / Google Colab |

---

## 👤 Author

**Trần Trọng Khang**  
📎 [GitHub](https://github.com/trantrongkhangttns)
