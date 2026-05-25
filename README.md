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

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** ⭐ | **98.21%** | **98.19%** | **98.21%** | **98.18%** |
| C-Support Vector Classifier | 98.09% | 98.07% | 98.09% | 98.06% |
| Epsilon-Support Vector Regression | 98.03% | 98.09% | 98.03% | 98.00% |
| Ridge Classifier | 97.79% | 97.87% | 97.79% | 97.74% |
| Random Forest Classifier | 97.43% | 97.50% | 97.43% | 97.32% |
| Decision Tree Classifier | 97.01% | 96.96% | 97.01% | 96.98% |

> ✅ **Logistic Regression** was selected as the final model due to best overall performance.

---

## 🏆 Final Model — Logistic Regression (Threshold-tuned)

After threshold tuning on the saved model:

| Metric | Score |
|--------|-------|
| Threshold | 0.4962 |
| F1-Score | 0.531 |
| Recall | 0.678 |
| Precision | 0.436 |

> Note: Threshold was tuned to prioritize **Recall** — minimizing missed bullying cases is more critical than false positives in this context.

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
git clone https://github.com/trantrongkhangtns/cyber_bullying_logistic_regression
cd cyber_bullying_logistic_regression
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model files**

The trained model files are stored on Google Drive (too large for GitHub):

📦 [Download from Google Drive](YOUR_DRIVE_LINK_HERE) — includes:
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

**Trần Trọng Khánh**  
📎 [GitHub](https://github.com/trantrongkhangtns)
