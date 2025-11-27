# Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![NLP](https://img.shields.io/badge/NLP-NLTK-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.14%25-success.svg)

## 📌 Overview

The **Fake News Detection System** is an end-to-end Machine Learning project that classifies news articles as **Real** or **Fake** using Natural Language Processing and Logistic Regression.  
The system achieves **99.14% accuracy** using optimized TF-IDF features.

### Highlights
- High accuracy (99.14%)
- Streamlit web interface
- Fast inference (<1 second)
- TF-IDF vectorization (1–3 n-grams)
- Clean & lightweight ML pipeline

---

## ✨ Features

- Binary classification (Fake / Real)
- Text preprocessing (cleaning, tokenizing, stopword removal)
- TF-IDF feature extraction
- Logistic Regression classifier
- Confidence scoring
- Real-time prediction
- Simple Streamlit UI

---

## 🎬 Demo

### Sample Prediction


<img width="1920" height="1080" alt="Screenshot 2025-11-27 215608" src="https://github.com/user-attachments/assets/d36c02a2-72b4-460e-9a84-0d2b1121de78" />


**Input:**  
`Enter News Artical`


<img width="1920" height="1080" alt="Screenshot 2025-11-27 215647" src="https://github.com/user-attachments/assets/85a78bb2-05cb-4185-971a-7fc397af2632" />

**Output:**  
❌ Fake News (Confidence: 90.3%)


<img width="1920" height="1080" alt="Screenshot 2025-11-27 215712" src="https://github.com/user-attachments/assets/1f0bbb8e-11d4-4a33-93fc-5df3a18ccb38" />

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```
### 2. Install Requirements
```
pip install -r requirements.txt
```
### 3. Install NLTK Data
```
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```
### 💻 Usage
## Train the Model
```
python train_model.py
```
## Run the Streamlit App
```
streamlit run app.py
```

### 📊 Dataset
```
Dataset	Articles
True News	21,417
Fake News	23,481
Total	44,898
```
## Dataset Source: Kaggle Fake News Dataset
# Train/Test Split: 80/20

### 📈 Model Performance
```
Metric	Score
Accuracy	99.14%
Precision	98.77%
Recall	99.44%
F1-Score	99.10%
ROC-AUC	0.9994
```
```
Confusion Matrix
                Predicted
              Fake    Real
Actual Fake   4643     24
Actual Real     53   4260
```
### 🔮 Future Enhancements (Simplified)

- Add support for more languages (Hindi, Marathi)

- Add BERT-based model

- Add explainability tools like LIME/SHAP

- Build mobile app or browser extension
