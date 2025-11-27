import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, precision_score, 
                            recall_score, f1_score, roc_auc_score)
import re
from nltk.corpus import stopwords
import numpy as np
import time

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

# Start timing
start_time = time.time()

# STEP 1: Load Data
print("\nSTEP 1: Loading datasets...")
try:
    true_df = pd.read_csv('data/true.csv')
    fake_df = pd.read_csv('data/fake.csv')
    print(f"Loaded {len(true_df)} true news articles")
    print(f"Loaded {len(fake_df)} fake news articles")
except FileNotFoundError:
    print("Error: Please make sure true.csv and fake.csv are in the 'data' folder")
    exit()

# STEP 2: Add Labels
print("\nSTEP 2: Adding labels...")
true_df['label'] = 1  # 1 = True News
fake_df['label'] = 0  # 0 = Fake News
print("Labels added: 1=True, 0=Fake")

# STEP 3: Combine Datasets
print("\nSTEP 3: Combining datasets...")
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
print(f"Total articles: {len(df)}")
print(f"   Real News: {len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)")
print(f"   Fake News: {len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%)")

# STEP 4: Prepare Text Data
print("\nSTEP 4: Preparing text data...")
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'] + " " + df['text']
elif 'text' in df.columns:
    df['content'] = df['text']
elif 'title' in df.columns:
    df['content'] = df['title']
else:
    print("Error: No 'title' or 'text' column found in CSV files")
    exit()

# Remove missing values
df = df.dropna(subset=['content', 'label'])
print(f"Prepared {len(df)} articles")

# STEP 5: Clean Text
print("\nSTEP 5: Cleaning text...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

df['content'] = df['content'].apply(clean_text)
print("Text cleaning completed")

# STEP 6: Split Data
print("\nSTEP 6: Splitting data into train and test sets...")
X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} articles")
print(f"   Real: {sum(y_train==1)} | Fake: {sum(y_train==0)}")
print(f"Testing set: {len(X_test)} articles")
print(f"   Real: {sum(y_test==1)} | Fake: {sum(y_test==0)}")

# STEP 7: Convert Text to Numbers (TF-IDF)
print("\nSTEP 7: Converting text to numerical features...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words='english',
    min_df=2,
    max_df=0.8
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Created {X_train_tfidf.shape[1]} features")
print(f"   Feature matrix shape: {X_train_tfidf.shape}")
print(f"   Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")

# STEP 8: Train Model
print("\nSTEP 8: Training Logistic Regression model...")
training_start = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
training_time = time.time() - training_start
print(f"Model training completed in {training_time:.2f} seconds")

# STEP 9: Evaluate Model on Training Data (for diagnosis)
print("\n" + "=" * 60)
print("STEP 9: MODEL EVALUATION RESULTS")
print("=" * 60)

# First evaluate on training data to check for overfitting
print("\nTRAINING SET EVALUATION:")
print("-" * 60)
y_train_pred = model.predict(X_train_tfidf)
y_train_pred_proba = model.predict_proba(X_train_tfidf)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print(f"   Training Accuracy:  {train_accuracy * 100:.2f}%")
print(f"   Training Precision: {train_precision * 100:.2f}%")
print(f"   Training Recall:    {train_recall * 100:.2f}%")
print(f"   Training F1-Score:  {train_f1 * 100:.2f}%")

# Training confusion matrix
cm_train = confusion_matrix(y_train, y_train_pred)
tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
print(f"\n   Training Confusion Matrix:")
print(f"                    Predicted")
print(f"                 Fake    Real")
print(f"   Actual Fake   {tn_train:5d}   {fp_train:5d}")
print(f"   Actual Real   {fn_train:5d}   {tp_train:5d}")

# Check for misclassified samples in training data (diagnostic)
print(f"\n   Training Set Misclassifications:")
print(f"   False Positives (Real marked as Fake): {fp_train}")
print(f"   False Negatives (Fake marked as Real): {fn_train}")
if fp_train > 0 or fn_train > 0:
    print(f"   ⚠️  Model has errors on training data - checking sample misclassifications...")
    # Show a few examples of misclassifications
    misclassified_indices = []
    for idx in range(len(y_train)):
        if y_train.iloc[idx] != y_train_pred[idx]:
            misclassified_indices.append(idx)
    
    if len(misclassified_indices) > 0:
        print(f"   Found {len(misclassified_indices)} misclassified training samples")
        # Show first 3 examples
        for i, idx in enumerate(misclassified_indices[:3]):
            actual_label = "True News" if y_train.iloc[idx] == 1 else "Fake News"
            pred_label = "True News" if y_train_pred[idx] == 1 else "Fake News"
            confidence = y_train_pred_proba[idx][y_train_pred[idx]]
            sample_text = X_train.iloc[idx][:100] + "..." if len(X_train.iloc[idx]) > 100 else X_train.iloc[idx]
            print(f"\n   Example {i+1}:")
            print(f"   Actual: {actual_label}, Predicted: {pred_label} (confidence: {confidence*100:.1f}%)")
            print(f"   Text preview: {sample_text}")

# Test Model on Test Data
print("\nTEST SET EVALUATION:")
print("-" * 60)
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# For binary classification, we can calculate AUC
try:
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
except:
    auc_score = 0.0

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nOVERALL PERFORMANCE METRICS (TEST SET):")
print("-" * 60)
print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   Precision: {precision * 100:.2f}%")
print(f"   Recall:    {recall * 100:.2f}%")
print(f"   F1-Score:  {f1 * 100:.2f}%")
if auc_score > 0:
    print(f"   ROC-AUC:   {auc_score:.4f}")

# Compare training vs test performance
print("\nTRAINING vs TEST COMPARISON:")
print("-" * 60)
print(f"   Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"   Test Accuracy:     {accuracy * 100:.2f}%")
accuracy_diff = train_accuracy - accuracy
if accuracy_diff > 0.1:
    print(f"   ⚠️  WARNING: Large gap ({accuracy_diff*100:.2f}%) suggests possible overfitting!")
elif accuracy_diff < -0.05:
    print(f"   ⚠️  WARNING: Test accuracy higher than training - possible data leakage or issue!")
else:
    print(f"   ✅ Gap is reasonable ({accuracy_diff*100:.2f}%)")

print("\nCONFUSION MATRIX (TEST SET):")
print("-" * 60)
print(f"                    Predicted")
print(f"                 Fake    Real")
print(f"   Actual Fake   {tn:5d}   {fp:5d}")
print(f"   Actual Real   {fn:5d}   {tp:5d}")

print("\nDETAILED BREAKDOWN (TEST SET):")
print("-" * 60)
print(f"   True Negatives (Correctly identified Fake):  {tn} ({tn/len(y_test)*100:.1f}%)")
print(f"   True Positives (Correctly identified Real):  {tp} ({tp/len(y_test)*100:.1f}%)")
print(f"   False Positives (Real marked as Fake):       {fp} ({fp/len(y_test)*100:.1f}%)")
print(f"   False Negatives (Fake marked as Real):       {fn} ({fn/len(y_test)*100:.1f}%)")

print("\nCLASSIFICATION REPORT (TEST SET):")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Fake News', 'True News']))

# Calculate confidence distribution (TEST SET)
print("\nCONFIDENCE DISTRIBUTION (TEST SET):")
print("-" * 60)
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"   Mean Confidence: {np.mean(confidence_scores)*100:.2f}%")
print(f"   Median Confidence: {np.median(confidence_scores)*100:.2f}%")
print(f"   Min Confidence: {np.min(confidence_scores)*100:.2f}%")
print(f"   Max Confidence: {np.max(confidence_scores)*100:.2f}%")

# High/Low confidence predictions
high_conf = sum(confidence_scores > 0.9)
low_conf = sum(confidence_scores < 0.6)
print(f"   High Confidence (>90%): {high_conf} predictions ({high_conf/len(confidence_scores)*100:.1f}%)")
print(f"   Low Confidence (<60%):  {low_conf} predictions ({low_conf/len(confidence_scores)*100:.1f}%)")

# Training confidence distribution
print("\nCONFIDENCE DISTRIBUTION (TRAINING SET):")
print("-" * 60)
train_confidence_scores = np.max(y_train_pred_proba, axis=1)
print(f"   Mean Confidence: {np.mean(train_confidence_scores)*100:.2f}%")
print(f"   Median Confidence: {np.median(train_confidence_scores)*100:.2f}%")
print(f"   Min Confidence: {np.min(train_confidence_scores)*100:.2f}%")
print(f"   Max Confidence: {np.max(train_confidence_scores)*100:.2f}%")

train_high_conf = sum(train_confidence_scores > 0.9)
train_low_conf = sum(train_confidence_scores < 0.6)
print(f"   High Confidence (>90%): {train_high_conf} predictions ({train_high_conf/len(train_confidence_scores)*100:.1f}%)")
print(f"   Low Confidence (<60%):  {train_low_conf} predictions ({train_low_conf/len(train_confidence_scores)*100:.1f}%)")

# STEP 10: Save Model
print("\nSTEP 10: Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved as 'model.pkl'")
print("Vectorizer saved as 'vectorizer.pkl'")

# Summary
total_time = time.time() - start_time
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nTotal Training Time: {total_time:.2f} seconds")
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print(f"Model Size: ~50 MB (estimated)")
print(f"Vectorizer Size: ~10 MB (estimated)")
print(f"\nNext Step: Run the web app using 'streamlit run app.py'")
print("=" * 60)