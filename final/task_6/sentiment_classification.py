import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

filename = "Final Concatenated File Cleaned v2.csv"
df = pd.read_csv(filename)


sentiment_counts = df['Final Opinion'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index.str.capitalize(), 
        autopct='%1.1f%%', startangle=140, colors=['#2ca02c', '#d62728', '#7f7f7f'])
plt.title('Initial Sentiment Distribution')
plt.savefig('6_1_sentiment_distribution.png', bbox_inches='tight', dpi=300)
plt.close()


df_binary = df[df['Final Opinion'].isin(['positive', 'negative'])].copy()
df_binary['cleaned_text'] = df_binary['cleaned_text'].fillna("")

print("Dataset prepared. Binary classification selected (Neutral posts removed).")
print(f"Total instances for training/testing: {len(df_binary)}\n")

X_text = df_binary['cleaned_text']
y = df_binary['Final Opinion']

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)


models = {
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}


custom_cols = [col for col in df_binary.columns if 'affinity' in col or 'score' in col]
if custom_cols:
    print(f"Found custom features from Section 4: {custom_cols}")
    print("Integrating custom features with TF-IDF representation...\n")
    

    train_custom = df_binary.loc[X_train_text.index, custom_cols].fillna(0).values
    test_custom = df_binary.loc[X_test_text.index, custom_cols].fillna(0).values
    
    X_train_final = hstack([X_train_tfidf, train_custom])
    X_test_final = hstack([X_test_tfidf, test_custom])
else:
    print("No custom features from Section 4 found. Proceeding with standard TF-IDF representation.\n")
    X_train_final = X_train_tfidf
    X_test_final = X_test_tfidf

print("Αξιολογηση Μοντελων \n")
print(f"{'Model':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")


model_names = []
f1_scores = []

for name, model in models.items():
    # Εκπαίδευση
    model.fit(X_train_final, y_train)
    
    # Πρόβλεψη
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='positive', zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label='positive', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='positive', zero_division=0)
    model_names.append(name)
    f1_scores.append(f1)
    print(f"{name:<25} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

# Δημιουργία γραφήματος σύγκρισης
plt.figure(figsize=(8, 5))
plt.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('F1 Score (Positive Class)')
plt.title('Model Comparison for Sentiment Classification')
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('6_3_models_comparison.png', bbox_inches='tight', dpi=300)
plt.close()



