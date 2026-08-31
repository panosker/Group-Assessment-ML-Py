

#Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.sparse import hstack


#Import data
TRAIN_FILE = "train_split_scored.csv"
TEST_FILE = "test_split_scored.csv"
TEXT_COLUMN = "cleaned_text"
TOPIC_COLUMN = "Topic"
NGRAM_RANGE = (1, 3)
MIN_DF = 3

train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig").dropna(subset=[TEXT_COLUMN])
test_df = pd.read_csv(TEST_FILE, encoding="utf-8-sig").dropna(subset=[TEXT_COLUMN])

assert train_df[TOPIC_COLUMN].nunique() > 1, (
    "Only one topic present - re-run 4.1/4.2 on the merged 4-topic file first."
)

print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print(train_df[TOPIC_COLUMN].value_counts())

CUSTOM_FEATURE_COLUMNS = [
    c for c in train_df.columns
    if c.startswith("affinity_score_") and not c.endswith("_scaled")
] + (["affinity_margin"] if "affinity_margin" in train_df.columns else [])
print(f"\nCustom feature columns: {CUSTOM_FEATURE_COLUMNS}")



# 5.1. TF-IDF


vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)
X_train_tfidf = vectorizer.fit_transform(train_df[TEXT_COLUMN])
X_test_tfidf = vectorizer.transform(test_df[TEXT_COLUMN])

y_train = train_df[TOPIC_COLUMN]
y_test = test_df[TOPIC_COLUMN]

print(f"TF-IDF matrix shape: train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")


models = {
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    print(f"F1:        {f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return y_pred


print("BASELINE (TF-IDF only)")

baseline_results = {}
for name, model in models.items():
    baseline_results[name] = evaluate(name, model, X_train_tfidf, y_train, X_test_tfidf, y_test)



# 5.2. Enriched representation: TF-IDF + custom affinity scores


X_train_custom = train_df[CUSTOM_FEATURE_COLUMNS].to_numpy()
X_test_custom = test_df[CUSTOM_FEATURE_COLUMNS].to_numpy()

X_train_enriched = hstack([X_train_tfidf, X_train_custom])
X_test_enriched = hstack([X_test_tfidf, X_test_custom])

print(f"Enriched matrix shape: train {X_train_enriched.shape}, test {X_test_enriched.shape}")

print("=" * 50)
print("ENRICHED (TF-IDF + per-topic affinity scores)")
print("=" * 50)
enriched_results = {}
for name, model in models.items():
    model_copy = type(model)(**model.get_params())
    enriched_results[name] = evaluate(name, model_copy, X_train_enriched, y_train, X_test_enriched, y_test)



#5.3. Compare


summary_rows = []
for name in models:
    baseline_f1 = f1_score(y_test, baseline_results[name], average="weighted", zero_division=0)
    enriched_f1 = f1_score(y_test, enriched_results[name], average="weighted", zero_division=0)
    summary_rows.append({
        "model": name,
        "baseline_f1": round(baseline_f1, 3),
        "enriched_f1": round(enriched_f1, 3),
        "diff": round(enriched_f1 - baseline_f1, 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df

best_row = summary_df.loc[summary_df[["baseline_f1", "enriched_f1"]].max(axis=1).idxmax()]
print(f"Best performing combination: {best_row['model']}, "
      f"{'enriched' if best_row['enriched_f1'] >= best_row['baseline_f1'] else 'baseline'} representation "
      f"(weighted F1 = {max(best_row['baseline_f1'], best_row['enriched_f1']):.3f})")