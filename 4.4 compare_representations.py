"""
Section 4, Part 4: Baseline vs. Enriched Representation Comparison
-----------------------------------------------------------------
Compares topic-classification performance using:
  BASELINE:  TF-IDF (unigrams+bigrams+trigrams) alone
  ENRICHED:  the same TF-IDF matrix + the custom_score_sum_scaled
             feature built in custom_scoring.py

Uses RandomForestClassifier for this comparison specifically (handles
mixed sparse+dense features cleanly and gives interpretable feature
importances). The full 3-classifier comparison (Naive Bayes, KNN,
Random Forest) happens in Section 5 -- this is just to answer whether
the custom feature helps at all.

SETUP:
    pip install pandas scikit-learn scipy

USAGE:
    python compare_representations.py
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------- CONFIG ----------------------
TRAIN_FILE = "train_split_scored.csv"
TEST_FILE = "test_split_scored.csv"
TEXT_COLUMN = "cleaned_text"
TOPIC_COLUMN = "Topic"
CUSTOM_FEATURE_COLUMNS = [
    "affinity_score_AIvsHuman_scaled", "affinity_score_Climate_scaled",
    "affinity_score_Eurovision_scaled", "affinity_score_Mundial_scaled",
    "affinity_margin_scaled"
]
NGRAM_RANGE = (1, 3)
MIN_DF = 3
RANDOM_STATE = 42
# ------------------------------------------------------


def run_classifier(X_train, y_train, X_test, y_test, label):
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")

    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"F1 (macro):    {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}\n")
    print(classification_report(y_test, preds))

    return clf, acc, f1_macro, f1_weighted


def main():
    train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    test_df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

    y_train = train_df[TOPIC_COLUMN]
    y_test = test_df[TOPIC_COLUMN]

    # Fit TF-IDF on train only, transform both
    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    X_train_tfidf = vectorizer.fit_transform(train_df[TEXT_COLUMN])
    X_test_tfidf = vectorizer.transform(test_df[TEXT_COLUMN])
    print(f"TF-IDF baseline matrix shape: train {X_train_tfidf.shape}, test {X_test_tfidf.shape}")

    # ---------------- BASELINE ----------------
    clf_base, acc_base, f1m_base, f1w_base = run_classifier(
        X_train_tfidf, y_train, X_test_tfidf, y_test,
        "BASELINE: TF-IDF only"
    )

    # ---------------- ENRICHED ----------------
    custom_train = train_df[CUSTOM_FEATURE_COLUMNS].values
    custom_test = test_df[CUSTOM_FEATURE_COLUMNS].values

    X_train_enriched = hstack([X_train_tfidf, custom_train])
    X_test_enriched = hstack([X_test_tfidf, custom_test])
    print(f"\nEnriched matrix shape: train {X_train_enriched.shape}, test {X_test_enriched.shape}")

    clf_enr, acc_enr, f1m_enr, f1w_enr = run_classifier(
        X_train_enriched, y_train, X_test_enriched, y_test,
        "ENRICHED: TF-IDF + per-topic affinity scores"
    )

    # ---------------- COMPARISON SUMMARY ----------------
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Metric':<15}{'Baseline':<12}{'Enriched':<12}{'Difference':<12}")
    print(f"{'Accuracy':<15}{acc_base:<12.4f}{acc_enr:<12.4f}{acc_enr-acc_base:+.4f}")
    print(f"{'F1 (macro)':<15}{f1m_base:<12.4f}{f1m_enr:<12.4f}{f1m_enr-f1m_base:+.4f}")
    print(f"{'F1 (weighted)':<15}{f1w_base:<12.4f}{f1w_enr:<12.4f}{f1w_enr-f1w_base:+.4f}")

    # ---------------- INTERPRETABILITY: where do the custom features rank? ----------------
    feature_names = list(vectorizer.get_feature_names_out()) + CUSTOM_FEATURE_COLUMNS
    importances = clf_enr.feature_importances_
    ranking = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}\nINTERPRETABILITY\n{'='*60}")
    for col in CUSTOM_FEATURE_COLUMNS:
        rank_position = next(i for i, (name, _) in enumerate(ranking, start=1) if name == col)
        imp = importances[feature_names.index(col)]
        print(f"{col:35s} rank #{rank_position:4d}/{len(feature_names)}   importance={imp:.5f}")

    print(f"\nTop 10 features overall by importance:")
    for name, imp in ranking[:10]:
        marker = "  <-- custom feature" if name in CUSTOM_FEATURE_COLUMNS else ""
        print(f"  {name:35s} {imp:.5f}{marker}")


if __name__ == "__main__":
    main()
