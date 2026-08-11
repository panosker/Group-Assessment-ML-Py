"""
Section 4, Part 1: Identify the Most Important Terms Per Topic
-----------------------------------------------------------------
Uses TF-IDF (fit on TRAINING data only) to find the top distinguishing
terms for each topic, as both unigrams and bigrams.

Reused later: the train/test split saved here (train_df / test_df) should
be reused for the rest of Section 4 (custom scoring) and Section 5
(topic classification) so the whole pipeline stays consistent.

SETUP:
    pip install pandas scikit-learn

USAGE:
    python topic_terms.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ---------------------- CONFIG ----------------------
INPUT_FILE = "Final_Concatenated_File_Cleaned.csv"
TEXT_COLUMN = "cleaned_text"
TOPIC_COLUMN = "Topic"
TOP_N = 20                # how many top terms to show per topic
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ------------------------------------------------------


def load_and_clean(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    before = len(df)
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing/empty {TEXT_COLUMN}")
    return df.reset_index(drop=True)


def top_terms_per_topic(train_df, text_col, topic_col, ngram_range, top_n):
    """Fits TF-IDF on the training text only, then for each topic sums
    the TF-IDF scores of its rows and returns the top_n highest-scoring
    terms. Fitting on train-only avoids leaking test-set vocabulary into
    the term rankings."""
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=3)
    tfidf_matrix = vectorizer.fit_transform(train_df[text_col])
    terms = vectorizer.get_feature_names_out()

    results = {}
    for topic in train_df[topic_col].unique():
        mask = (train_df[topic_col] == topic).values
        topic_scores = tfidf_matrix[mask].sum(axis=0).A1  # sum TF-IDF across topic's rows
        top_indices = topic_scores.argsort()[::-1][:top_n]
        results[topic] = [(terms[i], round(topic_scores[i], 3)) for i in top_indices if topic_scores[i] > 0]
    return results


def print_results(title, results):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for topic, terms in results.items():
        print(f"\n--- {topic} ---")
        for term, score in terms:
            print(f"  {term:25s} {score}")


def main():
    df = load_and_clean(INPUT_FILE)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TOPIC_COLUMN], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    train_df.to_csv("train_split.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("test_split.csv", index=False, encoding="utf-8-sig")

    unigram_results = top_terms_per_topic(train_df, TEXT_COLUMN, TOPIC_COLUMN, (1, 1), TOP_N)
    print_results("TOP UNIGRAMS PER TOPIC (TF-IDF, train-only)", unigram_results)

    bigram_results = top_terms_per_topic(train_df, TEXT_COLUMN, TOPIC_COLUMN, (2, 2), TOP_N)
    print_results("TOP BIGRAMS PER TOPIC (TF-IDF, train-only)", bigram_results)


if __name__ == "__main__":
    main()
