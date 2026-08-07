# %%
"""
PART 4 - Topic Discovery and Custom Feature Scoring [25 Marks]
Group Coursework - Machine Learning & Python
(Athina, Alex)

4.1  Important terms per topic (TF-IDF)                       [8 Marks]
4.2  Custom topic-specificity scoring mechanism                [9 Marks]
4.3  Turn the scores into per-document classification features [4 Marks]
4.4  Compare custom features vs baseline TF-IDF representation [4 Marks]

Reads ONE combined CSV (all_topics_posts_clean.csv) containing the cleaned
posts from all 4 topics, with columns 'text_clean' and 'topic'. If you
still have 4 separate per-topic CSVs from Part 3, see combine_topic_files()
below to merge them into that single file first.

Requirements:  pip install pandas scikit-learn numpy
"""

import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# CONFIG - ONE combined CSV with the cleaned posts from all 4 topics.
# Required columns: 'text_clean' (cleaned post text) and 'topic'
# (e.g. "Mundial", "Eurovision", "Climate", "Politics").
# ---------------------------------------------------------------
INPUT_FILE = "Final Concatenated File Cleaned.csv"

TEXT_COL = "cleaned_text"   # column with the cleaned post text in this file
TOPIC_COL = "Topic"         # column with the topic label in this file
RANDOM_STATE = 42


# ---------------------------------------------------------------
# Load the single combined corpus
# ---------------------------------------------------------------
def load_corpus(path=INPUT_FILE, text_col=TEXT_COL, topic_col=TOPIC_COL):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(
            f"Could not find '{path}'.\n"
            f"  Looked for it at: {os.path.abspath(path)}\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"Fix: either copy the CSV into that folder, or change INPUT_FILE "
            f"at the top of this script to the file's full path."
        )

    missing = [c for c in (text_col, topic_col) if c not in df.columns]
    if missing:
        raise SystemExit(
            f"'{path}' is missing required column(s): {missing}. "
            f"Expected at least '{text_col}' and '{topic_col}'."
        )

    df = df.rename(columns={text_col: TEXT_COL, topic_col: "topic"})
    df = df.dropna(subset=[TEXT_COL])
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]
    df = df[[TEXT_COL, "topic"]].reset_index(drop=True)

    print(f"Loaded corpus: {len(df)} posts across {df['topic'].nunique()} topic(s) "
          f"({', '.join(sorted(df['topic'].unique()))})")
    return df


# ---------------------------------------------------------------
# OPTIONAL one-off utility: if your group still has 4 separate per-topic
# cleaned CSVs (the output of each member's Part 3 script), run this ONCE
# to combine them into the single file that this script expects. After
# that, just re-run this script normally against INPUT_FILE.
# ---------------------------------------------------------------
def combine_topic_files(topic_files, output_path=INPUT_FILE, text_col=TEXT_COL):
    frames = []
    for topic, path in topic_files.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  [!] {path} not found yet - skipping '{topic}'.")
            continue
        df = df[[text_col]].copy()
        df["topic"] = topic
        frames.append(df)

    if not frames:
        raise SystemExit("No topic files found - nothing to combine.")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Combined {len(combined)} posts from {len(frames)} topic file(s) -> {output_path}")
    return combined


# ---------------------------------------------------------------
# 4.1 - Important terms per topic (TF-IDF based)
# ---------------------------------------------------------------
def top_terms_per_topic(train_df, top_n=15):
    """
    Fits one TF-IDF space on the TRAINING corpus, then for each topic
    averages the TF-IDF vectors of its own documents. Terms with the
    highest average TF-IDF are the ones that best distinguish that topic
    (TF-IDF already downweights words common across every topic, e.g.
    'today', 'people', so what's left near the top is genuinely
    topic-informative).
    """
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
    tfidf = vectorizer.fit_transform(train_df[TEXT_COL])
    terms = np.array(vectorizer.get_feature_names_out())

    results = {}
    for topic in sorted(train_df["topic"].unique()):
        idx = np.where(train_df["topic"].values == topic)[0]
        mean_scores = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        results[topic] = list(zip(terms[top_idx], mean_scores[top_idx]))

    for topic, term_scores in results.items():
        print(f"\nTop terms for '{topic}':")
        for term, score in term_scores[:10]:
            print(f"  {term:<20s} {score:.4f}")

    return results, vectorizer


# ---------------------------------------------------------------
# 4.2 - Custom topic-specificity scoring mechanism
# ---------------------------------------------------------------
def compute_topic_scores(train_df, min_total_count=5, eps=1e-6):
    """
    For each term w and topic T (TRAIN split only):

        rel_freq(w, T)    = count of w in T's docs        / total words in T's docs
        rel_freq(w, ~T)   = count of w in the OTHER docs   / total words in the OTHER docs
        score(w, T)       = log( (rel_freq(w,T) + eps) / (rel_freq(w,~T) + eps) )

    score(w, T) > 0  -> w appears relatively MORE inside topic T than elsewhere
                        (topic-specific term)
    score(w, T) < 0  -> w appears relatively LESS inside topic T (generic /
                        anti-topic term)
    score(w, T) ~ 0  -> no strong topic association

    A term is only scored if it appears at least `min_total_count` times in
    the whole training corpus, to stop rare/noisy words from getting wild
    scores based on 1-2 occurrences.
    """
    topics = sorted(train_df["topic"].unique())

    topic_word_counts = {t: Counter() for t in topics}
    topic_total_words = {t: 0 for t in topics}
    corpus_word_counts = Counter()

    for _, row in train_df.iterrows():
        tokens = row[TEXT_COL].split()
        topic_word_counts[row["topic"]].update(tokens)
        topic_total_words[row["topic"]] += len(tokens)
        corpus_word_counts.update(tokens)

    grand_total_words = sum(topic_total_words.values())

    scores = {t: {} for t in topics}
    eligible_terms = [w for w, c in corpus_word_counts.items() if c >= min_total_count]

    for topic in topics:
        rest_counts = corpus_word_counts - topic_word_counts[topic]
        rest_total = grand_total_words - topic_total_words[topic]

        for w in eligible_terms:
            freq_in_topic = topic_word_counts[topic].get(w, 0) / max(topic_total_words[topic], 1)
            freq_in_rest = rest_counts.get(w, 0) / max(rest_total, 1)
            scores[topic][w] = math.log((freq_in_topic + eps) / (freq_in_rest + eps))

    for topic in topics:
        top5 = sorted(scores[topic].items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"Most topic-specific terms for '{topic}': "
              + ", ".join(f"{w} ({s:.2f})" for w, s in top5))

    return scores


# ---------------------------------------------------------------
# 4.3 - Turn the scores into per-document features (no leakage)
# ---------------------------------------------------------------
def score_document(tokens, term_scores):
    matched = [term_scores[t] for t in tokens if t in term_scores]
    return float(np.mean(matched)) if matched else 0.0


def build_score_features(df, topic_scores):
    """
    Adds one numeric column per topic: score_<topic>. `topic_scores` was
    computed on the TRAINING split only (see compute_topic_scores); here we
    just look values up, so calling this on the test split introduces no
    leakage - the test documents never influence the score table itself.
    """
    out = pd.DataFrame(index=df.index)
    for topic, term_scores in topic_scores.items():
        out[f"score_{topic}"] = df[TEXT_COL].apply(
            lambda text: score_document(text.split(), term_scores)
        )
    return out


# ---------------------------------------------------------------
# 4.4 - Compare custom scoring representation vs baseline TF-IDF
# ---------------------------------------------------------------
def compare_representations(train_df, test_df, train_feats, test_feats):
    y_train, y_test = train_df["topic"], test_df["topic"]

    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
    X_train_tfidf = vectorizer.fit_transform(train_df[TEXT_COL])
    X_test_tfidf = vectorizer.transform(test_df[TEXT_COL])

    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_train_tfidf, csr_matrix(train_feats.values)])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(test_feats.values)])

    configs = {
        "Baseline: TF-IDF only": (X_train_tfidf, X_test_tfidf),
        "TF-IDF + custom topic scores": (X_train_combined, X_test_combined),
        "Custom topic scores only": (train_feats.values, test_feats.values),
    }

    print("\n--- Representation comparison (Logistic Regression) ---")
    results = []
    for name, (X_tr, X_te) in configs.items():
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results.append({"representation": name, "accuracy": acc, "macro_f1": f1})
        print(f"{name:<32s}  accuracy={acc:.3f}  macro-F1={f1:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    # One-off step: if you still have 4 separate per-topic cleaned CSVs from
    # Part 3, uncomment this to build all_topics_posts_clean.csv once, then
    # comment it back out and just re-run against INPUT_FILE from then on.
    # combine_topic_files({
    #     "Mundial": "mundial_posts_clean.csv",
    #     "Eurovision": "eurovision_posts_clean.csv",
    #     "Climate": "climate_posts_clean.csv",
    #     "Politics": "politics_posts_clean.csv",
    # })

    corpus = load_corpus(INPUT_FILE)

    train_df, test_df = train_test_split(
        corpus, test_size=0.25, stratify=corpus["topic"], random_state=RANDOM_STATE
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    print("\n=== 4.1 Important terms per topic ===")
    top_terms, _ = top_terms_per_topic(train_df)

    print("\n=== 4.2 Custom topic-specificity scores (train split only) ===")
    topic_scores = compute_topic_scores(train_df)

    print("\n=== 4.3 Building per-document score features ===")
    train_feats = build_score_features(train_df, topic_scores)
    test_feats = build_score_features(test_df, topic_scores)
    print(train_feats.head())

    print("\n=== 4.4 Comparing representations ===")
    comparison = compare_representations(train_df, test_df, train_feats, test_feats)

    train_out = pd.concat([train_df.reset_index(drop=True), train_feats], axis=1)
    test_out = pd.concat([test_df.reset_index(drop=True), test_feats], axis=1)
    train_out.to_csv("topic_features_train.csv", index=False)
    test_out.to_csv("topic_features_test.csv", index=False)
    print("\nSaved topic_features_train.csv / topic_features_test.csv for Part 5.")