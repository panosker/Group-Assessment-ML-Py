"""
Section 4, Part 2: Custom Word/Phrase Scoring Mechanism
-----------------------------------------------------------------
Scores each term (unigram, bigram, or trigram -- length not fixed)
by how informative it is for a specific topic, using three blended
signals:

  1. INFORMATION GAIN (entropy-based):
     How concentrated a term's occurrences are across the 4 topics.
     A term spread evenly across all topics (like "like", "im") has
     HIGH entropy -> LOW information gain -> LOW importance.
     A term appearing almost only in one topic (like "messi") has
     LOW entropy -> HIGH information gain -> HIGH importance.

  2. TOPIC SHARE:
     What fraction of a term's total occurrences belong to this
     specific topic. Turns a term's overall informativeness into a
     PER-TOPIC score (informative FOR Mundial vs FOR Climate, etc).

  3. AVERAGE TF-IDF WEIGHT:
     How strongly the term is weighted within that topic's documents
     by standard TF-IDF. Prevents single fluke occurrences (which
     can score a perfect concentration by pure chance) from dominating.

  final_score(term, topic) = InfoGain(term) * topic_share(term, topic)
                              * avg_tfidf(term, topic)

Terms are unigrams, bigrams, AND trigrams together (ngram_range=(1,3))
so phrase length is not fixed -- whichever length is most informative
wins on its own merit.

The score table is built from TRAINING data only, then applied to
both train and test to produce a document-level feature with no
information leakage (see build_doc_scores()).

SETUP:
    pip install pandas scikit-learn numpy

USAGE:
    python custom_scoring.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------- CONFIG ----------------------
INPUT_FILE = "Final_Concatenated_File_Cleaned.csv"
TEXT_COLUMN = "cleaned_text"
TOPIC_COLUMN = "Topic"
NGRAM_RANGE = (1, 3)      # unigrams, bigrams, trigrams -- no fixed phrase length
MIN_DF = 3                # a term must appear at least 3 times total to be scored
TEST_SIZE = 0.2
RANDOM_STATE = 42         # matches the split used in topic_terms.py
TOP_N_DISPLAY = 20
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


def build_score_table(train_df, text_col, topic_col, ngram_range, min_df):
    """Computes final_score(term, topic) for every term in the training
    vocabulary. Returns a dict {(term, topic): score} plus the fitted
    vectorizers (needed later to tokenize new documents the same way)."""

    topics = sorted(train_df[topic_col].unique())
    K = len(topics)

    # Raw counts -- used for the entropy/information-gain calculation
    count_vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    count_matrix = count_vec.fit_transform(train_df[text_col])
    terms = count_vec.get_feature_names_out()

    # TF-IDF -- used as the "strength" multiplier
    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    tfidf_matrix = tfidf_vec.fit_transform(train_df[text_col])
    # sanity check: both vectorizers should produce the same vocabulary
    assert list(terms) == list(tfidf_vec.get_feature_names_out())

    topic_masks = {t: (train_df[topic_col] == t).values for t in topics}

    # freq(term, topic) for every topic -> shape (n_terms, K)
    freq_per_topic = np.zeros((len(terms), K))
    tfidf_sum_per_topic = np.zeros((len(terms), K))
    doc_count_per_topic = np.array([m.sum() for m in topic_masks.values()])

    for i, topic in enumerate(topics):
        mask = topic_masks[topic]
        freq_per_topic[:, i] = count_matrix[mask].sum(axis=0).A1
        tfidf_sum_per_topic[:, i] = tfidf_matrix[mask].sum(axis=0).A1

    total_freq = freq_per_topic.sum(axis=1)  # freq(term) across all topics

    # --- 1. Information gain (entropy-based) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        p = freq_per_topic / total_freq[:, None]          # p_i(term) per topic
        p_safe = np.where(p > 0, p, 1)                     # avoid log(0)
        entropy = -np.sum(p * np.log2(p_safe), axis=1)     # H(term)
    info_gain = np.log2(K) - entropy                       # 0 (uninformative) .. log2(K) (fully concentrated)

    # --- 2. Topic share = p_T(term), already computed above as `p` ---
    topic_share = p  # shape (n_terms, K)

    # --- 3. Average TF-IDF weight per topic ---
    avg_tfidf = tfidf_sum_per_topic / doc_count_per_topic[None, :]

    # --- combine ---
    final_scores = info_gain[:, None] * topic_share * avg_tfidf  # shape (n_terms, K)

    score_table = {}
    for i, term in enumerate(terms):
        for j, topic in enumerate(topics):
            score_table[(term, topic)] = float(final_scores[i, j])

    return score_table, count_vec, topics


def top_terms_from_table(score_table, topics, top_n):
    results = {}
    for topic in topics:
        scored = [(term, score) for (term, t), score in score_table.items() if t == topic]
        scored.sort(key=lambda x: x[1], reverse=True)
        results[topic] = scored[:top_n]
    return results


def build_doc_scores(df, score_table, count_vec, topic_col, text_col):
    """Applies the (already-frozen, train-only) score table to every
    document -- works for train OR test since the table itself was
    built from training data only. Terms not present in the training
    vocabulary simply contribute nothing (score 0), which is correct
    behaviour, not an error."""
    analyzer = count_vec.build_analyzer()  # tokenizes text into the same n-grams used to build the table
    sums, avgs, matched_counts = [], [], []

    for _, row in df.iterrows():
        terms_in_doc = analyzer(row[text_col])
        topic = row[topic_col]
        scores = [score_table.get((t, topic), 0.0) for t in terms_in_doc]
        scores = [s for s in scores if s != 0.0]
        sums.append(sum(scores))
        avgs.append(sum(scores) / len(scores) if scores else 0.0)
        matched_counts.append(len(scores))

    df = df.copy()
    df["custom_score_sum"] = sums
    df["custom_score_avg"] = avgs
    df["custom_score_matched_terms"] = matched_counts
    return df


def build_doc_scores_all_topics(df, score_table, count_vec, topics, text_col):
    """Unlike build_doc_scores, this does NOT use the document's own
    true topic label. Instead it scores every document against EVERY
    topic's profile independently, producing one column per topic
    (e.g. affinity_score_Climate, affinity_score_Mundial, ...).

    This is what makes the feature usable at real prediction time --
    the classifier sees "how well does this text match each topic's
    vocabulary" and learns to pick the highest-matching one, rather
    than being handed a single number that was only computable because
    the true label was already known.

    Still leak-free: score_table itself was built from TRAIN data only;
    this function only ever looks values up, for train or test alike.
    """
    analyzer = count_vec.build_analyzer()
    df = df.copy()

    per_topic_sums = {t: [] for t in topics}
    for _, row in df.iterrows():
        terms_in_doc = analyzer(row[text_col])
        for topic in topics:
            scores = [score_table.get((t, topic), 0.0) for t in terms_in_doc]
            per_topic_sums[topic].append(sum(scores))

    for topic in topics:
        df[f"affinity_score_{topic}"] = per_topic_sums[topic]

    # Extra engineered signals: which topic scored highest, and by how much
    score_cols = [f"affinity_score_{t}" for t in topics]
    scores_matrix = df[score_cols].values
    best_idx = scores_matrix.argmax(axis=1)
    sorted_scores = -np.sort(-scores_matrix, axis=1)  # descending
    df["affinity_best_topic"] = [topics[i] for i in best_idx]
    df["affinity_margin"] = sorted_scores[:, 0] - sorted_scores[:, 1]  # gap between top 2 topics

    return df


def main():
    df = load_and_clean(INPUT_FILE)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TOPIC_COLUMN], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows\n")

    # Build score table from TRAIN ONLY
    score_table, count_vec, topics = build_score_table(
        train_df, TEXT_COLUMN, TOPIC_COLUMN, NGRAM_RANGE, MIN_DF
    )
    print(f"Vocabulary size (unigrams+bigrams+trigrams, min_df={MIN_DF}): {len(count_vec.get_feature_names_out())}")

    # Show top terms per topic under the new custom score
    top_terms = top_terms_from_table(score_table, topics, TOP_N_DISPLAY)
    for topic, terms in top_terms.items():
        print(f"\n--- {topic} (custom score) ---")
        for term, score in terms:
            print(f"  {term:30s} {round(score, 4)}")

    # Apply the frozen score table to BOTH train and test (no leakage,
    # and no use of the true label at inference time -- scores every
    # doc against all 4 topics independently)
    train_df = build_doc_scores_all_topics(train_df, score_table, count_vec, topics, TEXT_COLUMN)
    test_df = build_doc_scores_all_topics(test_df, score_table, count_vec, topics, TEXT_COLUMN)

    # Scale the affinity features -- fit on train only, apply to both
    affinity_cols = [f"affinity_score_{t}" for t in topics] + ["affinity_margin"]
    scaler = StandardScaler()
    train_df[[c + "_scaled" for c in affinity_cols]] = scaler.fit_transform(train_df[affinity_cols])
    test_df[[c + "_scaled" for c in affinity_cols]] = scaler.transform(test_df[affinity_cols])

    train_df.to_csv("train_split_scored.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv("test_split_scored.csv", index=False, encoding="utf-8-sig")

    print("\n\nSaved train_split_scored.csv and test_split_scored.csv")
    print("\nSample of new columns (train):")
    print(train_df[[TOPIC_COLUMN] + affinity_cols + ["affinity_best_topic"]].head(10))


if __name__ == "__main__":
    main()
