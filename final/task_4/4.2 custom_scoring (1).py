import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "Final Concatenated File Cleaned v2.csv"
TEXT_COLUMN = "cleaned_text"
TOPIC_COLUMN = "Topic"
NGRAM_RANGE = (1, 3)
MIN_DF = 3                
TEST_SIZE = 0.2
RANDOM_STATE = 42         
TOP_N_DISPLAY = 20

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
    topics = sorted(train_df[topic_col].unique())
    K = len(topics)
    count_vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df)
    count_matrix = count_vec.fit_transform(train_df[text_col])
    terms = count_vec.get_feature_names_out()
    tfidf_vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
    tfidf_matrix = tfidf_vec.fit_transform(train_df[text_col])
    assert list(terms) == list(tfidf_vec.get_feature_names_out())
    topic_masks = {t: (train_df[topic_col] == t).values for t in topics}
    freq_per_topic = np.zeros((len(terms), K))
    tfidf_sum_per_topic = np.zeros((len(terms), K))
    doc_count_per_topic = np.array([m.sum() for m in topic_masks.values()])

    for i, topic in enumerate(topics):
        mask = topic_masks[topic]
        freq_per_topic[:, i] = count_matrix[mask].sum(axis=0).A1
        tfidf_sum_per_topic[:, i] = tfidf_matrix[mask].sum(axis=0).A1
    total_freq = freq_per_topic.sum(axis=1) 

    with np.errstate(divide="ignore", invalid="ignore"):
        p = freq_per_topic / total_freq[:, None]         
        p_safe = np.where(p > 0, p, 1)                     
        entropy = -np.sum(p * np.log2(p_safe), axis=1)     
    info_gain = np.log2(K) - entropy                  
    topic_share = p 
    avg_tfidf = tfidf_sum_per_topic / doc_count_per_topic[None, :]
    final_scores = info_gain[:, None] * topic_share * avg_tfidf  

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
    analyzer = count_vec.build_analyzer() 
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
    score_cols = [f"affinity_score_{t}" for t in topics]
    scores_matrix = df[score_cols].values
    best_idx = scores_matrix.argmax(axis=1)
    sorted_scores = -np.sort(-scores_matrix, axis=1)  # descending
    df["affinity_best_topic"] = [topics[i] for i in best_idx]
    df["affinity_margin"] = sorted_scores[:, 0] - sorted_scores[:, 1] 
    return df


def main():
    df = load_and_clean(INPUT_FILE)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TOPIC_COLUMN], random_state=RANDOM_STATE)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows\n")

    score_table, count_vec, topics = build_score_table(
        train_df, TEXT_COLUMN, TOPIC_COLUMN, NGRAM_RANGE, MIN_DF)
    print(f"Vocabulary size (unigrams+bigrams+trigrams, min_df={MIN_DF}): {len(count_vec.get_feature_names_out())}")

    top_terms = top_terms_from_table(score_table, topics, TOP_N_DISPLAY)
    for topic, terms in top_terms.items():
        print(f"\n--- {topic} (custom score) ---")
        for term, score in terms:
            print(f"  {term:30s} {round(score, 4)}")
    train_df = build_doc_scores_all_topics(train_df, score_table, count_vec, topics, TEXT_COLUMN)
    test_df = build_doc_scores_all_topics(test_df, score_table, count_vec, topics, TEXT_COLUMN)
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
