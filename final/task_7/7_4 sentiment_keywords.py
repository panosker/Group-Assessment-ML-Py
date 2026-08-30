import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import pos_tag
from collections import Counter

DATA_FILE = "Final_Concatenated_File_Cleaned.csv"
CLEAN_TEXT_COL = "cleaned_text"
SENTIMENT_COL = "Final Opinion"
MIN_APPEARANCES = 3
TOP_K = 15
VALID_POS_TAGS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}


def prepare_dataset(path):
    data = pd.read_csv(path, encoding="utf-8-sig")
    row_count_before = len(data)
    data = data.dropna(subset=[CLEAN_TEXT_COL, SENTIMENT_COL])
    data = data[data[CLEAN_TEXT_COL].str.strip() != ""]
    removed = row_count_before - len(data)
    if removed:
        print(f"Αφαιρέθηκαν {removed} γραμμές με κενό κείμενο ή συναίσθημα")
    return data.reset_index(drop=True)


def filter_by_pos(data, text_col, candidate_terms):
    term_lookup = set(candidate_terms)
    tag_history = {}
    for comment in data[text_col]:
        words = comment.split()
        if not words:
            continue
        tagged_words = pos_tag(words)
        for word, tag in tagged_words:
            if word in term_lookup:
                tag_history.setdefault(word, Counter())[tag] += 1
    surviving_terms = set()
    for word, tag_counter in tag_history.items():
        dominant_tag, _ = tag_counter.most_common(1)[0]
        if dominant_tag in VALID_POS_TAGS:
            surviving_terms.add(word)
    print(f"Κρατήθηκαν {len(surviving_terms)} από {len(candidate_terms)} όρους ως επίθετα ή επιρρήματα")
    return surviving_terms


def rank_by_raw_frequency(data, text_col, class_col, vocab, top_k):
    tfidf = TfidfVectorizer(vocabulary=sorted(vocab))
    matrix = tfidf.fit_transform(data[text_col])
    feature_names = tfidf.get_feature_names_out()

    output = {}
    for sentiment_class in sorted(data[class_col].unique()):
        row_mask = (data[class_col] == sentiment_class).values
        class_scores = matrix[row_mask].sum(axis=0).A1
        best_indices = class_scores.argsort()[::-1][:top_k]
        output[sentiment_class] = [
            (feature_names[i], round(class_scores[i], 3)) for i in best_indices if class_scores[i] > 0
        ]
    return output


def rank_by_custom_score(data, text_col, class_col, vocab, top_k):
    sentiment_classes = sorted(data[class_col].unique())
    num_classes = len(sentiment_classes)
    ordered_vocab = sorted(vocab)

    counts = CountVectorizer(vocabulary=ordered_vocab)
    count_matrix = counts.fit_transform(data[text_col])
    feature_names = counts.get_feature_names_out()

    tfidf = TfidfVectorizer(vocabulary=ordered_vocab)
    tfidf_matrix = tfidf.fit_transform(data[text_col])

    class_masks = {c: (data[class_col] == c).values for c in sentiment_classes}
    class_freq = np.zeros((len(feature_names), num_classes))
    class_tfidf_sum = np.zeros((len(feature_names), num_classes))
    class_doc_counts = np.array([mask.sum() for mask in class_masks.values()])

    for idx, sentiment_class in enumerate(sentiment_classes):
        mask = class_masks[sentiment_class]
        class_freq[:, idx] = count_matrix[mask].sum(axis=0).A1
        class_tfidf_sum[:, idx] = tfidf_matrix[mask].sum(axis=0).A1

    total_occurrences = class_freq.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        class_probability = class_freq / total_occurrences[:, None]
        safe_probability = np.where(class_probability > 0, class_probability, 1)
        term_entropy = -np.sum(class_probability * np.log2(safe_probability), axis=1)

    info_gain = np.log2(num_classes) - term_entropy
    avg_tfidf_weight = class_tfidf_sum / class_doc_counts[None, :]
    combined_score = info_gain[:, None] * class_probability * avg_tfidf_weight

    output = {}
    for idx, sentiment_class in enumerate(sentiment_classes):
        ranked = [
            (feature_names[i], round(combined_score[i, idx], 4))
            for i in range(len(feature_names)) if combined_score[i, idx] > 0
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        output[sentiment_class] = ranked[:top_k]
    return output


def display_comparison(frequency_results, association_results, sentiment_classes, top_k):
    for sentiment_class in sentiment_classes:
        print(f"\nΣυναίσθημα: {sentiment_class}")
        print(f"{'Συχνοί όροι':<40}{'Ουσιαστικά συνδεδεμένοι όροι'}")

        freq_terms = frequency_results.get(sentiment_class, [])
        assoc_terms = association_results.get(sentiment_class, [])

        for i in range(top_k):
            left = f"{freq_terms[i][0]} ({freq_terms[i][1]})" if i < len(freq_terms) else ""
            right = f"{assoc_terms[i][0]} ({assoc_terms[i][1]})" if i < len(assoc_terms) else ""
            print(f"{left:<40}{right}")


def main():
    #φόρτωση και καθαρισμός του dataset από κενές τιμές
    dataset = prepare_dataset(DATA_FILE)
    sentiment_classes = sorted(dataset[SENTIMENT_COL].unique())
    print(f"Κατηγορίες συναισθήματος: {sentiment_classes}\n")

    #δημιουργία αρχικού λεξιλογίου υποψήφιων όρων με βάση την ελάχιστη συχνότητα εμφάνισης
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=MIN_APPEARANCES)
    vectorizer.fit(dataset[CLEAN_TEXT_COL])
    candidate_terms = vectorizer.get_feature_names_out()
    print(f"Υποψήφιο λεξιλόγιο πριν το φιλτράρισμα: {len(candidate_terms)} όροι")

    #διατήρηση μόνο των όρων που λειτουργούν ως επίθετα ή επιρρήματα στο πλαίσιο της πρότασής τους
    valid_terms = filter_by_pos(dataset, CLEAN_TEXT_COL, candidate_terms)

    #υπολογισμός των πιο συχνών όρων ανά κατηγορία συναισθήματος
    frequency_results = rank_by_raw_frequency(dataset, CLEAN_TEXT_COL, SENTIMENT_COL, valid_terms, TOP_K)

    #υπολογισμός των όρων με την υψηλότερη πραγματική συσχέτιση με κάθε κατηγορία συναισθήματος
    association_results = rank_by_custom_score(dataset, CLEAN_TEXT_COL, SENTIMENT_COL, valid_terms, TOP_K)

    #παρουσίαση των δύο λιστών παράλληλα για σύγκριση
    display_comparison(frequency_results, association_results, sentiment_classes, TOP_K)


if __name__ == "__main__":
    main()