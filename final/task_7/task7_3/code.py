import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_representation(X_train, X_test, y_train, y_test, ngram_range, rep_name):
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=2)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test) 
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='positive', zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label='positive', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='positive', zero_division=0)
    
    print(f".  {rep_name}")
    print(f"   Features: {len(tfidf.get_feature_names_out())}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")
    print("\n")

def main():
    filename = "Final Concatenated File Cleaned v2.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"❌ Το αρχείο '{filename}' δεν βρέθηκε.")
        return

    df_binary = df[df['Final Opinion'].isin(['positive', 'negative'])].copy()
    df_binary['cleaned_text'] = df_binary['cleaned_text'].fillna("")
    X = df_binary['cleaned_text']
    y = df_binary['Final Opinion']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("ΣΥΓΚΡΙΣΗ RANDOM FOREST")
    evaluate_representation(X_train, X_test, y_train, y_test, (1, 1), "Μεμονωμένες Λέξεις")
    evaluate_representation(X_train, X_test, y_train, y_test, (2, 2), "Bi-grams")

if __name__ == "__main__":
    main()