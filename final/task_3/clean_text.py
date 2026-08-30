import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_comment(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    text = ' '.join(cleaned_words)
    return text

def main():
    input_file = "Final_Concatenated_File_Cleaned.csv"
    print(f"{input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"'{input_file}' not found")
        return
    df['cleaned_text'] = df['text'].apply(clean_comment)
    output_file = "eurovision_FINAL_cleaned_data.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n New file : {output_file}") 
    print("\n Before and after cleaning (5 comments):")
    for idx, row in df.head(5).iterrows():
        print(f"\n--- Σχόλιο {idx + 1} ---")
        print(f"ΑΡΧΙΚΟ : {row['text']}")
        print(f"ΚΑΘΑΡΟ : {row['cleaned_text']}")

if __name__ == "__main__":
    main()
