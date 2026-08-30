import csv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = "eurovision_youtube_comments.csv"      
SCOREBOARD_FILE = "eurovision_scoreboard.csv"     
TEXT_COLUMN = "text"                             
ID_COLUMN = "id"                                  

VALID_LABELS = {
    "p": "positive", "pos": "positive", "positive": "positive",
    "n": "negative", "neg": "negative", "negative": "negative",
    "u": "neutral", "neu": "neutral", "neutral": "neutral",
}

def detect_delimiter(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,").delimiter
    except csv.Error:
        return ","  

def load_comments(path):
    delimiter = detect_delimiter(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return {row[ID_COLUMN]: row for row in reader}

def load_scoreboard(path):
    if not os.path.exists(path):
        return {}, []
    delimiter = detect_delimiter(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        rows = {row[ID_COLUMN]: row for row in reader}
    return rows, fieldnames

def save_scoreboard(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)

def get_ai_opinion(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"

def ask_for_label(comment_num, total, text):
    print(f"\n[{comment_num}/{total}] {text}")
    while True:
        raw = input("  Positive / Negative / Neutral / Skip / Quit  (p/n/u/s/q): ").strip().lower()
        if raw == "q":
            return "QUIT"
        if raw == "s":
            return "SKIP"
        if raw in VALID_LABELS:
            return VALID_LABELS[raw]
        print("  Not recognized -- enter p, n, u, s, or q.")

def main():
    annotator = input("Your name (used as your column header): ").strip()
    if not annotator:
        print("Name can't be empty.")
        return

    comments = load_comments(INPUT_FILE)
    scoreboard, fieldnames = load_scoreboard(SCOREBOARD_FILE)

    if not fieldnames:
        fieldnames = [ID_COLUMN, TEXT_COLUMN, "ai_opinion"]

    if annotator not in fieldnames:
        fieldnames.append(annotator)

    if "ai_opinion" not in fieldnames:
        fieldnames.insert(2, "ai_opinion")

    analyzer = SentimentIntensityAnalyzer()

    for comment_id, comment in comments.items():
        if comment_id not in scoreboard:
            scoreboard[comment_id] = {
                ID_COLUMN: comment_id,
                TEXT_COLUMN: comment[TEXT_COLUMN],
            }
        row = scoreboard[comment_id]

        if not row.get("ai_opinion"):
            row["ai_opinion"] = get_ai_opinion(comment[TEXT_COLUMN], analyzer)

        row.setdefault(annotator, "")

    pending_ids = [cid for cid, row in scoreboard.items() if not row.get(annotator)]
    total = len(pending_ids)
    print(f"\n{total} comments left for you to annotate.\n(Progress saves after every answer -- quit any time with 'q')")

    for i, comment_id in enumerate(pending_ids, start=1):
        row = scoreboard[comment_id]
        label = ask_for_label(i, total, row[TEXT_COLUMN])

        if label == "QUIT":
            break
        if label == "SKIP":
            continue

        row[annotator] = label
        save_scoreboard(SCOREBOARD_FILE, scoreboard, fieldnames)  
    save_scoreboard(SCOREBOARD_FILE, scoreboard, fieldnames)
    print(f"\nSaved progress to {SCOREBOARD_FILE}")


if __name__ == "__main__":
    main()