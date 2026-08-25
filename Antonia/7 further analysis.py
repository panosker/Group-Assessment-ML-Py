"""
7. Further Analysis]

7.1. Sentiment over time: covers all 4 topics (Climate, Eurovision,
Mundial, AIvsHuman), one subplot per topic, using train_split_scored.csv/test_split_scored.csv.
This needs post timestamps, which live in each topic's raw collection file, not in the merged
Final Concatenated File. The csv all_topics_timestamps.csv merges those in by id.
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TOPIC_COLUMN = "Topic"
SENTIMENT_COLUMN = "Final Opinion"
TEXT_COLUMN = "cleaned_text"
VALID_SENTIMENTS = {"positive", "neutral", "negative"}
TOPICS_ORDER = ["Climate", "Eurovision", "Mundial", "AIvsHuman"]

# ---------------------------------------------------------------------------
# Load the full 4-topic dataset (used by 7a/7b/7c/7d)
# ---------------------------------------------------------------------------
train_df = pd.read_csv("train_split_scored.csv", encoding="utf-8-sig")
test_df = pd.read_csv("test_split_scored.csv", encoding="utf-8-sig")
df = pd.concat([train_df, test_df], ignore_index=True)
df = df.dropna(subset=[TEXT_COLUMN])
df = df[df[SENTIMENT_COLUMN].isin(VALID_SENTIMENTS)]
print(f"Full dataset for 7a/7b/7c/7d: {len(df)} posts across {df[TOPIC_COLUMN].nunique()} topics")

# ---------------------------------------------------------------------------
# 7.1. Sentiment over time with all 4 topics, one subplot each
#One subplot per topic (2x2 grid), each showing daily positive/neutral/negative post counts.
# ---------------------------------------------------------------------------
timestamps = pd.read_csv("all_topics_timestamps.csv")
timestamps["published_at"] = pd.to_datetime(timestamps["published_at"], utc=True, format="mixed")

time_df = df.merge(timestamps[["id", "published_at"]], on="id", how="left")
missing = time_df["published_at"].isna().sum()
if missing:
    print(f"[warning] {missing} posts had no matching timestamp and will be dropped from 7a")
time_df = time_df.dropna(subset=["published_at"])
time_df["date"] = time_df["published_at"].dt.date

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
for ax, topic in zip(axes.flat, TOPICS_ORDER):
    topic_df = time_df[time_df[TOPIC_COLUMN] == topic]
    daily_sentiment = topic_df.groupby(["date", SENTIMENT_COLUMN]).size().unstack(fill_value=0)
    daily_sentiment = daily_sentiment.reindex(columns=["positive", "neutral", "negative"], fill_value=0)
    daily_sentiment.plot(kind="line", marker="o", ax=ax, legend=(topic == TOPICS_ORDER[0]))
    ax.set_title(topic)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Posts")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Sentiment Over Time by Topic")
plt.tight_layout()
plt.savefig("sentiment_over_time.png")
plt.show()
print("Saved sentiment_over_time.png (all 4 topics, one subplot each)")