"""
YouTube Comments Scraper for Sentiment/Topic Analysis Project
-----------------------------------------------------------------
Searches YouTube for videos matching a topic, then collects top-level
comments from those videos using the YouTube Data API v3.

SETUP (one-time):
1. pip install google-api-python-client langdetect
2. Go to https://console.cloud.google.com/
3. Create a project (or use an existing one) -- top left "Select a project" -> "New Project"
4. Go to "APIs & Services" -> "Library", search "YouTube Data API v3", click Enable
5. Go to "APIs & Services" -> "Credentials" -> "Create Credentials" -> "API key"
6. Copy that key and paste it below (or set it as an env variable).
   No billing account / credit card is required for the free quota tier
   (10,000 units/day, which is plenty for a few hundred comments).

USAGE:
    python youtube_comments_scraper.py
    (edit the CONFIG section below to change topic / search query / count)
"""

import os
import csv
from datetime import datetime
from googleapiclient.discovery import build
from langdetect import detect, LangDetectException


API_KEY = os.environ.get("Eurovision_YouTube_Key", "AIzaSyAD04aLH2ms4JDJYiUS6YTUyz_j70DaTuY")

TOPIC_LABEL = "Eurovision"            
SEARCH_QUERY = "Eurovision"      
MAX_VIDEOS = 10                      
COMMENTS_PER_VIDEO = 25             
TARGET_LANGUAGE = "en" 
OUTPUT_FILE = f"{TOPIC_LABEL.lower()}_youtube_comments.csv"



def is_target_language(text, target_language):
    """Returns True if the text is detected as the target language.
    Over-fetches and filters client-side since the YouTube API has no
    language filter for comments. Short/emoji-only comments that can't
    be detected reliably are dropped rather than guessed."""
    if target_language is None:
        return True
    try:
        return detect(text) == target_language
    except LangDetectException:
        return False


def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


def search_videos(youtube, query, max_results):
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=max_results,
        order="relevance",
    )
    response = request.execute()
    return [
        {"video_id": item["id"]["videoId"], "title": item["snippet"]["title"]}
        for item in response.get("items", [])
    ]


def fetch_comments(youtube, video_id, video_title, max_results, target_language):
    comments = []
    # Over-fetch (up to the API's page cap of 100) since some comments will
    # get dropped by the language filter -- we still trim to max_results after.
    fetch_count = min(max_results * 3, 100)
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=fetch_count,
            order="relevance",
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            if len(comments) >= max_results:
                break
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            text = top_comment["textDisplay"]

            if not is_target_language(text, target_language):
                continue

            comments.append({
                "id": item["id"],
                "topic": TOPIC_LABEL,
                "video_id": video_id,
                "video_title": video_title,
                "text": text,
                "author": top_comment["authorDisplayName"],
                "timestamp": top_comment["publishedAt"],
                "like_count": top_comment["likeCount"],
            })
    except Exception as e:
        # Some videos have comments disabled -- skip them instead of crashing
        print(f"Skipping video {video_id} ({video_title[:40]}...): {e}")

    return comments


def save_to_csv(comments, filename):
    if not comments:
        print("No comments collected.")
        return
    # utf-8-sig adds a BOM so Excel correctly detects UTF-8 instead of
    # misreading accented/special characters as Windows-1252 (the "â„¢" bug).
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=comments[0].keys())
        writer.writeheader()
        writer.writerows(comments)
    print(f"Saved {len(comments)} comments to {filename}")


def main():
    youtube = get_youtube_client()

    videos = search_videos(youtube, SEARCH_QUERY, MAX_VIDEOS)
    print(f"Found {len(videos)} videos for query '{SEARCH_QUERY}'")

    all_comments = []
    for video in videos:
        video_comments = fetch_comments(
            youtube, video["video_id"], video["title"], COMMENTS_PER_VIDEO,
            TARGET_LANGUAGE
        )
        all_comments.extend(video_comments)

    # Print first 10-20 comments (required for the assignment's IDE output)
    for c in all_comments[:20]:
        print(f"[{c['timestamp']}] ({c['like_count']} likes) {c['text'][:80]}")

    save_to_csv(all_comments, OUTPUT_FILE)


if __name__ == "__main__":
    main()