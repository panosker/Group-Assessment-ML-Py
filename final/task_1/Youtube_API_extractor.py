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
        print(f"Skipping video {video_id} ({video_title[:40]}...): {e}")

    return comments


def save_to_csv(comments, filename):
    if not comments:
        print("No comments collected.")
        return
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
    for c in all_comments[:20]:
        print(f"[{c['timestamp']}] ({c['like_count']} likes) {c['text'][:80]}")

    save_to_csv(all_comments, OUTPUT_FILE)


if __name__ == "__main__":
    main()