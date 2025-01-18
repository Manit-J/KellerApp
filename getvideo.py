import requests

def get_video_url(word, source="signstation"):
    # Construct the video URL
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"

    # Check if the video exists
    response = requests.head(video_url)
    if response.status_code == 200:
        return video_url
    else:
        return None

# Example usage
word = "thanks"
video_url = get_video_url(word)

if video_url:
    print(f"Video for '{word}': {video_url}")
else:
    print(f"No video found for '{word}'.")
