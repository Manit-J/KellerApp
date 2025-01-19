import streamlit as st
import time
import requests
import re
import uuid
from PIL import Image
from groq import Groq

###############################################################################
# 1) BSL “Simplification” with Groq
###############################################################################
def bsl_simplify_with_groq(client, text, max_keywords=20):
    """
    Uses Groq's LLM to convert an English sentence into a list of essential
    words for British Sign Language (BSL). Ensures each word is separate.
    """
    example_input = "What is your name?"
    example_output = "what, your, name"

    prompt = f"""
You are an assistant that converts English sentences into a list of essential words for British Sign Language (BSL).
Preserve question words (who, what, when, where, why, how), pronouns (I, you, she, he, we, they),
and time references (when, today, tomorrow). Remove only minimal filler words ('is', 'are', 'am', 'the', 'of').

IMPORTANT:
1) Return each essential word separately. Do not merge multiple words.
2) Return your final answer as a comma-separated list.

Example:
Input: "{example_input}"
Output: "{example_output}"

Now convert this sentence:
"{text.strip()}"
""".strip()

    st.write("[DEBUG] BSL Simplify Prompt:", prompt)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or whichever Groq text model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts English sentences into BSL-friendly keywords."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_completion_tokens=128,
        top_p=1,
        stop=None,
        stream=False
    )

    simplified_text = response.choices[0].message.content.strip()
    st.write("[DEBUG] Groq returned simplified text:", simplified_text)

    # Split on commas or newlines -> individual words
    keywords = [w.strip().lower() for w in re.split(r"[,\n]+", simplified_text) if w.strip()]
    return keywords[:max_keywords]

###############################################################################
# 2) signbsl.com Lookup
###############################################################################
def get_video_url(word, source="signstation"):
    """
    Checks signbsl.com for a .mp4 for 'word'. We add a 1-second delay to avoid
    hammering the site too rapidly.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"

    st.write(f"[DEBUG] Checking BSL for '{word}' => {video_url}")
    response = requests.head(video_url)
    st.write(f"[DEBUG] HTTP status for '{word}': {response.status_code}")
    time.sleep(1)

    return video_url if response.status_code == 200 else None

###############################################################################
# 3) Groq synonyms if direct sign not found
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    prompt = (
        f"We are working with British Sign Language (BSL). The user said '{original_word}', "
        "but it wasn't found on signbsl.com. Provide up to "
        f"{max_alternatives} synonyms in British English as a comma-separated list."
    )

    st.write(f"[DEBUG] Asking for synonyms of '{original_word}' from Groq...")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_completion_tokens=256,
        top_p=1,
        stop=None,
        stream=False
    )

    text_out = response.choices[0].message.content.strip()
    synonyms = [w.strip().lower() for w in re.split(r"[,\n]+", text_out) if w.strip()]
    st.write(f"[DEBUG] Synonyms for '{original_word}':", synonyms)

    return synonyms[:max_alternatives]

###############################################################################
# 4) Convert English text -> BSL video items
###############################################################################
def process_text_bsl(client, raw_text):
    """
    1) Simplify raw_text to BSL-friendly keywords,
    2) For each, check signbsl.com. If not found, synonyms,
    3) Return list of { "word": <display text>, "url": <mp4 link or None> }.
    """
    bsl_words = bsl_simplify_with_groq(client, raw_text)
    st.write("[DEBUG] BSL words:", bsl_words)

    results = []
    for word in bsl_words:
        url = get_video_url(word)
        if url:
            results.append({"word": word, "url": url})
        else:
            st.write(f"[DEBUG] No direct sign for '{word}'. Checking synonyms...")
            synonyms = get_bsl_alternatives_from_groq(client, word)
            found_alt = None
            used_synonym = None
            for alt in synonyms:
                alt_url = get_video_url(alt)
                if alt_url:
                    found_alt = alt_url
                    used_synonym = alt
                    break
            if found_alt:
                display_text = f"{word} (using '{used_synonym}')"
                results.append({"word": display_text, "url": found_alt})
            else:
                results.append({"word": f"{word} (no sign)", "url": None})
    return results

###############################################################################
# Callbacks for next/previous
###############################################################################
def next_word():
    idx = st.session_state.get("bsl_index", 0)
    if st.session_state["bsl_videos"] and idx < len(st.session_state["bsl_videos"]) - 1:
        st.session_state["bsl_index"] = idx + 1

def prev_word():
    idx = st.session_state.get("bsl_index", 0)
    if idx > 0:
        st.session_state["bsl_index"] = idx - 1

###############################################################################
# 5) Main Streamlit App
###############################################################################
def main():
    # Replace with your actual Groq API key
    api_key = ""
    client = Groq(api_key=api_key)

    st.set_page_config(page_title="BSL Video Carousel", layout="wide")

    # Initialize session state if needed
    if "bsl_videos" not in st.session_state:
        st.session_state["bsl_videos"] = []
    if "bsl_index" not in st.session_state:
        st.session_state["bsl_index"] = 0

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ("Camera Translation", "Speech Translation", "Text Translation"))

    # --- Camera (Placeholder) ---
    if page == "Camera Translation":
        st.header("Camera Translation (Placeholder)")
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            img = Image.open(camera_image)
            st.image(img, caption="Captured Image", use_column_width=True)
            st.success("Detected gesture: (placeholder)")

    # --- Speech (st.audio_input) ---
    elif page == "Speech Translation":
        st.header("Speech Translation")
        st.write("Record audio, transcribe with Groq, and produce BSL videos.")

        audio_file = st.audio_input("Record a voice message")
        if audio_file is not None:
            st.write("### Playback of your recording:")
            st.audio(audio_file)

            file_bytes = audio_file.read()
            st.write(f"**File size**: {len(file_bytes)} bytes")

            if st.button("Transcribe & Generate BSL Videos"):
                try:
                    with st.spinner("Transcribing with Groq..."):
                        transcription = client.audio.transcriptions.create(
                            file=(audio_file.name or "recorded.wav", file_bytes),
                            model="whisper-large-v3-turbo",
                            response_format="json",
                            language="en",
                        )
                    raw_text = transcription.text.lower().strip()
                    st.success("Transcription complete!")
                    st.write("### Recognized text:")
                    st.write(raw_text)

                    with st.spinner("Generating BSL video items..."):
                        videos = process_text_bsl(client, raw_text)

                    st.session_state["bsl_videos"] = videos
                    st.session_state["bsl_index"] = 0

                    st.success("Generated BSL video items. See below!")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Click 'Record a voice message' above to capture audio.")

    # --- Text Translation ---
    else:
        st.header("Text Translation")
        user_text = st.text_area("Enter your text here", placeholder="e.g., 'Hello, what is your name?'")
        if user_text.strip():
            if st.button("Convert to BSL Videos"):
                with st.spinner("Processing text..."):
                    videos = process_text_bsl(client, user_text)
                st.session_state["bsl_videos"] = videos
                st.session_state["bsl_index"] = 0
                st.success("Generated BSL video items.")

    # --- Video Carousel ---
    st.markdown("---")
    st.header("BSL Video Carousel")
    st.write("Use the buttons below to navigate through each word's BSL video.")

    videos = st.session_state["bsl_videos"]
    idx = st.session_state["bsl_index"]

    if not videos:
        st.info("No BSL videos generated yet. Use Speech or Text translation first.")
    else:
        current_item = videos[idx]
        word = current_item["word"]
        url = current_item["url"]

        st.write(f"**Word {idx+1} of {len(videos)}:** {word}")

        if url:
            unique_id = uuid.uuid4()
            final_url = f"{url}?nocache={idx}-{unique_id}"

            # Print the final URL for debugging
            st.write(f"**Final URL** for {word}: {final_url}")

            # Embed the video with <video> tag, height=300
            video_html = f"""
                <video width="auto" height="300" controls>
                  <source src="{final_url}" type="video/mp4">
                  Your browser does not support the video tag.
                </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        else:
            st.error(f"No BSL video available for '{word}'.")

        # Navigation: no rerun needed; callbacks auto refresh
        col1, col2 = st.columns(2)
        with col1:
            st.button("Previous Word", on_click=prev_word, disabled=(idx == 0))
        with col2:
            st.button("Next Word", on_click=next_word, disabled=(idx == len(videos)-1))

        st.write("Use the buttons above to step through each word's sign.")

if __name__ == "__main__":
    main()