import streamlit as st
import time
import requests
import re
import io
import uuid
from PIL import Image
import numpy as np
from groq import Groq

# Set page configuration at the very top.
st.set_page_config(page_title="BSL Video Carousel", layout="wide")

###############################################################################
# Inject custom CSS for the carousel video only.
###############################################################################
st.markdown(
    """
    <style>
       /* Only target video elements inside the carousel container */
       video {
           height: 500px !important;
           width: 600px !important;
       }
    </style>
    """,
    unsafe_allow_html=True
)

###############################################################################
# 1) BSL “Simplification” with Groq
###############################################################################
def bsl_simplify_with_groq(client, text, max_keywords=20):
    """
    Converts an English sentence into a list of essential words for BSL.
    The LLM is instructed to return each word separately as a comma-separated list.
    """
    example_input = "What is your name?"
    example_output = "what, your, name"

    prompt = f"""
You are an assistant that converts English sentences into a list of essential words for British Sign Language (BSL).
Preserve question words (who, what, when, where, why, how), pronouns (I, you, she, he, we, they),
and time references (when, today, tomorrow). Remove only minimal filler words such as 'is', 'are', 'am', 'the', 'of'.

IMPORTANT:
1) Return each essential word separately. Do not merge multiple words.
2) Return your final answer as a comma-separated list.

For example:
Input: "{example_input}"
Output: "{example_output}"

Now convert this sentence:
"{text.strip()}"
""".strip()

    if st.session_state.get("debug", False):
        st.write("[DEBUG] BSL Simplify Prompt:", prompt)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Use your preferred Groq text model
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
    if st.session_state.get("debug", False):
        st.write("[DEBUG] Groq returned simplified text:", simplified_text)
    keywords = [w.strip().lower() for w in re.split(r"[,\n]+", simplified_text) if w.strip()]
    return keywords[:max_keywords]

###############################################################################
# 2) signbsl.com Lookup
###############################################################################
def get_video_url(word, source="signstation"):
    """
    Performs a HEAD request on signbsl.com for a .mp4 matching 'word'.
    Adds a 1-second delay to avoid overloading the site.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"
    if st.session_state.get("debug", False):
        st.write(f"[DEBUG] Checking BSL for '{word}' => {video_url}")
    response = requests.head(video_url)
    if st.session_state.get("debug", False):
        st.write(f"[DEBUG] HTTP status for '{word}':", response.status_code)
    time.sleep(1)
    return video_url if response.status_code == 200 else None

###############################################################################
# 3) Groq Synonyms if Direct Sign Not Found
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    prompt = (
        f"We are working with British Sign Language (BSL). The user said '{original_word}', "
        "but it wasn't found on signbsl.com. Provide up to "
        f"{max_alternatives} synonyms in British English as a comma-separated list."
    )
    if st.session_state.get("debug", False):
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
    if st.session_state.get("debug", False):
        st.write(f"[DEBUG] Synonyms for '{original_word}':", synonyms)
    return synonyms[:max_alternatives]

###############################################################################
# 4) Process English Text into BSL Video Items
###############################################################################
def process_text_bsl(client, raw_text):
    bsl_words = bsl_simplify_with_groq(client, raw_text)
    if st.session_state.get("debug", False):
        st.write("[DEBUG] BSL words:", bsl_words)
    results = []
    for word in bsl_words:
        url = get_video_url(word)
        if url:
            results.append({"word": word, "url": url})
        else:
            if st.session_state.get("debug", False):
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
    if st.session_state.get("debug", False):
        st.write("[DEBUG] Final BSL video items:", results)
    return results

###############################################################################
# 5) Navigation Callback Functions (using st.experimental_rerun)
###############################################################################
def next_word_and_rerun():
    if "bsl_videos" in st.session_state and st.session_state["bsl_videos"]:
        idx = st.session_state.get("bsl_index", 0)
        if idx < len(st.session_state["bsl_videos"]) - 1:
            st.session_state["bsl_index"] = idx + 1
    try:
        st.experimental_rerun()
    except Exception:
        pass

def prev_word_and_rerun():
    if "bsl_videos" in st.session_state and st.session_state["bsl_videos"]:
        idx = st.session_state.get("bsl_index", 0)
        if idx > 0:
            st.session_state["bsl_index"] = idx - 1
    try:
        st.experimental_rerun()
    except Exception:
        pass

###############################################################################
# 6) Main Streamlit App
###############################################################################
def main():
    # Set debug flag in session state (default off)
    if "debug" not in st.session_state:
        st.session_state["debug"] = False

    # Replace with your actual Groq API key.
    api_key = "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"
    client = Groq(api_key=api_key)

    # Initialize session state for BSL videos and index if needed.
    if "bsl_videos" not in st.session_state:
        st.session_state["bsl_videos"] = []
    if "bsl_index" not in st.session_state:
        st.session_state["bsl_index"] = 0

    # Debug toggle (default off)
    st.sidebar.checkbox("Show Debug Info", value=st.session_state["debug"], key="debug")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Camera Translation", "Speech Translation", "Text Translation"))

    # --- Camera Translation Page ---
    if page == "Camera Translation":
        st.header("Camera Translation")
        st.write("Capture your ASL gesture using your device camera (placeholder).")
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_column_width=True)
            st.success("Detected gesture: (placeholder)")

    # --- Speech Translation Page ---
    elif page == "Speech Translation":
        st.header("Speech Translation")
        st.write("Record a voice message, transcribe with Groq, and generate BSL video items.")
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
                    st.success("Generated BSL video items. See the carousel below!")
                except Exception as e:
                    st.error(f"Error during transcription/processing: {e}")
        else:
            st.info("Click 'Record a voice message' above to capture your audio.")

    # --- Text Translation Page ---
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

    # --- Video Carousel Section ---
    st.markdown("---")
    st.header("Video Carousel")
    st.write("Navigate through each word's BSL video:")

    videos = st.session_state.get("bsl_videos", [])
    idx = st.session_state.get("bsl_index", 0)

    if st.session_state.get("debug"):
        st.sidebar.write("[DEBUG] Current Index:", idx)
        st.sidebar.write("[DEBUG] BSL Videos:", videos)

    if not videos:
        st.info("No BSL videos generated yet. Please run Speech or Text Translation first.")
    else:
        current_item = videos[idx]
        word = current_item["word"]
        url = current_item["url"]

        st.write(f"**Word {idx+1} of {len(videos)}:** {word}")
        if url:
            unique_param = f"{idx}-{uuid.uuid4()}"
            final_url = f"{url}?nocache={unique_param}"
            if st.session_state.get("debug"):
                st.write("[DEBUG] Final video URL:", final_url)
            # Here we simply use st.video(); our custom CSS will only affect the carousel if needed.
            st.video(final_url, format="video/mp4")
        else:
            st.error(f"No BSL video available for '{word}'.")

        col_prev, col_next = st.columns(2)
        with col_prev:
            st.button("Previous Word", on_click=prev_word_and_rerun, disabled=(idx == 0))
        with col_next:
            st.button("Next Word", on_click=next_word_and_rerun, disabled=(idx == len(videos) - 1))

        st.write("Use the buttons above to navigate through the words.")
        st.markdown("""
        **Note**: If the same video appears repeatedly, try clearing your browser cache or using Incognito mode.
        """)

if __name__ == "__main__":
    main()
