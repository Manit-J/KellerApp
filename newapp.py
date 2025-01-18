import streamlit as st
import time
import requests
import re
import io
import numpy as np
import av
from PIL import Image
from groq import Groq

# streamlit-webrtc for mic capture
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

###############################################################################
# 1. BSL “Simplification” with Groq LLM
###############################################################################
def bsl_simplify_with_groq(client, text, max_keywords=15):
    """
    Calls Groq's LLM to convert the given English sentence
    into a BSL-friendly list of essential words, removing filler words
    while keeping question words, pronouns, etc.
    """
    example_input = "What is the name of your mother when she was happy?"
    example_output = "what name your mother when she was happy"

    prompt = f"""
You are an assistant that converts English sentences into a concise list of BSL-friendly keywords.
Keep question words (who, what, when, where, why, how), pronouns (she, he, you, we, they),
and time references (when, today, tomorrow), but remove filler words like "is", "are", "the" if possible.
Return them as a comma-separated list.

Example:
Input: "{example_input}"
Output: "{example_output}"

Now convert this sentence in the same style:
"{text.strip()}"
""".strip()

    st.write("**[DEBUG] BSL prompt:**", prompt)  # Debug prints in Streamlit

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or whichever text model your Groq account supports
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
    st.write("**[DEBUG] BSL simplified text from Groq:**", simplified_text)

    # Split on commas/newlines -> list of words
    keywords = [w.strip().lower() for w in re.split(r"[,\n]+", simplified_text) if w.strip()]
    return keywords[:max_keywords]

###############################################################################
# 2. SignBSL Lookup
###############################################################################
def get_video_url(word, source="signstation"):
    """
    HEAD-check signbsl.com to see if there's a .mp4 for 'word'.
    Adds a 1-second delay to avoid flooding the site with requests.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"

    st.write(f"[DEBUG] Checking BSL for '{word}' -> {video_url}")
    response = requests.head(video_url)
    st.write(f"[DEBUG] HTTP status for '{word}': {response.status_code}")

    time.sleep(1)  # delay to avoid rapid calls

    if response.status_code == 200:
        return video_url
    else:
        return None

###############################################################################
# 3. Groq synonyms if direct sign not found
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    """
    If 'original_word' is not found on signbsl.com, ask Groq for synonyms.
    """
    prompt = (
        f"We are working with British Sign Language. The user said '{original_word}', "
        "but it wasn't found on signbsl.com. Give up to "
        f"{max_alternatives} synonyms in British English that might exist there, "
        "as a comma-separated list."
    )

    st.write(f"[DEBUG] Asking LLM for synonyms of '{original_word}'")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful British English assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_completion_tokens=256,
        top_p=1,
        stop=None,
        stream=False
    )

    if not response.choices:
        st.write("[DEBUG] No synonyms returned by the LLM.")
        return []

    text_out = response.choices[0].message.content.strip()
    synonyms = [w.strip().lower() for w in re.split(r"[,\n]+", text_out) if w.strip()]
    st.write(f"[DEBUG] Synonyms for '{original_word}': {synonyms}")

    return synonyms[:max_alternatives]

###############################################################################
# 4. Convert text -> BSL
###############################################################################
def process_text_bsl(client, raw_text):
    """
    1) BSL-simplify with Groq
    2) For each keyword, check signbsl or synonyms
    3) Return dict of (keyword -> link or message)
    """
    bsl_words = bsl_simplify_with_groq(client, raw_text)
    st.write("**[DEBUG] BSL words:**", bsl_words)

    results = {}
    for word in bsl_words:
        url = get_video_url(word)
        if url:
            results[word] = url
        else:
            # synonyms
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
                results[word] = f"No sign for '{word}' - using '{used_synonym}': {found_alt}"
            else:
                results[word] = f"No BSL sign found for '{word}' or its synonyms."

    return results

###############################################################################
# 5. Use streamlit-webrtc to capture mic audio in the browser
###############################################################################
# We'll store raw audio frames in memory, then "Stop" to finalize
# and transcribe with Groq.

AUDIO_FRAMES = []  # store raw frames globally or in session_state

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    """
    Called every time an audio frame is received from the mic.
    We convert the frame to numpy, store it for later.
    """
    global AUDIO_FRAMES
    pcm = frame.to_ndarray()  # shape: (samples, channels)
    AUDIO_FRAMES.append(pcm)
    return frame

###############################################################################
# 6. Streamlit App
###############################################################################
def main():
    api_key = ""
    client = Groq(api_key=api_key)

    st.set_page_config(page_title="Keller App", page_icon="🧏", layout="wide")

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ("Camera Translation", "Speech Translation", "Text Translation"))

    if page == "Camera Translation":
        st.header("Camera Translation")
        st.write("Capture your ASL gesture using your device camera (placeholder).")
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_column_width=True)
            # Placeholder logic
            translation = "Hello (translated from ASL gesture)"
            st.success(f"Detected ASL: {translation}")

    elif page == "Speech Translation":
        st.header("Speech Translation")
        st.write("Record audio from your microphone, then convert to BSL videos via Groq + signbsl.com.")

        # WebRTC config (for local or HTTPS usage)
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Start the webrtc streamer
        webrtc_ctx = webrtc_streamer(
            key="speech-translation",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": False, "audio": True},
            audio_receiver_size=1024,
            video_receiver_size=0,
            on_audio_frame=audio_frame_callback
        )

        st.write("**Instructions**: Press 'Start' to allow mic access. Speak. Then press 'Stop' to finalize.")

        # We'll offer a button to "Transcribe"
        if st.button("Stop and Transcribe"):
            if len(AUDIO_FRAMES) == 0:
                st.warning("No audio frames recorded yet!")
            else:
                # Merge frames into WAV bytes
                st.info("Processing recorded audio...")

                # Convert stored frames (PCM) to a single numpy array
                audio_np = np.concatenate(AUDIO_FRAMES, axis=0)
                # Clear for next run
                AUDIO_FRAMES.clear()

                # We'll assume 16k or 48k sample rate from the browser.
                # streamlit-webrtc default is 48k stereo, let's do 16-bit WAV.
                import wave
                import io

                sample_rate = 48000  # typical from web browser mic
                channels = audio_np.shape[1] if audio_np.ndim > 1 else 1
                wav_bytes = io.BytesIO()
                with wave.open(wav_bytes, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    # Ensure data is int16
                    if audio_np.dtype != np.int16:
                        # convert float32 or float64 => int16
                        audio_np = (audio_np * 32767).astype(np.int16)
                    wf.writeframes(audio_np.tobytes())

                # Send to Groq
                with st.spinner("Transcribing with Groq..."):
                    transcription = client.audio.transcriptions.create(
                        file=("recorded.wav", wav_bytes.getvalue()),
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en",
                    )
                raw_text = transcription.text.lower().strip()
                st.success(f"Transcribed text: {raw_text}")

                # Now do BSL logic
                with st.spinner("Converting text to BSL..."):
                    results_dict = process_text_bsl(client, raw_text)

                # Display results
                st.write("## BSL Video Links / Messages:")
                for word, link_or_msg in results_dict.items():
                    # If it's a raw .mp4 link
                    if isinstance(link_or_msg, str) and link_or_msg.startswith("http"):
                        st.markdown(f"**{word}:** [Video Link]({link_or_msg})")
                    elif link_or_msg.startswith("No sign"):
                        # might contain 'using <alt>'
                        st.warning(f"**{word}:** {link_or_msg}")
                    else:
                        # direct .mp4 or fallback text
                        st.write(f"**{word}:** {link_or_msg}")

    else:  # Text Translation
        st.header("Text Translation")
        st.write("Type your text in English, and we'll convert it to BSL videos.")
        user_text = st.text_area("Enter your text here", placeholder="Type your sentence...")

        if user_text:
            with st.spinner("Converting text to BSL..."):
                client = Groq(api_key=api_key)
                results_dict = process_text_bsl(client, user_text)

            st.write("## BSL Video Links / Messages:")
            for word, link_or_msg in results_dict.items():
                if isinstance(link_or_msg, str) and link_or_msg.startswith("http"):
                    st.markdown(f"**{word}:** [Video Link]({link_or_msg})")
                elif link_or_msg.startswith("No sign"):
                    st.warning(f"**{word}:** {link_or_msg}")
                else:
                    st.write(f"**{word}:** {link_or_msg}")

if __name__ == "__main__":
    main()
