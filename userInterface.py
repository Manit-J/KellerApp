import streamlit as st
import time
import requests
import re
import io
import uuid
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from collections import deque

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
# Mediapipe Setup for Real-Time Gesture Detection (with Letter Detection)
###############################################################################
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# For wave detection, track wrist positions
wrist_positions = deque(maxlen=20)

def detect_letter(hand_landmarks):
    """
    A VERY basic, partial rule-based approach to detect some letters (A, B, C, D, E).
    For a full alphabet, you'd define more conditions or train a custom model.
    Returns one of {"A", "B", "C", "D", "E"} or None if no match.
    """
    # Landmarks for fingertips and MCP joints
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    index_mcp = hand_landmarks.landmark[5]
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]

    # A naive approach: consider a finger "extended" if its tip is above (smaller y) its MCP joint.
    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_mcp.y
    ring_extended = ring_tip.y < ring_mcp.y
    pinky_extended = pinky_tip.y < pinky_mcp.y

    # For letter "A": All fingers are curled; thumb stays near index base.
    all_fingers_curled = (not index_extended and not middle_extended and not ring_extended and not pinky_extended)
    thumb_near_index_base = abs(thumb_tip.y - index_mcp.y) < 0.05
    if all_fingers_curled and (not thumb_near_index_base):
        return "A"

    # For letter "B": All four fingers extended.
    if index_extended and middle_extended and ring_extended and pinky_extended:
        return "B"

    # For letter "C": As an approximation, if index and middle are extended but ring and pinky are not.
    if index_extended and middle_extended and (not ring_extended) and (not pinky_extended):
        return "C"

    # For letter "D": Index finger extended; other fingers curled.
    if index_extended and (not middle_extended) and (not ring_extended) and (not pinky_extended):
        # A rough check for thumb position (naively if thumb tip is to the left of index MCP for a right-hand)
        if thumb_tip.x < index_mcp.x:
            return "D"

    # For letter "E": All fingers curled but thumb is not close to the index base.
    if all_fingers_curled and thumb_near_index_base:
        return "E"

    return None

def recognize_gesture(hand_landmarks, handedness, sequence_state):
    """
    1) Try to detect a letter (A, B, C, D, or E) using detect_letter().
    2) If no letter is detected, fallback to the existing gestures:
       "How" (thumb and fingers curved), "You" (index pointing),
       and a wave (detected as "Hello!").
    3) Return the detected letter, gesture, or "Unknown".
    """
    # First, try to detect a letter.
    letter = detect_letter(hand_landmarks)
    if letter is not None:
        return letter

    # Fallback to existing gesture detection.
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Update wrist positions for wave detection.
    wrist_positions.append(wrist.x)

    # "How" gesture: thumb and all fingers curved.
    thumb_curved = thumb_tip.y > wrist.y and thumb_tip.x > index_tip.x
    fingers_curved = all(tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    # "You" gesture: index finger pointing.
    is_pointing = (index_tip.y < middle_tip.y and
                   index_tip.y < ring_tip.y and
                   index_tip.y < pinky_tip.y and
                   abs(index_tip.x - thumb_tip.x) > 0.1)

    # Detect wave for "Hello!"
    if len(wrist_positions) >= 5:
        direction_changes = 0
        total_movement = 0
        for i in range(1, len(wrist_positions)):
            movement = abs(wrist_positions[i] - wrist_positions[i - 1])
            total_movement += movement
            if i > 1 and (wrist_positions[i] - wrist_positions[i - 1]) * (wrist_positions[i - 1] - wrist_positions[i - 2]) < 0:
                direction_changes += 1
        if direction_changes >= 4 and total_movement >= 0.2:
            wrist_positions.clear()
            if sequence_state is None:
                return "Hello!"

    if thumb_curved and fingers_curved:
        return "How"
    elif is_pointing:
        return "You"
    return "Unknown"

def run_realtime_detection():
    """
    Runs a loop to capture video from your webcam,
    detects letters (A-E) or gestures ("How", "You", "Hello!") using Mediapipe,
    and shows the frames in Streamlit.
    Press the 'Stop Gesture Detection' button to end.
    """
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure it's connected and accessible.")
        return

    last_gesture = None
    last_gesture_time = 0.0
    display_duration = 3.0  # seconds to display the last recognized gesture

    gesture_sequence = deque(maxlen=5)
    sequence_state = None

    st.write("**Real-time gesture detection running...**")
    st.write("Click '**Stop Gesture Detection**' in the sidebar to quit.")

    if "stop_detection" not in st.session_state:
        st.session_state["stop_detection"] = False

    stop_btn = st.sidebar.button("Stop Gesture Detection")
    if stop_btn:
        st.session_state["stop_detection"] = True

    while cap.isOpened() and not st.session_state["stop_detection"]:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = recognize_gesture(hand_landmarks, handedness, sequence_state)
                if gesture != "Unknown":
                    if gesture != last_gesture:
                        last_gesture = gesture
                        last_gesture_time = time.time()

                    if gesture == "How":
                        sequence_state = "How"
                    elif sequence_state == "How" and gesture == "You":
                        last_gesture = "How are you?"
                        last_gesture_time = time.time()
                        sequence_state = None
                        gesture_sequence.clear()
                    elif gesture == "Hello!":
                        sequence_state = None

        if last_gesture and (time.time() - last_gesture_time < display_duration):
            cv2.putText(
                frame,
                last_gesture,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(display_frame, channels="RGB")
        time.sleep(0.03)

    cap.release()
    st.session_state["stop_detection"] = False
    st.write("Real-time gesture detection stopped.")

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

    client_response = client.chat.completions.create(
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
    simplified_text = client_response.choices[0].message.content.strip()
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

        # Button to start real-time detection
        if st.button("Start Real-time Gesture Detection"):
            run_realtime_detection()

        # # Existing single-image approach (unchanged)
        # st.write("Or capture a snapshot using your device camera:")
        # camera_image = st.camera_input("Take a picture")
        # if camera_image:
        #     image = Image.open(camera_image)
        #     st.image(image, caption="Captured Image", use_column_width=True)
        #     st.success("Detected gesture: (placeholder)")
        
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

    # --- Speech Translation Page ---
    elif page == "Speech Translation":
        st.header("Speech Translation")
        st.write("Record a voice message, transcribe with Groq, and generate BSL video items.")

        # Existing speech logic remains the same
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
    st.markdown("All videos have been retrieved from: [SignBSL](https://www.signbsl.com/)")

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
            st.video(final_url, format="video/mp4", loop=True, autoplay=True)
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
