import tkinter as tk
import time
import sounddevice as sd
import numpy as np
import wave
import requests
import webbrowser
import re
from groq import Groq

###############################################################################
# 1. BSL “Simplification” – Remove filler words, e.g. “is”, “are”
###############################################################################
def bsl_simplify(text):
    """
    Naive removal of certain filler words to mimic (very roughly) BSL grammar.
    Example: "What is your name?" -> "what your name"
    """
    # Define words to remove (expand as needed)
    filler_words = {"is", "are", "am", "the", "a", "an", "to", "and", "do", "does", "did", "was", "were"}

    # Extract words, lowercase
    words = re.findall(r"[a-zA-Z']+", text.lower())

    # Filter out filler words
    filtered = [w for w in words if w not in filler_words]

    # Return them as a space-joined string (for debugging) or list
    return filtered  # returning as a list for direct usage

###############################################################################
# 2. Check if BSL video for a given word exists on signbsl.com
###############################################################################
def get_video_url(word, source="signstation"):
    """
    Returns a direct .mp4 URL for the given word if it exists on signbsl.com,
    or None if not found. Adds a 1-second delay after the HEAD request
    to reduce risk of looking like a bot.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"

    print(f"[DEBUG] Checking BSL for '{word}' -> {video_url}")
    response = requests.head(video_url)
    print(f"[DEBUG] HTTP status for '{word}': {response.status_code}")

    # Delay to help avoid 'bot-like' rapid requests
    time.sleep(1)

    if response.status_code == 200:
        return video_url
    else:
        return None

###############################################################################
# 3. Use Groq LLM to get British English synonyms
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    """
    Calls the Groq chat/completion endpoint with a text model
    (e.g., 'llama-3.3-70b-versatile') to get synonyms or alternative words
    specifically for British usage / BSL. Returns a list of strings (lowercase).
    """
    prompt = (
        f"We are working with British Sign Language (BSL). The user said '{original_word}', "
        "which isn't recognized by signbsl.com. Please provide up to "
        f"{max_alternatives} alternative English words (preferred in British usage) "
        "that might have a BSL sign on signbsl.com. Provide them in a comma-separated list."
    )

    print(f"[DEBUG] Asking LLM for synonyms of '{original_word}'")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or another text model on Groq
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_completion_tokens=512,
        top_p=1,
        stop=None,
        stream=False
    )

    if not response.choices:
        print("[DEBUG] No synonyms returned by the LLM.")
        return []

    # Use dot notation:
    text_out = response.choices[0].message.content.strip()

    possible_words = re.split(r"[,\n]+", text_out)
    possible_words = [w.strip().lower() for w in possible_words if w.strip()]

    print(f"[DEBUG] Synonyms for '{original_word}': {possible_words}")

    return possible_words[:max_alternatives]

###############################################################################
# 4. Main Tkinter Application
###############################################################################
class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BSL Finder with Groq")

        # Recording attributes
        self.recording = False
        self.audio_data = []
        self.samplerate = 16000
        self.filename = "recorded_audio.wav"

        # Groq API setup
        self.api_key = ""  # Replace with your valid key
        self.client = Groq(api_key=self.api_key)

        # --- GUI ELEMENTS ---

        # Single button to start/stop
        self.record_button = tk.Button(
            root, text="Record", command=self.toggle_recording, width=20
        )
        self.record_button.pack(pady=5)

        # Flashing label (not shown initially)
        self.recording_label = tk.Label(
            root, text="Recording...", fg="red", font=("Arial", 14, "bold")
        )
        # Do NOT pack/place it yet. We will only do so while recording.

        # We'll use a flag to control label flashing
        self.flashing = False

        # Processing label for during transcription & lookups
        self.processing_label = tk.Label(
            root, text="Processing...", fg="blue", font=("Arial", 14, "bold")
        )
        # Also hidden by default

        # Results frame
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=10)

    # -------------------------------------------------------------------------
    # RECORD/STOP LOGIC
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        """
        Single-button approach:
         - If not recording -> start
         - If currently recording -> stop, automatically transcribe
        """
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            self.transcribe_audio()  # no separate button

    def start_recording(self):
        self.recording = True
        self.audio_data = []

        # Change button text
        self.record_button.config(text="Stop")

        # Show the flashing "Recording..." label
        self.recording_label.place(x=100, y=50)  # position as needed
        self.flashing = True
        self.flash_label()

        print("[DEBUG] Recording started...")

        # Start capturing audio
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Record")

        # Stop flashing
        self.flashing = False
        self.recording_label.place_forget()

        # Stop audio stream
        self.stream.stop()
        self.stream.close()

        # Save to WAV
        self.save_audio()
        print(f"[DEBUG] Recording stopped and saved as '{self.filename}'.")

    def flash_label(self):
        """
        Toggles the text color of self.recording_label between red and black
        every 500ms while self.flashing is True.
        """
        if self.flashing:
            current_color = self.recording_label.cget("fg")
            next_color = "black" if current_color == "red" else "red"
            self.recording_label.config(fg=next_color)
            self.root.after(500, self.flash_label)

    def audio_callback(self, indata, frames, time_, status):
        if status:
            print("[DEBUG] Recording status:", status)
        self.audio_data.append(indata.copy())

    def save_audio(self):
        data = np.concatenate(self.audio_data, axis=0)
        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.samplerate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())

    # -------------------------------------------------------------------------
    # TRANSCRIPTION & BSL LOOKUP
    # -------------------------------------------------------------------------
    def transcribe_audio(self):
        """
        1) Show "Processing..." label
        2) Transcribe with Groq
        3) Apply naive BSL simplification (remove filler words)
        4) For each word, check signbsl.com or synonyms
        5) Hide "Processing..." label
        """
        # Show processing label
        self.processing_label.place(x=100, y=80)

        # Clear old results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        try:
            # --- 1. Transcribe with Groq ---
            with open(self.filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(self.filename, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="en",
                )

            # Force to lowercase & strip
            raw_text = transcription.text.lower().strip()
            print(f"[DEBUG] Raw transcription: {raw_text}")

            # --- 2. Simplify to approximate BSL structure ---
            # e.g. remove "is", "are", etc.
            bsl_words = bsl_simplify(raw_text)  
            print(f"[DEBUG] BSL-simplified words: {bsl_words}")

            # --- 3. Look up each word on signbsl.com or synonyms ---
            for word in bsl_words:
                print(f"[DEBUG] Checking word '{word}'...")
                url = get_video_url(word)

                if url:
                    # Found a direct BSL link
                    self._create_video_button(word, url)
                else:
                    print(f"[DEBUG] '{word}' not found. Getting synonyms from LLM...")
                    synonyms = get_bsl_alternatives_from_groq(self.client, word)

                    found_alternative_url = None
                    used_synonym = None

                    for alt in synonyms:
                        print(f"[DEBUG] Trying synonym '{alt}' for '{word}'")
                        alt_url = get_video_url(alt)
                        if alt_url:
                            found_alternative_url = alt_url
                            used_synonym = alt
                            break

                    if found_alternative_url:
                        display_text = f"No sign for '{word}' - using '{used_synonym}'"
                        self._create_video_button(display_text, found_alternative_url)
                    else:
                        msg = f"No BSL video for '{word}' or its synonyms."
                        tk.Label(self.results_frame, text=msg).pack(anchor="w", pady=2)

        except Exception as e:
            print("[DEBUG] Error:", e)
            tk.Label(self.results_frame, text=f"Error: {e}", fg="red").pack(anchor="w", pady=2)

        # Hide processing label
        self.processing_label.place_forget()

    def _create_video_button(self, label_text, url):
        btn = tk.Button(
            self.results_frame,
            text=f"Play BSL video for '{label_text}'",
            command=lambda link=url: self.open_in_browser(link),
            fg="blue", cursor="hand2"
        )
        btn.pack(anchor="w", pady=2)

    def open_in_browser(self, url):
        webbrowser.open(url)

###############################################################################
# 5. MAIN
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x300")  # Adjust size as desired

    app = AudioRecorderApp(root)
    root.mainloop()
