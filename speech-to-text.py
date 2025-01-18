import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import wave
import requests
import webbrowser
import re
from groq import Groq

###############################################################################
# 1. Check if BSL video for a given word exists on signbsl.com
###############################################################################
def get_video_url(word, source="signstation"):
    """
    Returns a direct .mp4 URL for the given word if it exists on signbsl.com,
    or None if not found.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"

    print(f"[DEBUG] Checking BSL for '{word}' -> {video_url}")
    response = requests.head(video_url)
    print(f"[DEBUG] HTTP status for '{word}': {response.status_code}")

    if response.status_code == 200:
        return video_url
    else:
        return None

###############################################################################
# 2. Get British English synonyms from Groq (LLM) and print them
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    """
    Calls the Groq chat/completion endpoint with a text model (e.g., 'llama-3.3-70b-versatile')
    to get synonyms or alternative words specifically for British usage / BSL.
    Returns a list of possible word strings (lowercase) and prints them for debugging.

    This now uses dot notation (response.choices[0].message.content)
    to avoid the 'Choice' object is not subscriptable error.
    """
    prompt = (
        f"We are working with British Sign Language (BSL). The user said '{original_word}', "
        "which isn't recognized by signbsl.com. Please provide up to "
        f"{max_alternatives} alternative English words (preferred in British usage) "
        "that might have a BSL sign on signbsl.com. Provide them in a comma-separated list."
    )

    print(f"[DEBUG] Asking LLM for synonyms of '{original_word}'")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or whichever text model Groq provides
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

    # -- FIXED: use dot notation (message.content) instead of subscript:
    text_out = response.choices[0].message.content.strip()

    # Parse synonyms (assuming comma-separated)
    possible_words = re.split(r"[,\n]+", text_out)
    possible_words = [w.strip().lower() for w in possible_words if w.strip()]

    # Print synonyms to console
    print(f"[DEBUG] Synonyms for '{original_word}': {possible_words}")

    return possible_words[:max_alternatives]

###############################################################################
# 3. Main Tkinter Application
###############################################################################
class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio to BSL Video Finder with Groq")

        # Recording attributes
        self.recording = False
        self.audio_data = []
        self.samplerate = 16000
        self.filename = "recorded_audio.wav"

        # Groq API setup
        # WARNING: Storing an API key in code is not recommended.
        # For demonstration only:
        self.api_key = " "
        self.client = Groq(api_key=self.api_key)

        # --- GUI: Buttons & Results Frame ---
        self.record_button = tk.Button(
            root, text="Record", command=self.toggle_recording, width=20
        )
        self.record_button.pack(pady=5)

        self.transcribe_button = tk.Button(
            root, text="Send to Groq",
            command=self.transcribe_audio,
            state=tk.DISABLED,
            width=20
        )
        self.transcribe_button.pack(pady=5)

        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=10)

    # -------------------------------------------------------------------------
    # Recording Logic
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_button.config(text="Stop Recording")
        messagebox.showinfo("Recording", "Recording started. Click again to stop.")
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Record")
        self.stream.stop()
        self.stream.close()
        self.save_audio()
        self.transcribe_button.config(state=tk.NORMAL)
        messagebox.showinfo("Recording", f"Recording saved as '{self.filename}'.")

    def audio_callback(self, indata, frames, time, status):
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
    # Transcription + Word Lookup + Synonyms
    # -------------------------------------------------------------------------
    def transcribe_audio(self):
        """
        1) Transcribes the recorded WAV file using Groq's Whisper-like endpoint.
           We force the text to lowercase.
        2) Removes punctuation and splits words.
        3) For each word, checks signbsl.com. If not found, calls the LLM
           'llama-3.3-70b-versatile' to suggest up to 5 synonyms in British usage.
        4) Displays clickable links or a label if none is found.
        5) Prints debug statements to console, including synonyms.
        """
        try:
            # --- 1. Transcribe ---
            with open(self.filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(self.filename, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="en",
                )

            transcription_text = transcription.text.lower().strip()
            print(f"[DEBUG] Raw transcription: {transcription_text}")

            # --- 2. Clean out punctuation. Example: 'washroom.' -> 'washroom'
            words = re.findall(r"[a-zA-Z']+", transcription_text)
            print(f"[DEBUG] Cleaned word list: {words}")

            # Show a popup with the raw transcription
            messagebox.showinfo("Transcription", f"Groq Transcription:\n{transcription_text}")

            # Clear old results from results_frame
            for widget in self.results_frame.winfo_children():
                widget.destroy()

            # --- 3. For each word, try signbsl. If not found, get synonyms. ---
            for word in words:
                if not word:
                    continue

                print(f"[DEBUG] Looking for BSL sign of '{word}'")
                url = get_video_url(word)

                if url:
                    # Found a direct BSL link
                    self._create_video_button(word, url)
                else:
                    print(f"[DEBUG] '{word}' not found. Getting synonyms from LLM...")
                    synonyms = get_bsl_alternatives_from_groq(self.client, word)

                    # Try each synonym
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
                        # Found a workable synonym
                        display_text = f"No sign for '{word}' - using '{used_synonym}'"
                        self._create_video_button(display_text, found_alternative_url)
                    else:
                        # None of the synonyms worked
                        lbl = tk.Label(
                            self.results_frame,
                            text=f"No BSL video for '{word}' or its synonyms."
                        )
                        lbl.pack(anchor="w", pady=2)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # -------------------------------------------------------------------------
    # Utility to create a clickable video button
    # -------------------------------------------------------------------------
    def _create_video_button(self, label_text, url):
        btn = tk.Button(
            self.results_frame,
            text=f"Play BSL video for '{label_text}'",
            command=lambda link=url: self.open_in_browser(link),
            fg="blue", cursor="hand2"
        )
        btn.pack(anchor="w", pady=2)

    # -------------------------------------------------------------------------
    # Utility to open a URL in the default browser
    # -------------------------------------------------------------------------
    def open_in_browser(self, url):
        webbrowser.open(url)

###############################################################################
# 4. Entry Point
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()
