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
# 1. BSL Simplification using Groq (LLM)
###############################################################################

def bsl_simplify_with_groq(client, text, max_keywords=20):
    """
    Uses Groq's chat/completion endpoint to convert the given English sentence
    into a BSL-style list of essential words. We preserve question words (who, what, why...),
    pronouns (she, he, you...), relevant time words (when, today...), 
    and other important context. We only remove minimal filler (like 'is', 'are', 'the', etc.).
    Returns a list of keywords in lowercase.
    """
    # Example-based prompt to guide the model
    example_input = "What is the name of your mother when she was happy?"
    example_output = "what name your mother when she was happy"
    
    prompt = f"""
You are an assistant that converts English sentences into a list of essential words for British Sign Language (BSL).
Preserve question words (who, what, when, where, why, how), pronouns (I, you, she, he, we, they),
and time references (when, today, tomorrow, etc.). 
Remove only minimal filler words such as 'is', 'are', 'am', 'the', 'of' where they do not affect the meaning.

For example:
Input: "{example_input}"
Output: "{example_output}"

Now, convert this sentence in the same style:
"{text.strip()}"

Return your answer as a comma-separated list of keywords.
""".strip()

    print(f"[DEBUG] Sending prompt for BSL simplification:\n{prompt}")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or whichever model your Groq account supports
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts English sentences into a concise list of BSL-friendly keywords."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_completion_tokens=128,
        top_p=1,
        stop=None,
        stream=False
    )

    simplified_text = response.choices[0].message.content.strip()
    print(f"[DEBUG] Groq returned simplified text: {simplified_text}")

    # Split on commas or newlines and remove extra whitespace
    keywords = [word.strip().lower() for word in re.split(r"[,\n]+", simplified_text) if word.strip()]
    return keywords[:max_keywords]


###############################################################################
# 2. Check if BSL video exists on signbsl.com (with delay)
###############################################################################
def get_video_url(word, source="signstation"):
    """
    Returns a direct .mp4 URL for the given word if it exists on signbsl.com,
    or None otherwise. A 1-second delay is added after the HEAD request.
    """
    base_url = "https://media.signbsl.com/videos/bsl"
    video_url = f"{base_url}/{source}/{word}.mp4"
    
    print(f"[DEBUG] Checking BSL for '{word}' -> {video_url}")
    response = requests.head(video_url)
    print(f"[DEBUG] HTTP status for '{word}': {response.status_code}")
    
    time.sleep(1)  # Delay to help avoid rapid requests
    
    if response.status_code == 200:
        return video_url
    else:
        return None

###############################################################################
# 3. Use Groq to get synonyms (if no direct sign exists)
###############################################################################
def get_bsl_alternatives_from_groq(client, original_word, max_alternatives=5):
    """
    Calls Groq's chat/completion endpoint asking for synonyms, specifically for
    British usage, for the original word (which might then have a BSL sign on signbsl.com).
    Returns a list (up to max_alternatives) of alternative words in lowercase.
    """
    prompt = (
        f"We are working with British Sign Language (BSL). The user said '{original_word}', "
        "but I did not find a sign for this word on signbsl.com. Please provide up to "
        f"{max_alternatives} alternative English words (preferably in British usage) "
        "that might have a BSL sign on signbsl.com. Return them as a comma-separated list."
    )
    
    print(f"[DEBUG] Asking LLM for synonyms of '{original_word}'")
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Adjust as needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
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
    
    text_out = response.choices[0].message.content.strip()
    possible_words = [w.strip().lower() for w in re.split(r"[,\n]+", text_out) if w.strip()]
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
        
        # Groq API setup; replace the API key with your valid key.
        self.api_key = ""  
        self.client = Groq(api_key=self.api_key)
        
        # --- GUI ELEMENTS ---
        # (A) Audio: Record/Stop button & flashing "Recording..." label.
        self.record_button = tk.Button(root, text="Record", command=self.toggle_recording, width=20)
        self.record_button.pack(pady=5)
        
        self.recording_label = tk.Label(root, text="Recording...", fg="red", font=("Arial", 14, "bold"))
        self.flashing = False  # Controls flashing; not shown until recording starts.
        
        # (B) Text: Entry field and Submit button.
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=10)
        
        self.text_label = tk.Label(self.input_frame, text="Type Sentence:")
        self.text_label.pack(side=tk.LEFT)
        
        self.text_entry = tk.Entry(self.input_frame, width=30)
        self.text_entry.pack(side=tk.LEFT, padx=5)
        
        self.submit_button = tk.Button(self.input_frame, text="Submit", command=self.on_submit_text)
        self.submit_button.pack(side=tk.LEFT)
        
        # (C) Processing label.
        self.processing_label = tk.Label(root, text="Processing...", fg="blue", font=("Arial", 14, "bold"))
        
        # (D) Results frame
        self.results_frame = tk.Frame(root)
        self.results_frame.pack(pady=10)
    
    # -------------------------------------------------------------------------
    # Audio Recording Logic
    # -------------------------------------------------------------------------
    def toggle_recording(self):
        """
        Single-button approach:
         - If not recording: start recording.
         - If recording: stop recording and process the recorded audio.
        """
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            self.process_recorded_audio()  # Automatically process after stopping.
    
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_button.config(text="Stop")
        
        # Show and start flashing the "Recording..." label.
        self.recording_label.place(x=100, y=50)
        self.flashing = True
        self.flash_label()
        
        print("[DEBUG] Recording started...")
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Record")
        self.flashing = False
        self.recording_label.place_forget()
        
        self.stream.stop()
        self.stream.close()
        self.save_audio()
        print(f"[DEBUG] Recording stopped and saved as '{self.filename}'.")
    
    def flash_label(self):
        """
        Toggles the recording label color every 500ms.
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
    # Handling Typed Input
    # -------------------------------------------------------------------------
    def on_submit_text(self):
        """
        Called when the user clicks "Submit" after typing text.
        Processes the typed sentence for BSL conversion.
        """
        typed_text = self.text_entry.get().strip()
        if not typed_text:
            print("[DEBUG] No text entered.")
            return
        
        print(f"[DEBUG] User typed: {typed_text}")
        # Clear old results.
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.processing_label.place(x=100, y=80)
        self.process_text_for_bsl(typed_text)
        self.processing_label.place_forget()
    
    # -------------------------------------------------------------------------
    # Processing Recorded Audio
    # -------------------------------------------------------------------------
    def process_recorded_audio(self):
        """
        Called after audio recording stops. Transcribes the audio via Groq
        and then processes the resulting text for BSL conversion.
        """
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        self.processing_label.place(x=100, y=80)
        
        try:
            with open(self.filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(self.filename, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="en",
                )
            raw_text = transcription.text.lower().strip()
            print(f"[DEBUG] Raw transcription: {raw_text}")
            
            self.process_text_for_bsl(raw_text)
        
        except Exception as e:
            print("[DEBUG] Error transcribing audio:", e)
            tk.Label(self.results_frame, text=f"Error: {e}", fg="red").pack(anchor="w", pady=2)
        
        self.processing_label.place_forget()
    
    # -------------------------------------------------------------------------
    # Main BSL Processing Function (shared by typed or audio input)
    # -------------------------------------------------------------------------
    def process_text_for_bsl(self, raw_text):
        """
        Converts the raw English text into a simplified BSL keyword list using Groq,
        then looks up each keyword on signbsl.com (or fetches synonyms if necessary).
        """
        # Use Groq to simplify the sentence.
        bsl_words = bsl_simplify_with_groq(self.client, raw_text)
        print(f"[DEBUG] BSL-simplified keywords: {bsl_words}")
        
        for word in bsl_words:
            print(f"[DEBUG] Checking word '{word}'...")
            url = get_video_url(word)
            if url:
                self._create_video_button(word, url)
            else:
                print(f"[DEBUG] No direct sign for '{word}'. Fetching synonyms...")
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
                    msg = f"No BSL video found for '{word}' or its synonyms."
                    tk.Label(self.results_frame, text=msg).pack(anchor="w", pady=2)
    
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
# Main Entry Point
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("500x350")  # Adjust window size as needed
    app = AudioRecorderApp(root)
    root.mainloop()
