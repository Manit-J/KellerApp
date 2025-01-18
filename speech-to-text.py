import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import wave
import os
from groq import Groq

class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder with Groq")

        self.recording = False
        self.audio_data = []
        self.samplerate = 16000
        self.filename = "recorded_audio.wav"

        # Groq API setup
        self.api_key = "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"  # Replace with your Groq API key
        self.client = Groq(api_key=self.api_key)

        # UI Elements
        self.record_button = tk.Button(root, text="Record", command=self.toggle_recording, width=20)
        self.record_button.pack(pady=20)

        self.transcribe_button = tk.Button(root, text="Send to Groq", command=self.transcribe_audio, state=tk.DISABLED, width=20)
        self.transcribe_button.pack(pady=20)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.record_button.config(text="Stop Recording")
        messagebox.showinfo("Recording", "Recording started. Click the button again to stop.")
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.audio_callback)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Record")
        self.stream.stop()
        self.stream.close()
        self.save_audio()
        self.transcribe_button.config(state=tk.NORMAL)
        messagebox.showinfo("Recording", "Recording stopped and saved as 'recorded_audio.wav'.")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_data.append(indata.copy())

    def save_audio(self):
        data = np.concatenate(self.audio_data, axis=0)
        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes((data * 32767).astype(np.int16).tobytes())

    def transcribe_audio(self):
        try:
            with open(self.filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=(self.filename, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="json",
                    language="en",
                )
                transcription_text = transcription.text

            messagebox.showinfo("Transcription", transcription_text)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()
