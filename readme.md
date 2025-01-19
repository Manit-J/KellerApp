# KellerApp

A Streamlit application that translates English text, speech, and camera gestures into British Sign Language (BSL) videos using GROQ AI.

## Features

- **Camera Translation**: Real-time gesture detection using MediaPipe for BSL signs
- **Speech Translation**: Convert spoken English to BSL videos
- **Text Translation**: Convert written English text to BSL videos
- **Video Carousel**: Navigate through BSL videos for each word
- **Intelligent Word Processing**: Uses Groq AI for text simplification and synonym finding

## Prerequisites
```bash
Python 3.9.0
```

```bash
pip install streamlit
pip install opencv-python
pip install mediapipe
pip install groq
pip install pillow
pip install requests
pip install numpy
```

## Setup

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Groq API key:
   ```python
   api_key = "your_groq_api_key_here"
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run userInterface.py
```

### Camera Translation
- Click "Start Real-time Gesture Detection" to begin
- Make BSL gestures in front of your camera
- The app will detect and display recognized signs
- Click "Stop Gesture Detection" in the sidebar to end

### Speech Translation
- Click the microphone icon to record your voice
- Speak clearly in English
- The app will transcribe your speech and convert it to BSL videos

### Text Translation
- Enter English text in the text area
- Click "Convert to BSL Videos" 
- Navigate through the BSL videos using Previous/Next buttons

## Technical Details

### Gesture Detection
- Uses MediaPipe Hands for real-time hand landmark detection
- Implements rule-based gesture recognition for BSL letters and common phrases
- Tracks wrist positions for dynamic gesture detection (e.g., waving)

### BSL Processing
- Simplifies English text into BSL-friendly keywords using Groq AI
- Searches signbsl.com for matching video content
- Falls back to synonym search when direct translations aren't available

### Video Carousel
- Displays BSL videos with word context
- Supports navigation between words
- Handles missing videos gracefully

## Credits

- BSL videos sourced from [SignBSL](https://www.signbsl.com/)
- Hand gesture detection powered by [MediaPipe](https://mediapipe.dev/)
- Natural language processing by [Groq](https://groq.com/)

