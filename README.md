# RPi Voice Assistant with Mistral AI

## Overview

A voice assistant for Raspberry Pi 5 using:

- Whisper for speech recognition
- OpenWakeWord for wake word detection
- Piper for text-to-speech
- Mistral AI for conversational AI

## Prerequisites

- Raspberry Pi 5 (This is what I tested on, Piper was built for Raspberry Pi 4 but I haven't tried.)
- Python 3.8+
- Mistral AI API Key

## Setup

1. Clone the repository
2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set Mistral AI API Key in Data/.env:

   ```bash
   MISTRAL_API_KEY='your_api_key_here'
   ```

5. Download Wake Word and TTS Models:
   - Train/download OpenWakeWord model for your wake word. "Hey jarvis" is built in and what I used.
   - Download Piper TTS model

## Running the Assistant

```bash
python main.py
```

## Customization

- Modify wake word in `main.py`
- Change Mistral AI model
- Adjust audio parameters

## Troubleshooting

- Ensure all dependencies are installed
- Check microphone permissions
- Verify API key
