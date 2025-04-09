from flask import Flask
from flask_socketio import SocketIO
import whisper
import numpy as np
import threading
import sounddevice as sd

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = whisper.load_model("tiny")

RATE = 16000
CHUNK = int(RATE * 3)
CHANNELS = 1

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
        return
    
    try:
        audio_data = np.copy(indata[:, 0])
        if np.abs(audio_data).mean() < 0.01: 
            return
            
        audio_data = np.clip(audio_data, -1, 1)
        
        result = model.transcribe(audio_data)
        text = result["text"].strip()
        
        if text:
            print(f"Recognized: {text}")
            socketio.emit("text", {"data": text})
    
    except Exception as e:
        print(f"Error in audio processing: {e}")

def start_audio_stream():
    try:
        with sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            callback=audio_callback
        ):
            print("Audio stream started. Listening...")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"Error in audio stream: {e}")

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

if __name__ == '__main__':
    print("Loading Whisper model...")
    threading.Thread(target=start_audio_stream, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=8765)