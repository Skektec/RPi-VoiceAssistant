from flask import Flask
from flask_socketio import SocketIO
import whisper
import numpy as np
import threading
import sounddevice as sd
import queue

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = whisper.load_model("tiny")

RATE = 16000
CHUNK = int(RATE * 1) 
CHANNELS = 1
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
        return
    
    try:
        audio_data = np.copy(indata[:, 0])
        energy = np.abs(audio_data).mean()
        
        if 0.01 <= energy <= 0.5:
            audio_queue.put(audio_data)
    
    except Exception as e:
        print(f"Error in audio callback: {e}")

def process_audio():
    while True:
        try:
            audio_data = []
            try:
                while len(audio_data) < 2:
                    chunk = audio_queue.get(timeout=1.0)
                    audio_data.append(chunk)
            except queue.Empty:
                continue

            audio_data = np.concatenate(audio_data)
            audio_data = np.clip(audio_data, -1, 1)
            
            result = model.transcribe(
                audio_data,
                language='en',  
                fp16=False,    
                temperature=0.0
            )
            
            text = result["text"].strip()
            
            if text and not text.isspace() and len(text) > 3:
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
            process_audio()
    except Exception as e:
        print(f"Error in audio stream: {e}")

if __name__ == '__main__':
    print("Loading Whisper model...")
    threading.Thread(target=start_audio_stream, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=8765)