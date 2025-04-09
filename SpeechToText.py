from flask import Flask
from flask_socketio import SocketIO
from faster_whisper import WhisperModel
import numpy as np
import threading
import sounddevice as sd
import queue
import time
import os

os.environ["OMP_NUM_THREADS"] = "4"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

RATE = 16000
CHUNK = int(RATE * 2)
CHANNELS = 1
NOISE_THRESHOLD = 0.005

audio_queue = queue.Queue()
silence_counter = 0
MAX_SILENCE_COUNT = 3

print("Loading Faster-Whisper model...")
compute_type = "int8"
model_size = "base"

try:
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=2
    )
    print(f"Model loaded using CPU with {compute_type} precision")
except Exception as e:
    print(f"Error loading optimized model: {e}")
    model = WhisperModel(
        "tiny",
        device="cpu", 
        compute_type=compute_type
    )
    print("Fallback model loaded")

def is_speech(audio_data, threshold=NOISE_THRESHOLD):
    energy = np.abs(audio_data).mean()
    return energy > threshold

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
        return
    
    try:
        audio_data = np.copy(indata[:, 0])
        audio_queue.put(audio_data)
    except Exception as e:
        print(f"Error in audio capture: {e}")

def process_audio():
    global silence_counter
    buffer = []
    last_transcription_time = time.time()
    
    while True:
        try:
            audio_data = audio_queue.get(timeout=1)
            
            if is_speech(audio_data):
                buffer.append(audio_data)
                silence_counter = 0
            else:
                silence_counter += 1
            
            current_time = time.time()
            buffer_duration = len(buffer) * (CHUNK / RATE)
            time_since_last = current_time - last_transcription_time
            
            if (silence_counter >= MAX_SILENCE_COUNT and len(buffer) > 0) or \
               (buffer_duration > 10) or \
               (len(buffer) > 0 and time_since_last > 5):
                
                if len(buffer) > 0:
                    audio_segment = np.concatenate(buffer)
                    
                    audio_segment = audio_segment / np.max(np.abs(audio_segment)) if np.max(np.abs(audio_segment)) > 0 else audio_segment
                    
                    audio_int16 = (audio_segment * 32767).astype(np.int16)
                    
                    audio_float32 = audio_int16.astype(np.float32) / 32767
                    
                    segments, info = model.transcribe(
                        audio_float32, 
                        beam_size=3,         
                        language="en",       
                        vad_filter=True,    
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=False
                    )
                    
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text.strip())
                    
                    text = " ".join(text_parts).strip()
                    
                    if text:
                        print(f"Recognized: {text}")
                        socketio.emit("text", {"data": text})
                    
                    buffer = []  
                    last_transcription_time = current_time
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in audio processing: {e}")
            buffer = [] 

def start_audio_stream():
    try:
        with sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK,
            callback=audio_callback,
            dtype='float32'
        ):
            print("Audio stream started. Listening...")
            process_audio() 
    except Exception as e:
        print(f"Error in audio stream: {e}")

@app.route('/')
def index():
    return "Faster-Whisper streaming server is running"

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

@socketio.on("set_threshold")
def handle_threshold(data):
    global NOISE_THRESHOLD
    try:
        new_threshold = float(data['threshold'])
        if 0 < new_threshold < 1:
            NOISE_THRESHOLD = new_threshold
            return {"status": "success", "threshold": NOISE_THRESHOLD}
    except:
        pass
    return {"status": "error", "message": "Invalid threshold value"}

@socketio.on("set_model")
def handle_model_change(data):
    global model
    try:
        model_size = data.get('model', 'base')
        compute_type = data.get('compute_type', 'int8')
        
        if model_size not in ["tiny", "base", "small", "medium", "large"]:
            return {"status": "error", "message": "Invalid model size"}
        
        if compute_type not in ["int8", "float16", "float32"]:
            return {"status": "error", "message": "Invalid compute type"}
            
        print(f"Changing model to {model_size} with {compute_type} precision...")
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=2
        )
        return {"status": "success", "model": model_size, "compute_type": compute_type}
    except Exception as e:
        print(f"Error changing model: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    threading.Thread(target=start_audio_stream, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=8765, debug=False)