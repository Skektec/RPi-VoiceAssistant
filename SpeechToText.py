from flask import Flask
from flask_socketio import SocketIO
from faster_whisper import WhisperModel
import numpy as np
import threading
import sounddevice as sd
import queue
import time
import os
import gc
import logging
from threading import Event

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["OMP_NUM_THREADS"] = "4" 

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

RATE = 16000
CHUNK = int(RATE * 2)
CHANNELS = 1
NOISE_THRESHOLD = 0.005

restart_event = Event()
shutdown_event = Event()
audio_queue = queue.Queue(maxsize=100)

def initialize_model(model_size="base", compute_type="int8"):
    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=4,  
            num_workers=2   
        )
        logger.info(f"Model loaded: {model_size} using CPU with {compute_type} precision")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = WhisperModel(
            "tiny",  
            device="cpu", 
            compute_type=compute_type
        )
        logger.info("Fallback model loaded")
        return model

model = initialize_model()

def is_speech(audio_data, threshold=NOISE_THRESHOLD):
    energy = np.abs(audio_data).mean()
    return energy > threshold

def audio_callback(indata, frames, time, status):
    if shutdown_event.is_set():
        return
        
    if status:
        logger.warning(f"Status: {status}")
        return
    
    try:
        audio_data = np.copy(indata[:, 0])
        
        try:
            audio_queue.put(audio_data, block=True, timeout=0.1)
        except queue.Full:
            try:
                for _ in range(audio_queue.qsize() // 2):
                    audio_queue.get_nowait()
                audio_queue.put(audio_data, block=False)
                logger.warning("Queue was full - cleared half to recover")
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in audio capture: {e}")

def process_audio():
    buffer = []
    last_transcription_time = time.time()
    silence_counter = 0
    consecutive_errors = 0
    heartbeat_time = time.time()
    
    while not shutdown_event.is_set():
        try:
            if restart_event.is_set():
                logger.info("Restarting audio processing")
                buffer = []
                silence_counter = 0
                consecutive_errors = 0
                
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except:
                        pass
                
                restart_event.clear()
                socketio.emit("status", {"data": "System restarted"})
            
            current_time = time.time()
            if current_time - heartbeat_time > 10:
                heartbeat_time = current_time
                logger.info("Audio processing heartbeat")
                
            try:
                audio_data = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if is_speech(audio_data):
                buffer.append(audio_data)
                silence_counter = 0
            else:
                silence_counter += 1
            
            current_time = time.time()
            buffer_duration = len(buffer) * (CHUNK / RATE)
            time_since_last = current_time - last_transcription_time
            
            if (silence_counter >= 3 and len(buffer) > 0) or \
               (buffer_duration > 10) or \
               (len(buffer) > 0 and time_since_last > 5):
                
                if len(buffer) > 0:
                    audio_segment = np.concatenate(buffer)
                    
                    if len(audio_segment) > RATE * 30: 
                        logger.warning("Buffer too large, trimming to 30 seconds")
                        audio_segment = audio_segment[:RATE * 30]
                    
                    max_val = np.max(np.abs(audio_segment))
                    if max_val > 0:
                        audio_segment = audio_segment / max_val
                    
                    audio_float32 = audio_segment.astype(np.float32)
                    
                    try:
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
                            logger.info(f"Recognized: {text}")
                            socketio.emit("text", {"data": text})
                            consecutive_errors = 0 
                    except Exception as e:
                        logger.error(f"Transcription error: {e}")
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            try:
                                logger.warning("Multiple errors, attempting to reload model")
                                global model
                                del model
                                gc.collect()
                                model = initialize_model()
                                consecutive_errors = 0
                                socketio.emit("status", {"data": "Model reloaded after errors"})
                            except Exception as reload_error:
                                logger.error(f"Failed to reload model: {reload_error}")
                    
                    buffer = [] 
                    last_transcription_time = current_time
            
        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
            time.sleep(1) 

def monitor_thread():
    """Monitor the processing thread and restart if necessary"""
    last_queue_size = 0
    stuck_counter = 0
    
    while not shutdown_event.is_set():
        current_size = audio_queue.qsize()
        
        if current_size > 5 and current_size == last_queue_size:
            stuck_counter += 1
            if stuck_counter >= 5:
                logger.warning("System appears stuck - triggering restart")
                restart_event.set()
                stuck_counter = 0
        else:
            stuck_counter = 0
            
        last_queue_size = current_size
        time.sleep(2)

def start_audio_stream():
    audio_stream = None
    process_thread = None
    monitor_thread_handle = None
    
    try:

        monitor_thread_handle = threading.Thread(target=monitor_thread, daemon=True)
        monitor_thread_handle.start()
 
        process_thread = threading.Thread(target=process_audio, daemon=True)
        process_thread.start()
        
        while not shutdown_event.is_set():
            try:
                if audio_stream is None or not audio_stream.active:
                    if audio_stream is not None:
                        audio_stream.close()
                    
                    logger.info("Starting audio stream")
                    audio_stream = sd.InputStream(
                        channels=CHANNELS,
                        samplerate=RATE,
                        blocksize=CHUNK,
                        callback=audio_callback,
                        dtype='float32'
                    )
                    audio_stream.start()
                    socketio.emit("status", {"data": "Audio stream started"})
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Audio stream error: {e}")
                if audio_stream is not None:
                    try:
                        audio_stream.close()
                    except:
                        pass
                    audio_stream = None
                time.sleep(2) 
                
    except Exception as e:
        logger.error(f"Fatal error in audio stream: {e}")
    finally:
        if audio_stream is not None:
            audio_stream.close()

@app.route('/')
def index():
    return "Faster-Whisper streaming server is running"

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on("restart")
def handle_restart():
    logger.info("Restart requested by client")
    restart_event.set()
    return {"status": "success", "message": "Restart initiated"}

@socketio.on("set_threshold")
def handle_threshold(data):
    global NOISE_THRESHOLD
    try:
        new_threshold = float(data['threshold'])
        if 0 < new_threshold < 1:
            NOISE_THRESHOLD = new_threshold
            logger.info(f"Noise threshold set to {NOISE_THRESHOLD}")
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
            
        logger.info(f"Changing model to {model_size} with {compute_type} precision...")
        del model
        gc.collect()
        
        model = initialize_model(model_size, compute_type)
        return {"status": "success", "model": model_size, "compute_type": compute_type}
    except Exception as e:
        logger.error(f"Error changing model: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    stream_thread = threading.Thread(target=start_audio_stream, daemon=True)
    stream_thread.start()
    
    try:
        logger.info("Starting server")
        socketio.run(app, host='0.0.0.0', port=8765, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        shutdown_event.set()