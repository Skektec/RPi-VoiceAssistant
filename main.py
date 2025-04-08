import os
import numpy as np
import sounddevice as sd
import whisper
import openwakeword
import piper
from dotenv import load_dotenv
from mistralai import Mistral, UserMessage, SystemMessage, AssistantMessage

print("Looking for .env file in:", os.path.abspath('Data/.env'))
print("Current working directory:", os.getcwd())

load_dotenv(dotenv_path='Data/.env')

print("Loaded environment variables:")
for key, value in os.environ.items():
    if 'API' in key or 'SECRET' in key:
        print(f"{key}: {'*' * len(value)}")

class VoiceAssistant:
    def __init__(self):
        self.whisper_model = whisper.load_model("tiny")
        
        try:
            potential_model_paths = [
                os.path.join(os.path.dirname(openwakeword.__file__), 'resources', 'models', 'hey_jarvis.tflite'),
                os.path.join(os.getcwd(), 'Data', 'hey_jarvis.tflite'),
                os.path.join(os.path.expanduser('~'), 'RPi-VoiceAssistant', 'Data', 'hey_jarvis.tflite')
            ]
            
            valid_models = [path for path in potential_model_paths if os.path.exists(path)]
            
            if valid_models:
                print(f"Found wake word models: {valid_models}")
                self.oww_model = openwakeword.Model(wakeword_models=valid_models)
            else:
                print("WARNING: No wake word models found. Wake word detection will be disabled.")
                self.oww_model = None
        
        except Exception as e:
            print(f"Error initializing wake word model: {e}")
            self.oww_model = None
        
        try:
            tts_model_paths = [
                "Data/en_US-arctic-medium.onnx",
                os.path.join(os.getcwd(), 'Data', 'en_US-arctic-medium.onnx'),
                os.path.join(os.path.expanduser('~'), 'RPi-VoiceAssistant', 'Data', 'en_US-arctic-medium.onnx')
            ]
            
            tts_model_path = next((path for path in tts_model_paths if os.path.exists(path)), None)
            
            if tts_model_path:
                print(f"Using TTS model: {tts_model_path}")
                self.tts_model = piper.PiperVoice.load(tts_model_path)
            else:
                print("ERROR: No TTS model found. TTS will be disabled.")
                self.tts_model = None
        
        except Exception as e:
            print(f"Error initializing TTS model: {e}")
            self.tts_model = None
        
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            print("ERROR: No Mistral API key found. Please set MISTRAL_API_KEY in Data/.env")
            self.mistral_client = None
        else:
            try:
                self.mistral_client = Mistral(api_key=api_key)
                print("Mistral AI client initialized successfully")
            except Exception as e:
                print(f"Error initializing Mistral client: {e}")
                self.mistral_client = None
        
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'float32'

    def listen_for_wakeword(self):
        if self.oww_model is None:
            print("Wake word detection is disabled. Proceeding to command.")
            return self.process_command()
        
        stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=self.channels,
            dtype=self.dtype
        )
        stream.start()
        
        while True:
            audio_chunk, _ = stream.read(4000)
            prediction = self.oww_model.predict(audio_chunk)
            
            if prediction > 0.5:
                print("Wake word detected!")
                return self.process_command()

    def transcribe_audio(self, audio_data):
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        max_audio_length = 20 * self.sample_rate
        if audio_data.size > max_audio_length:
            print(f"Audio too long. Trimming to {max_audio_length} samples.")
            audio_data = audio_data[:max_audio_length]
        
        try:
            result = self.whisper_model.transcribe(
                audio_data, 
                fp16=False
            )
            return result['text']
        except Exception as e:
            print(f"Transcription error: {e}")
            return "Could not transcribe audio"

    def get_ai_response(self, user_command):
        if self.mistral_client is None:
            print("Mistral AI client not initialized. Cannot get response.")
            return "I'm unable to process your request due to a configuration error."
        
        try:
            response = self.mistral_client.chat.complete(
                model="mistral-small-latest", 
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_command}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return "I encountered an error while processing your request."

    def speak_response(self, text):
        if self.tts_model is None:
            print(f"Cannot speak: {text}")
            return
        
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            wav_filename = temp_wav.name
        
        try:
            self.tts_model.synthesize(text, wav_filename)
            
            audio_data, sample_rate = sf.read(wav_filename)
            
            sd.play(audio_data, sample_rate)
            sd.wait()
        
        except Exception as e:
            print(f"Error speaking response: {e}")
        
        finally:
            import os
            try:
                os.unlink(wav_filename)
            except Exception:
                pass

    def process_command(self):
        print("Listening for command...")
        
        recording = sd.rec(
            int(10 * self.sample_rate),
            samplerate=self.sample_rate, 
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()
        
        recording = recording.flatten()
        
        user_command = self.transcribe_audio(recording)
        print(f"Transcribed: {user_command}")
        
        ai_response = self.get_ai_response(user_command)
        print(f"AI Response: {ai_response}")
        
        self.speak_response(ai_response)

    def run(self):
        print("Voice Assistant started. Say 'hey jarvis' to activate.")
        while True:
            self.listen_for_wakeword()

def main():
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
