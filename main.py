import os
import sounddevice as sd
import whisper
import openwakeword
import piper
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv(dotenv_path='Data/.env')

class VoiceAssistant:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        
        try:
            wake_word_models = []
            
            valid_models = []
            for model_name in wake_word_models:
                try:
                    model_path = os.path.join(
                        os.path.dirname(openwakeword.__file__), 
                        'resources', 'models', model_name
                    )
                    if os.path.exists(model_path):
                        valid_models.append(model_path)
                except Exception as e:
                    print(f"Could not load model {model_name}: {e}")
            
            self.oww_model = openwakeword.Model(
                wakeword_models=valid_models
            )
            
            if not valid_models:
                print("WARNING: No wake word models found. Wake word detection will not work.")
        
        except Exception as e:
            print(f"Error initializing wake word model: {e}")
            print("Wake word detection will be disabled.")
            self.oww_model = None
        
        try:
            tts_model_path = "Data/en_US-arctic-medium.onnx"
            if not os.path.exists(tts_model_path):
                print(f"TTS model not found at {tts_model_path}")
                print("Please download the Piper TTS model")
                self.tts_model = None
            else:
                self.tts_model = piper.PiperVoice.load(tts_model_path)
        except Exception as e:
            print(f"Error initializing TTS model: {e}")
            self.tts_model = None
        
        self.mistral_client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        
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
        result = self.whisper_model.transcribe(audio_data)
        return result['text']

    def get_ai_response(self, user_command):
        with self.mistral_client as mistral:
            response = mistral.chat.complete(
                model="mistral-small-latest", 
                messages=[{
                    "role": "user", 
                    "content": user_command
                }]
            )
        
        return response.choices[0].message.content

    def speak_response(self, text):
        if self.tts_model is None:
            print(f"Cannot speak: {text}")
            return
        
        audio_bytes = self.tts_model.synthesize(text)
        
        sd.play(audio_bytes, self.sample_rate)
        sd.wait()

    def process_command(self):
        print("Listening for command...")
        
        recording = sd.rec(
            int(10 * self.sample_rate),
            samplerate=self.sample_rate, 
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()
        
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
