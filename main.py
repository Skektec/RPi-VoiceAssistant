import os
import sounddevice as sd
import whisper
import openwakeword
import piper
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class VoiceAssistant:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        
        openwakeword.load_models()
        self.oww_model = openwakeword.Model(
            wakeword_models=["hey_jarvis"]  
        )
        
        self.tts_model = piper.Model("Data/en_US-arctic-medium.onnx")
        
        self.mistral_client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
        
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = 'float32'

    def listen_for_wakeword(self):
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
        messages = [
            ChatMessage(role="user", content=user_command)
        ]
        
        chat_response = self.mistral_client.chat(
            model="mistral-tiny",
            messages=messages
        )
        
        return chat_response.choices[0].message.content

    def speak_response(self, text):
        audio = self.tts_model.synthesize(text)
        sd.play(audio, self.sample_rate)
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
