from flask import Flask, jsonify
from RealtimeSTT import AudioToTextRecorder
import threading

app = Flask(__name__)
latest_text = ""

def process_text(text):
    global latest_text
    latest_text = text

@app.route('/get_text', methods=['GET'])
def get_text():
    return jsonify({"text": latest_text})

def start_recorder():
    recorder = AudioToTextRecorder()
    while True:
        recorder.text(process_text)

if __name__ == '__main__':
    threading.Thread(target=start_recorder, daemon=True).start()
    app.run(host='localhost', port=8765)