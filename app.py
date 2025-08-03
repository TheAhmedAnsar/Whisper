from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)

model = whisper.load_model("tiny")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)
        try:
            result = model.transcribe(temp_audio.name, language="hi", task="transcribe")
            os.remove(temp_audio.name)
            return jsonify(result)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🟢 Whisper Server is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
