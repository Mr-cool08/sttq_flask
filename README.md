# Speech Transcription & Diarization (Flask)

A simple Flask web app for transcribing audio or video files and optionally performing speaker diarization. Upload a file, choose your settings, and download the transcript as plain text, SRT, or JSON.

## Features
- Speech-to-text transcription using Hugging Face models
- Optional speaker diarization powered by `pyannote.audio`
- Outputs `transcript.txt`, `transcript.srt`, and `transcript.json`
- Lightweight web interface built with HTML, CSS, and JavaScript

## Project Structure
```
sttq_flask/
├─ app.py
├─ services/
│  ├─ __init__.py
│  ├─ types.py
│  ├─ audio_utils.py
│  ├─ stt.py
│  ├─ diarization.py
│  └─ writers.py
├─ templates/
│  └─ index.html
├─ static/
│  ├─ css/styles.css
│  └─ js/app.js
├─ requirements.txt
└─ .env.example
```

## Requirements
- Python 3.9+
- `ffmpeg` available in your `PATH`
- (Optional) NVIDIA GPU with CUDA for faster inference

## Installation
1. Clone the repository
   ```bash
   git clone https://github.com/USER/sttq_flask.git
   cd sttq_flask
   ```
2. (Optional) create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
   > `pyannote.audio` and `torch` may require specific versions depending on your platform/CUDA setup.

4. Copy `.env.example` to `.env` and fill in your values
   ```bash
   cp .env.example .env
   # then edit .env and provide your keys
   ```

## Usage
Run the development server:
```bash
python app.py
```
Open <http://127.0.0.1:5000> in your browser. Upload an audio or video file, pick your options, and download the resulting transcripts.

## Tips
- For long files consider running behind a reverse proxy and using a queue/background job system.
- Set `do_diarize` to `False` if you do not have a Hugging Face token.

