# STT + Diarisering (Flask)

En uppdelad version av appen i separata filer (HTML, CSS, JS och Python).

## Struktur
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

## Kör lokalt
1) Installera systemberoenden:
   - **ffmpeg** måste finnas i `PATH`
   - GPU (valfritt) med CUDA för högre fart

2) Skapa och aktivera virtuell miljö (valfritt) och installera Python-paket:
```
pip install -r requirements.txt
```

> Obs: `pyannote.audio` och `torch` kan ta tid att installera och kan kräva specifika versioner beroende på din plattform/CUDA.

3) Skapa `.env` från `.env.example` och fyll i:
```
FLASK_SECRET_KEY=byt-denna-hemliga-nyckel
HUGGINGFACE_TOKEN=din-huggingface-token
```

4) Starta servern:
```
python app.py
```
Öppna http://127.0.0.1:5000

5) Ladda upp en ljud-/videofil och välj inställningar. Resultat returneras som en ZIP med `transcript.txt`, `transcript.srt` och `transcript.json`.

## Tips
- Vid långa filer: kör bakom reverse proxy och överväg kö/bakgrundsjobb.
- Sätt `do_diarize` till av om du saknar HuggingFace-token.
