# 🎙 Phase-1 — Audio Transcription Pipeline

This phase converts **any video/audio file** → **clean WAV** → **transcribed text** and generates:

| Output | Description |
|--------|-------------|
| `transcript_words.csv` | Word-level timings & probabilities |
| `transcript_sentences.csv` | Sentence-level transcribed text |
| `silences.csv` | Automatic pause/silence detection |
| `transcript.json` | Combined JSON output |

---

## 🧠 Why Phase-1 Exists

Before editing or auto-jump-cut video, we need:
✔ Clean WAV audio in fixed format  
✔ Accurate timestamps for each spoken word  
✔ Sentence segmentation for summary/future NLP  
✔ Silence duration so Phase-2 can remove gaps  

---

## 🪄 High Level Flow

```
INPUT (mp4, mov, wav, mp3)
       │
       ▼
ffmpeg extracts audio → temp_audio.wav (16-khz, mono)
       │
       ▼
Whisper (CPU offline — base/medium/large)
       │
       ├─▶ word-level CSV
       ├─▶ sentence-level CSV
       ├─▶ transcript.json
       └─▶ silence-detection (energy-based) → silences.csv
```

---

## 🧾 Flowchart Diagram

```
phase1_flowchart.png
```

---

## 🎯 Output Folder Structure

```
output/
 └── video1/
     ├── medium/
     │    ├── transcript_words.csv
     │    ├── transcript_sentences.csv
     │    ├── silences.csv
     │    ├── transcript.json
     │    ├── audio.wav
     │
     └── large/
          ├── ...
```

---

## 🚀 CLI Usage

### Single File
```bash
python main.py input/video1.mov
```

### Folder Processing
```bash
python main.py input/
```

---

## 🔧 Change Model (Small/Medium/Large)

Edit `config.json`

```json
{
  "model_size": "medium",
  "language": "en"
}
```

Options:
```
base, small, medium, large
```

---

## 📎 Next Step — Phase-2

After timestamps exist, **Phase-2 will**:
Cut silent spaces, rebuild edited video, and align audio perfectly.

---
