# 🎤 Offline Video Transcriber – Whisper + Silence Detector

A fully offline Python tool that:
- Converts video/audio → text transcript
- Saves **word-level + sentence-level timestamps**
- Detects **silent pauses** and exports them
- Supports **multiple Whisper model sizes** (base, small, medium, large-v3)
- Creates **separate output folders** per-video & per-model for comparison

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🧠 Whisper transcription | Uses `faster-whisper` (CTranslate2) for fast CPU inference |
| 🔊 Word timestamps | Every word → start time, end time, confidence |
| 📝 Sentence timestamps | Each spoken segment is extracted |
| 🔇 Silence Detection | Detects gaps using audio energy analysis |
| 🗂 Organized Results | Output → `output/<video-name>/<model-name>/...` |
| 🔌 Offline | No API / No HuggingFace call required |
| 🧪 Multi-model comparison | Run once with `medium`, again with `large` to compare accuracy |

---

## 🧱 Folder Structure

