# 🦯 BlindAssist — AI Object Detection for the Visually Impaired

A real-time assistive system using **YOLOv8 + Text-to-Speech** to help
blind and visually-impaired users navigate their environment.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time Detection** | YOLOv8 detects 80+ object classes from webcam |
| **Voice Alerts (TTS)** | Speaks object name, direction, and distance |
| **Distance Estimation** | Very close / Nearby / Ahead / Far away (no depth sensor needed) |
| **Direction Guidance** | Far left / To your left / Straight ahead / To your right / Far right |
| **Danger Warnings** | Cars, dogs, bikes etc. trigger `"Warning!"` prefix with shorter cooldown |
| **Scene Description** | Press SPACE for a full spoken description of the scene |
| **Danger Zone HUD** | Visual red overlay + banner when objects are dangerously close |
| **Adjustable Confidence** | Live `+` / `-` keys to tune detection sensitivity |
| **Scene / Detect Modes** | Toggle between continuous detection and periodic scene narration |
| **Model Flexibility** | Swap between yolov8n / s / m / l / x models via config |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run
```bash
python main.py
```
YOLOv8 will auto-download `yolov8n.pt` (~6 MB) on first run if not present.

---

## 🎮 Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Toggle Scene Description mode |
| `SPACE` | Describe current scene now (spoken) |
| `+` / `=` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

---

## ⚙️ Configuration (`utils/config.py`)

```python
"model_path":              "yolov8n.pt"   # change to yolov8s.pt for better accuracy
"camera_source":           0              # 0=webcam, 1=external, or "video.mp4"
"confidence_threshold":    0.45           # 0.3–0.6 recommended
"announcement_cooldown":   4.0            # seconds between repeat announcements
```

**Model size vs speed trade-off:**

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| yolov8n.pt | 6 MB  | ⚡⚡⚡⚡ | ⭐⭐ |
| yolov8s.pt | 22 MB | ⚡⚡⚡  | ⭐⭐⭐ |
| yolov8m.pt | 52 MB | ⚡⚡    | ⭐⭐⭐⭐ |
| yolov8l.pt | 87 MB | ⚡      | ⭐⭐⭐⭐⭐ |

> **Recommendation for blind assist:** Use `yolov8s.pt` — best balance of speed and accuracy for real-time use.

---

## 📁 Project Structure

```
blind_assist/
├── main.py                  ← Entry point
├── requirements.txt
├── utils/
│   ├── config.py            ← All settings
│   └── draw.py              ← OpenCV HUD & bounding box drawing
└── modules/
    ├── tts_engine.py        ← Text-to-Speech (pyttsx3)
    ├── distance.py          ← Distance estimation from bbox size
    ├── direction.py         ← Left / Centre / Right classification
    ├── danger.py            ← Dangerous object list
    └── scene.py             ← Natural language scene description
```

---

## 🔊 Voice Output Examples

- *"Person, straight ahead, very close"*
- *"Warning! Car, to your left, nearby"*
- *"I can see 2 people, a chair, and a dog."* (scene mode)
- *"Warning! Dog, far right, nearby"*

---

## 💡 Tips for Best Results

1. **Hold phone/camera at chest height** facing forward for best detection
2. Use **yolov8s.pt** for a good speed/accuracy balance
3. **Lower confidence to 0.35** in low-light conditions
4. Connect **Bluetooth earbuds** for hands-free audio feedback
5. For outdoor use, **confidence 0.5** reduces false positives

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| No audio | `pip install pyttsx3` and ensure speakers/headphones are connected |
| Low FPS | Switch to `yolov8n.pt` in config |
| Too many false alerts | Raise confidence threshold (`+` key) |
| Missing detections | Lower confidence threshold (`-` key) |
| Camera not opening | Change `camera_source` to `1` in config |

---

## 📦 Dependencies

- `ultralytics` — YOLOv8 inference
- `opencv-python` — Camera & display
- `pyttsx3` — Offline text-to-speech
- `numpy` — Array operations
# blind-assist
