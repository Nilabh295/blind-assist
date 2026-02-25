# 🦯 BlindAssist — Web Interface

A modern web-based AI object detection system for the visually impaired, built with Flask and real-time WebSocket communication.

## 🌐 New Web Features

- **Real-time Camera Feed**: Live video stream with object detection overlays
- **Interactive Dashboard**: Modern, responsive UI with dark theme
- **Live Detection Updates**: Real-time object detection via WebSockets
- **Voice Log**: Visual log of all voice announcements
- **Remote Controls**: Adjust confidence and mode from web interface
- **Scene Description**: Trigger scene narration on demand
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the web interface
```bash
python app.py
```

### 3. Open browser
Navigate to **http://localhost:5000**

---

## 🎮 Web Controls

| Control | Function |
|---|---|
| **Toggle Mode** | Switch between Detect/Scene modes |
| **Describe Scene** | Trigger immediate scene description |
| **Confidence Slider** | Adjust detection sensitivity (10-95%) |
| **Live Feed** | Real-time camera with detection overlays |

---

## 📁 Project Structure

```
blind_assist/
├── app.py                    ← Flask web server
├── main.py                   ← Original desktop app
├── requirements.txt
├── templates/
│   └── index.html           ← Web interface
├── utils/
│   └── config.py
└── modules/
    └── __init__.py
```

---

## 🔧 API Endpoints

- `GET /` - Main web interface
- `GET /api/status` - System status
- `POST /api/settings` - Update confidence/mode
- `POST /api/describe_scene` - Trigger scene description

## 🔌 WebSocket Events

- `frame` - Live video frame with detections
- `detection` - Real-time object detection
- `scene_description` - Scene narration
- `mode_change` - Mode updates

---

## 🌟 Features Comparison

| Feature | Desktop App | Web App |
|---|---|---|
| Real-time Detection | ✅ | ✅ |
| Voice Alerts | ✅ | ✅ (visual log) |
| Scene Description | ✅ | ✅ |
| Remote Control | ❌ | ✅ |
| Mobile Access | ❌ | ✅ |
| Multiple Users | ❌ | ✅ |
| Dark Theme UI | ✅ | ✅ |

---

## 💡 Usage Tips

1. **Camera Access**: Allow browser camera permissions when prompted
2. **Performance**: Web interface runs at ~30 FPS for smooth experience
3. **Mobile**: Use landscape orientation on mobile devices for best view
4. **Multiple Tabs**: Open multiple browser tabs to share the same camera feed

---

## 🔒 Security Notes

- Web interface runs locally by default (localhost:5000)
- Camera feed is only accessible within your local network
- No external API calls or data transmission
- All processing happens locally on your machine

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| Camera not working | Check browser permissions and camera hardware |
| No video feed | Ensure camera is not being used by another application |
| Connection errors | Refresh the page and check if Flask server is running |
| Poor performance | Lower confidence threshold or close other browser tabs |

---

## 📦 Dependencies (New)

- `flask>=2.3.0` - Web framework
- `flask-socketio>=5.3.0` - WebSocket support
- `python-socketio>=5.8.0` - Socket.IO client

**Original dependencies remain unchanged:**
- `ultralytics` - YOLOv8 inference
- `opencv-python` - Camera & display
- `pyttsx3` - Text-to-speech
- `numpy` - Array operations
