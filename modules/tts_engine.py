import sys
import threading
import queue
import subprocess
import time

class VoiceEngine:
    """
    Background TTS thread.
    - Windows: uses built-in PowerShell SAPI (no install needed)
    - Other OS: uses pyttsx3
    """
    def __init__(self):
        self._q = queue.Queue()
        self._method = self._pick_method()
        print(f"[VOICE] Engine: {self._method}")
        threading.Thread(target=self._worker, daemon=True, name="TTS").start()

    def _pick_method(self):
        if sys.platform == "win32":
            return "sapi"
        try:
            import pyttsx3
            return "pyttsx3"
        except ImportError:
            return "none"

    def _worker(self):
        eng = None
        if self._method == "pyttsx3":
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", 155)
            eng.setProperty("volume", 1.0)

        while True:
            text = self._q.get()
            if text is None:
                break
            print(f"  >> VOICE: {text}")
            try:
                if self._method == "sapi":
                    safe = text.replace('"', "").replace("'", "")
                    ps = (
                        "Add-Type -AssemblyName System.Speech;"
                        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                        "$s.Rate=0;$s.Volume=100;"
                        f'$s.Speak("{safe}");'
                    )
                    subprocess.run(
                        ["powershell", "-WindowStyle", "Hidden", "-Command", ps],
                        timeout=15, capture_output=True
                    )
                elif self._method == "pyttsx3" and eng:
                    eng.say(text)
                    eng.runAndWait()
            except Exception as e:
                print(f"  [VOICE ERR] {e}")

    def say(self, text: str):
        if self._q.qsize() < 2:
            self._q.put(text)

    def stop(self):
        self._q.put(None)
