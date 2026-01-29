"""
REAL Emotion Detector (TFLite)
Runs the custom ResNet model on raw audio waveforms.
"""
import os
import time
import wave
import numpy as np           
from pathlib import Path
# --- IMPORT OPEN-CV (Native ONNX Support) ---
try:
    import cv2
    print("✅ Loaded OpenCV")
except ImportError:
    print("❌ OpenCV missing. Install 'opencv-python'")
    cv2 = None

class EmotionDetector:
    """Real emotion detection using TFLite ResNet"""
    
    # 1. SETUP UI MAPPING
    # We map the Model's 6 classes to the UI's colors/icons.
    # Note: 'Surprise' is removed (Model doesn't know it). 'Neutral' is added.
    EMOTIONS = {
        'neutral': {'color': '#9E9E9E', 'emoji': '😐', 'icon': 'sentiment_neutral'},
        'happy':   {'color': '#4CAF50', 'emoji': '😊', 'icon': 'sentiment_satisfied'},
        'sad':     {'color': '#2196F3', 'emoji': '😢', 'icon': 'sentiment_dissatisfied'},
        'angry':   {'color': '#F44336', 'emoji': '😠', 'icon': 'sentiment_very_dissatisfied'},
        'fearful': {'color': '#9C27B0', 'emoji': '😨', 'icon': 'psychology_alt'},
        'disgust': {'color': '#795548', 'emoji': '🤢', 'icon': 'sick'},
    }
    
    # Model Output Order (Must match your training labels.txt!)
    MODEL_CLASSES = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']
    
    INTENSITY_LEVELS = ['low', 'medium', 'high']
    
    def __init__(self, model_filename="voice_model.onnx"):
        """Initialize the OpenCV ONNX Runtime"""
        self.net = None
        
        # Look for model in assets or current dir
        paths_to_check = [
            Path("assets") / model_filename,
            Path(model_filename),
            Path("models") / model_filename
        ]
        
        model_path = next((p for p in paths_to_check if p.exists()), None)
        
        if model_path and cv2:
            try:
                print(f"🧠 Loading ONNX Model: {model_path}")
                # Load Model using OpenCV DNN module
                self.net = cv2.dnn.readNetFromONNX(str(model_path))
                print("✅ Model Loaded Successfully")
            except Exception as e:
                print(f"❌ Model Load Error: {e}")
                self.net = None
        else:
            print(f"❌ Model file '{model_filename}' not found. Check your folders.")


    def analyze_audio(self, audio_file_path):
        """
        Reads WAV, processes via OpenCV, and predicts.
        """
        if not Path(audio_file_path).exists():
            return self._build_result('neutral', 0.0, 'low')

        # If model failed to load, return dummy data
        if self.net is None:
            return self._get_fallback_result()

        try:
            # --- 1. LOAD RAW AUDIO ---
            with wave.open(str(audio_file_path), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0

            # --- 2. THE NOISE GATE (Kept your original logic) ---
            max_amp = np.max(np.abs(audio_float))
            gate_threshold = max_amp * 0.30 
            
            mask = np.abs(audio_float) > gate_threshold
            audio_gated = audio_float * mask
            audio_float = audio_gated
            
            # Silence Check
            if max_amp < 0.1:
                return self._build_result('neutral', 0.9, 'low')

            # --- 3. PREPARE FOR AI ---
            # Pad/Trim to 48000
            target_len = 48000
            if len(audio_float) > target_len:
                audio_float = audio_float[:target_len]
            else:
                padding = target_len - len(audio_float)
                audio_float = np.pad(audio_float, (0, padding), 'constant')

            # --- 4. INFERENCE (OpenCV Specific) ---
            # Reshape to (1, 1, 48000) for ONNX
            input_blob = audio_float.reshape(1, 1, target_len)
            
            self.net.setInput(input_blob)
            output_data = self.net.forward()[0] # Get first batch result

            # --- 5. DECODE RESULTS ---
            # Softmax to get probabilities (0.0 to 1.0)
            probs = np.exp(output_data) / np.sum(np.exp(output_data))
            
            max_index = np.argmax(probs)
            confidence = float(probs[max_index])
            emotion_key = self.MODEL_CLASSES[max_index]
            
            # Calculate Intensity
            if confidence > 0.85: intensity = 'high'
            elif confidence > 0.55: intensity = 'medium'
            else: intensity = 'low'

            # Debug Print
            print(f"🎤 Prediction: {emotion_key} ({confidence:.2f})")

            return self._build_result(emotion_key, confidence, intensity)

        except Exception as e:
            print(f"Analysis Error: {e}")
            return self._get_fallback_result()

    def get_emotion_info(self, emotion_name):
        """Metadata lookup for UI"""
        return self.EMOTIONS.get(emotion_name.lower(), self.EMOTIONS['neutral'])
    
    def get_all_emotions(self):
        return list(self.EMOTIONS.keys())

    def _build_result(self, emotion, confidence, intensity):
        """Helper to build consistent result dictionary"""
        meta = self.EMOTIONS.get(emotion, self.EMOTIONS['neutral'])
        return {
            'emotion': emotion,
            'intensity': intensity,
            'color': meta['color'],
            'emoji': meta['emoji'],
            'icon': meta['icon'],
            'timestamp': time.time(),
            'confidence': float(confidence)
        }

    def _get_fallback_result(self):
        """Returns neutral if things break, prevents app crash"""
        return {
            'emotion': 'neutral',
            'intensity': 'low', 
            'color': '#9E9E9E',
            'emoji': '❓', 
            'icon': 'help',
            'timestamp': time.time()
        }

if __name__ == "__main__":
    # Quick Test
    det = EmotionDetector()
    print("Testing Emotions config...")
    print(det.get_all_emotions())