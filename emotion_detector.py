"""
REAL Emotion Detector (TFLite)
Runs the custom ResNet model on raw audio waveforms.
"""
import os
import time
import wave
import numpy as np
from pathlib import Path

# --- IMPORT LOGIC FOR ANDROID / PC ---
tflite = None
try:
    # 1. Try the new Google AI Edge library (Recommended for Python 3.12+)
    import ai_edge_litert.interpreter as tflite_lib
    tflite = tflite_lib
    print("✅ Loaded ai-edge-litert")
except ImportError:
    try:
        # 2. Try the classic runtime (Raspberry Pi / Older Python)
        import tflite_runtime.interpreter as tflite_lib
        tflite = tflite_lib
        print("✅ Loaded tflite-runtime")
    except ImportError:
        try:
            # 3. Fallback to full TensorFlow (Desktop Dev)
            import tensorflow.lite as tflite_lib
            tflite = tflite_lib
            print("✅ Loaded full tensorflow")
        except ImportError:
            print("⚠️ WARNING: No TFLite library found. AI will be mocked.")
            tflite = None

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
    
    def __init__(self, model_filename="voice_model.tflite"):
        """Initialize the TFLite interpreter"""
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Look for model in assets or current dir
        paths_to_check = [
            Path("assets") / model_filename,
            Path(model_filename),
            Path("models") / model_filename
        ]
        
        model_path = next((p for p in paths_to_check if p.exists()), None)
        
        if model_path and tflite:
            try:
                self.interpreter = tflite.Interpreter(model_path=str(model_path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f" Model loaded: {model_path}")
            except Exception as e:
                print(f" Model Load Error: {e}")
        else:
            print(f" Model file '{model_filename}' not found. Check your folders.")

    def analyze_audio(self, audio_file_path):
        """
        Reads WAV, processes, and predicts.
        """
        if not Path(audio_file_path).exists():
            return {'emotion': 'neutral', 'intensity': 'low', 'error': 'File not found'}

        # If model failed to load, return dummy data so app doesn't crash
        if self.interpreter is None:
            return self._get_fallback_result()

        try:
            # --- 1. LOAD RAW AUDIO ---
            with wave.open(str(audio_file_path), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0

            # --- 2. THE NOISE GATE (Background Talker Killer) ---
            # Calculate the Peak volume of the Main Speaker (You)
            max_amp = np.max(np.abs(audio_float))
            
            # Threshold: Keep only sounds that are at least 30% as loud as the peak.
            # Background talking is usually 10-20% volume. You are 80-100%.
            gate_threshold = max_amp * 0.30 
            
            # Apply the gate: Everything below threshold becomes 0.0 (Silence)
            # We use a mask to avoid "choppy" artifacts
            mask = np.abs(audio_float) > gate_threshold
            audio_gated = audio_float * mask
            
            # Update metrics based on the NEW gated audio
            audio_float = audio_gated
            max_amp = np.max(np.abs(audio_float))
            print(f"🎤 Main Speaker Amp: {max_amp:.4f} (Background Silenced)")

            # --- 3. SILENCE CHECK ---
            # If the gate killed EVERYTHING (because you didn't speak), return Neutral.
            if max_amp < 0.1:
                return self._build_result('neutral', 0.9, 'low')

            # --- 4. PREPARE FOR AI ---
            # Pad/Trim to 48000
            target_len = 48000
            if len(audio_float) > target_len:
                audio_float = audio_float[:target_len]
            else:
                padding = target_len - len(audio_float)
                audio_float = np.pad(audio_float, (0, padding), 'constant')

            input_tensor = audio_float.reshape(1, target_len, 1)

            # --- 2. INFERENCE ---
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # --- 3. DECODE RESULTS ---
            max_index = np.argmax(output_data)
            confidence = output_data[max_index]
            emotion_key = self.MODEL_CLASSES[max_index]
            
            # Calculate Intensity based on confidence score
            if confidence > 0.85: intensity = 'high'
            elif confidence > 0.55: intensity = 'medium'
            else: intensity = 'low'

            # Build result dictionary matching the Mock's format
            meta = self.EMOTIONS.get(emotion_key, self.EMOTIONS['neutral'])
            
            # --- DEBUG PRINTS ---
            print(f"Max Amplitude: {np.max(np.abs(audio_float)):.4f}") 
            print("Raw Confidences:")
            for i, score in enumerate(output_data):
                print(f"   {self.MODEL_CLASSES[i]}: {score:.4f}")
            # -------------------------------

            return {
                'emotion': emotion_key,
                'intensity': intensity,
                'color': meta['color'],
                'emoji': meta['emoji'],
                'icon': meta['icon'],
                'timestamp': time.time(),
                'confidence': float(confidence) # Extra debug info
            }

        except Exception as e:
            print(f" Analysis Error: {e}")
            return self._get_fallback_result()

    def get_emotion_info(self, emotion_name):
        """Metadata lookup for UI"""
        return self.EMOTIONS.get(emotion_name.lower(), self.EMOTIONS['neutral'])
    
    def get_all_emotions(self):
        return list(self.EMOTIONS.keys())

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