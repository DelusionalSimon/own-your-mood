"""
REAL Emotion Detector (TFLite)
Runs the custom ResNet model on raw audio waveforms.
"""
import os
import time
import wave
import numpy as np
from pathlib import Path

# Try importing TFLite runtime (optimized for Edge), fallback to full TF
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("⚠️ CRITICAL: TensorFlow Lite not found. Install 'tflite-runtime' or 'tensorflow'.")
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
    
    def __init__(self, model_filename="voice_model_cremad_resnet.tflite"):
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
        Reads WAV, processes (NO NORMALIZATION), and predicts.
        """
        if not Path(audio_file_path).exists():
            return {'emotion': 'neutral', 'intensity': 'low', 'error': 'File not found'}

        # If model failed to load, return dummy data so app doesn't crash
        if self.interpreter is None:
            return self._get_fallback_result()

        try:
            # --- 1. PRE-PROCESSING (The "Scream Effect" Fix) ---
            # We use standard 'wave' lib to read raw bytes
            with wave.open(str(audio_file_path), 'rb') as wf:
                sr = wf.getframerate()
                
                # Warning if sample rate is wrong (Model needs 16000)
                if sr != 16000:
                    print(f" Sample Rate Mismatch: File is {sr}Hz, Model needs 16000Hz. Predictions may fail.")

                # Read all frames
                frames = wf.readframes(wf.getnframes())
                
                # Convert 16-bit PCM bytes to Int16 array
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                
                # CRITICAL STEP: Convert to Float (-1.0 to 1.0) WITHOUT altering relative volume
                # Do NOT use: audio / np.max(audio)
                audio_float = audio_int16.astype(np.float32) / 32768.0

            # Pad/Trim to exactly 48000 samples (3 seconds)
            target_len = 48000
            if len(audio_float) > target_len:
                audio_float = audio_float[:target_len]
            else:
                padding = target_len - len(audio_float)
                audio_float = np.pad(audio_float, (0, padding), 'constant')

            # Reshape for TFLite [Batch, Time, Channels] -> [1, 48000, 1]
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