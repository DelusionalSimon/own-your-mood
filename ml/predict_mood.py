import os
import numpy as np
import librosa
import tensorflow as tf

# --- CONFIGURATION ---
# 1. Path to your recorded .wav file
target_file = "../training_data/test/Surprise.wav" 

# 2. Which model to use? (Check your ./models folder)
model_path = "../models/voice_model_cremad_resnet.tflite" 

# 3. Constants (Must match training!)
SAMPLE_RATE = 16000
INPUT_LEN = 48000 # 3 seconds * 16000 Hz
# Dataset Specifics
DATASET = "CREMAD"   # "RAVDESS" or "CREMAD"
if DATASET == "RAVDESS":
    EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
    
else: # CREMA-D
    EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Fearful", "Disgust"]

def preprocess_single_file(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    try:
        # Load audio: Forces 16kHz and Mono (crucial for phone recordings)
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        # Normalize to -1.0 to 1.0 range (fixes "Angry" bias from loud mics)
        max_val = np.max(np.abs(audio))
        if max_val > 0: # Avoid dividing by zero if silent
            audio = audio / max_val
        # ---------------------------------
        # Anything below 10% volume is considered "silence" and set to 0.
        # This kills the fan noise/hiss that looks like "Anger".
        threshold = 0.2
        audio[np.abs(audio) < threshold] = 0
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Pad or Trim to exactly 3 seconds
    if len(audio) > INPUT_LEN:
        # If too long, take the middle 3 seconds (often better than the start)
        start = (len(audio) - INPUT_LEN) // 2
        audio = audio[start : start + INPUT_LEN]
    else:
        # If too short, pad with zeros
        padding = INPUT_LEN - len(audio)
        audio = audio.astype(np.float32)
        audio = np.pad(audio, (0, padding), 'constant')

    # Reshape for TFLite [1, 48000, 1]
    audio = audio.astype(np.float32)
    audio = audio.reshape(1, INPUT_LEN, 1)
    return audio

def run_inference(tflite_path, input_data):
    # Load Interpreter
    if not os.path.exists(tflite_path):
        print(f"Model not found at: {tflite_path}")
        return None

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run Prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    return interpreter.get_tensor(output_details[0]['index'])[0]

if __name__ == "__main__":
    # 1. Check File
    # Remove quotes if you copy-pasted path with them
    target_file = target_file.strip('"').strip("'")
    
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        print("Tip: Right-click your file, select 'Copy as path', and paste it in the script.")
        exit()

    # 2. Prepare Data
    input_tensor = preprocess_single_file(target_file)
    
    # 3. Predict
    if input_tensor is not None:
        probs = run_inference(model_path, input_tensor)
        
        if probs is not None:
            # 4. Results
            # --- HACK: PUNISH THE ANGRY CLASS ---
            # Index 4 is usually Angry in RAVDESS (Check your specific list order!)
            # We multiply it by 0.5 to suppress false positives.
            #angry_index = EMOTIONS.index("Angry") 
            ##probs[angry_index] = probs[angry_index] * 0.5
            
            # Re-normalize probabilities so they sum to 1.0 again
            #probs = probs / np.sum(probs)

            max_index = np.argmax(probs)
            confidence = probs[max_index] * 100
            
            print("\n" + "="*30)
            print(f"PREDICTION: {EMOTIONS[max_index]}")
            print(f"CONFIDENCE: {confidence:.2f}%")
            print("="*30)
            
            print("\n--- Detailed Breakdown ---")
            for i, score in enumerate(probs):
                # Ascii bar chart
                bar_len = int(score * 20) 
                bar = "#" * bar_len 
                print(f"{EMOTIONS[i]:<15} {bar} {score*100:.1f}%")