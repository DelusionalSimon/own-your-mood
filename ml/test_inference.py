import os
import numpy as np
import librosa
import tensorflow as tf
import random

# --- CONFIG ---
# Make sure this matches your saved filename
MODEL_FILE = "./models/voice_vitals_resnet_0.65acc.tflite" 
DATA_DIR = "../training_data/archive" # Path to RAVDESS
SAMPLE_RATE = 16000
DURATION = 3
INPUT_LEN = SAMPLE_RATE * DURATION
EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

def get_random_audio_file(data_dir):
    # Crawl directories to find all wav files
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                all_files.append(os.path.join(root, file))
    
    if not all_files:
        print("Error: No .wav files found in " + data_dir)
        return None
        
    return random.choice(all_files)

def preprocess_audio(file_path):
    # 1. Load
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # 2. Pad/Trim to exactly 48000
    if len(audio) > INPUT_LEN:
        audio = audio[:INPUT_LEN]
    else:
        padding = INPUT_LEN - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')

    # 3. Reshape for Model [1, 48000, 1]
    # TFLite expects float32
    audio = audio.astype(np.float32)
    audio = audio.reshape(1, INPUT_LEN, 1)
    return audio

def run_inference(interpreter, audio_data):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Point to the input data
    interpreter.set_tensor(input_details[0]['index'], audio_data)

    # Run
    interpreter.invoke()

    # Get result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0] # Return the probability array

if __name__ == "__main__":
    print(f"Loading model: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        print("Error: Model file not found. Did you run train_dual.py?")
        exit()

    # Initialize TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()

    # Pick a random file to test
    test_file = get_random_audio_file(DATA_DIR)
    if test_file:
        print(f"Testing File: {os.path.basename(test_file)}")
        
        # Parse Real Emotion from filename (e.g. 03-01-06...)
        try:
            filename = os.path.basename(test_file)
            parts = filename.split("-")
            true_label_idx = int(parts[2]) - 1
            true_label = EMOTIONS[true_label_idx]
            print(f"TRUE LABEL:   {true_label}")
        except:
            print("TRUE LABEL:   Unknown (Filename format mismatch)")

        # Run Prediction
        input_data = preprocess_audio(test_file)
        if input_data is not None:
            prediction_scores = run_inference(interpreter, input_data)
            
            # Find max score
            predicted_index = np.argmax(prediction_scores)
            predicted_label = EMOTIONS[predicted_index]
            confidence = prediction_scores[predicted_index] * 100

            print(f"PREDICTION:   {predicted_label} ({confidence:.1f}%)")
            
            # Visualization bar
            print("\n--- Confidence Breakdown ---")
            for i, score in enumerate(prediction_scores):
                bar = "#" * int(score * 20)
                print(f"{EMOTIONS[i]:<10}: {bar} {score*100:.1f}%")
            
            if predicted_index == true_label_idx:
                print("\nSUCCESS! Model got it right.")
            else:
                print("\nMISMATCH. Check the confidence breakdown.")