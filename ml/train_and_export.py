import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- CONFIG ---
SAMPLE_RATE = 16000
DURATION = 3 # seconds
INPUT_LEN = SAMPLE_RATE * DURATION
BATCH_SIZE = 16
EPOCHS = 25
# RAVDESS has 8 emotions: 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fearful, 07=Disgust, 08=Surprised
EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
NUM_CLASSES = len(EMOTIONS)

def load_audio_file(file_path):
    # Load and resample to 16kHz
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        # Pad or Trim to fixed length
        if len(audio) > INPUT_LEN:
            audio = audio[:INPUT_LEN]
        else:
            padding = INPUT_LEN - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return audio.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# --------------[RESNET APPROACH]------------
def build_resnet_1d():
    inputs = layers.Input(shape=(INPUT_LEN, 1))
    
    # Stem
    x = layers.Conv1D(32, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # Blocks (Simple ResNet-like structure)
    for filters in [64, 128, 256]:
        shortcut = layers.Conv1D(filters, 1, strides=2, padding='same')(x)
        x = layers.Conv1D(filters, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    # Output Layer: 8 Neurons for 8 Emotions
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# --------------[DATA LOADING]------------
def load_dataset(data_dir):
    X, y = [], []
    print(" Scanning RAVDESS folder...")
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                # Filename: 03-01-06-01-02-01-12.wav
                parts = file.split("-")
                try:
                    emotion_code = int(parts[2]) # 01 to 08
                    # Map 1-8 to 0-7 for training
                    label = emotion_code - 1 
                    
                    path = os.path.join(root, file)
                    audio = load_audio_file(path)
                    if audio is not None:
                        X.append(audio)
                        y.append(label)
                except ValueError:
                    continue
                    
    return np.array(X), np.array(y)

# --------------[MAIN]------------
if __name__ == "__main__":
    # 1. Load Data
    # CHANGE THIS PATH to where you unzipped RAVDESS
    dataset_path = "../training_data/archive" 
    X, y = load_dataset(dataset_path)
    
    if len(X) == 0:
        print(" No audio found! Check your dataset_path.")
        exit()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train
    model = build_resnet_1d()
    print(f" Training on {len(X_train)} files for {NUM_CLASSES} emotions...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # 3. Export
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open("voice_vitals_8class.tflite", "wb") as f:
        f.write(tflite_model)
    print(" Model saved as 'voice_vitals_8class.tflite'")