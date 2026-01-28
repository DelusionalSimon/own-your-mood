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

# --------------[TCN APPROACH]------------
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.05):
    prev_x = x
    
    # 1. Dilated Convolution (The Magic)
    # "causal" padding ensures we don't cheat by looking into the future
    # (though for classification, 'same' is also acceptable)
    x = layers.Conv1D(filters=filters, 
                      kernel_size=kernel_size, 
                      padding='same',  # Keep 'same' for simple classification
                      dilation_rate=dilation_rate,
                      activation='relu')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    
    # 2. Second Conv (Standard)
    x = layers.Conv1D(filters=filters, 
                      kernel_size=kernel_size, 
                      padding='same', 
                      dilation_rate=dilation_rate,
                      activation='relu')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(dropout_rate)(x)
    
    # 3. Residual Connection
    # If shapes don't match (e.g. filter change), project prev_x
    if prev_x.shape[-1] != filters:
        prev_x = layers.Conv1D(filters, 1, padding='same')(prev_x)
        
    x = layers.Add()([x, prev_x])
    return layers.ReLU()(x)

def build_tcn_model(input_shape=(48000, 1), num_classes=8):
    inputs = layers.Input(shape=input_shape)
    
    # Initial Downsampling (Stem)
    # Raw audio is huge (48k), we need to crunch it down a bit first
    x = layers.Conv1D(32, 7, strides=4, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
    
    # TCN Stack
    # We increase dilation rate (1, 2, 4, 8) to see wider context
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=1)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2)
    x = tcn_block(x, filters=128, kernel_size=3, dilation_rate=4)
    x = tcn_block(x, filters=128, kernel_size=3, dilation_rate=8)
    x = tcn_block(x, filters=256, kernel_size=3, dilation_rate=16)

    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name="TCN_Audio_Emotion")
    
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