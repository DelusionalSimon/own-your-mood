# Own Your Mood

**Real-time Voice Emotion Detection running 100% Offline.**

Own Your Mood is a privacy-first AI application that analyzes the emotional tone of your voice in real-time. Built with **Python & Flet**, it uses a custom **ResNet neural network** optimized for **TensorFlow Lite** to detect emotions like Happiness, Anger, and Sadness without ever sending audio to the cloud.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Flet](https://img.shields.io/badge/UI-Flet-purple?logo=flutter&logoColor=white)
![TFLite](https://img.shields.io/badge/AI-TensorFlow%20Lite-orange?logo=tensorflow&logoColor=white)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-red?logo=adblock&logoColor=white)

---

## Features

* **Real-Time Inference:** Instant feedback on your emotional tone as you speak.
* **Privacy First:** All processing happens locally on your device (TFLite Interpreter). No API keys, no cloud servers.
* **Cross-Platform:** Runs on **Windows**, **macOS**, **Linux**, and **Android**.
* **Smart Noise Gate:** Filters out background noise to focus only on your voice.
* **Dynamic UI:** Visualizes confidence levels and intensity in real-time.

## Tech Stack

* **Frontend:** [Flet](https://flet.dev) (Flutter for Python)
* **AI Engine:** TensorFlow Lite (or `tflite-runtime`)
* **Audio Processing:** `numpy` & `flet-audio-recorder`
* **Model:** Custom ResNet trained on RAVDESS/TESS datasets, converted to `.tflite`.

---

## Getting Started

### Prerequisites
* Python 3.10 or higher (3.11 recommended)
* A microphone

### 1. Clone the Repo
```bash
git clone [https://github.com/your-username/own-your-mood.git](https://github.com/your-username/own-your-mood.git)
cd own-your-mood
```
2. Install Dependencies
Important: We strictly pin numpy<2.0 because TensorFlow Lite is not yet compatible with the newer Numpy 2.0 versions.

```Bash
pip install -r requirements.txt
```
3. Run the App
```Bash
python main.py
```

**Building for Android**
This project includes a GitHub Action to automatically build an APK (in the android-deployment branch).

Push your code to GitHub.

The workflow .github/workflows/build_apk.yml will trigger.

Once finished, go to the Actions tab, click the latest run, and download the app-release artifact.

NOTE: WE couldn't resolve all dependencies

**Manual Build Requirements**
If you want to build locally, you need the Flet CLI:

```Bash
flet build apk
Project Structure
```
```Plaintext
own-your-mood/
├── assets/
│   └── voice_model.tflite  # The optimized AI model
├── emotion_detector.py     # AI Logic (Audio processing + TFLite inference)
├── main.py                 # UI Logic (Flet App)
├── requirements.txt        # Dependency lockfile
└── .github/
    └── workflows/
        └── build_apk.yml   # CI/CD Pipeline
```
**Troubleshooting**
"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2" This is the most common error. TensorFlow currently requires Numpy 1.x. Fix:

```Bash
pip install "numpy<2.0"
```
"No module named 'tflite_runtime'" If you are not using the full TensorFlow library, ensure you have the runtime installed:

```Bash
pip install tflite-runtime
```