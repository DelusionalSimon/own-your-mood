# Own Your Mood
**The Privacy-First Mental Health Screen | Powered by Edge AI**

![Platform](https://img.shields.io/badge/Platform-Android-green) ![Model](https://img.shields.io/badge/Model-ResNet1D-blue) ![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-red) ![Status](https://img.shields.io/badge/Status-Hackathon%20Prototype-orange)

> **"Your voice reveals your mental state. But you shouldn't have to trade your privacy to understand it."**

---

**⚠️ HACKATHON DISCLAIMER**
This project was "vibecoded" in 24 hours during the GoWest 2026 Hackathon. The architecture is sound, but the codebase reflects the speed of the event. Expect hardcoded paths, rapid prototyping patterns, and a focus on "making it work" over "making it perfect."

---

## The Problem
Mental health is a global crisis, but digital diagnostic tools are failing due to **Privacy Paralysis**.
* **The Trap:** Current AI solutions often require uploading intimate voice recordings to the cloud for processing.
* **The Risk:** Under regulations like GDPR and HIPAA, voice is a biometric. Storing it creates massive liability and user distrust. Patients won't talk to an app if they think Big Tech is listening.

## The Solution: Epistemic Edge AI
**Own Your Mood** is an "Air-Gapped" diagnostic tool. It detects biomarkers of **Depression** and **Anxiety** from the *physics* of your voice—without a single byte of audio ever leaving your phone.

We use **Embedl Hub** to optimize a deep **ResNet-1D** architecture, enabling it to run in real-time on standard Android hardware.

## The Science: Physics, Not Words
Unlike standard NLP models that analyze *what* you say (semantics), we analyze *how* you say it (prosody). Our model hunts for specific "Glitches in the Physics" of speech production:

| Biomarker | The "Glitch" | Clinical Correlation |
| :--- | :--- | :--- |
| **Flat Affect** | Reduced Fundamental Frequency (F0) Variance | Depression / Psychomotor Retardation |
| **Jitter** | High-frequency micro-tremors in pitch | Anxiety / Stress Response |
| **Shimmer** | Amplitude instability | Neurological Fatigue |
| **Pacing** | Abnormal pause duration (>500ms) | Cognitive Load / Depression |

## Embedl Integration & Performance
We used **Embedl Hub** to compress our ResNet architecture for mobile deployment, proving that deep learning is viable on edge devices without sacrificing user experience.

* **Model:** ResNet-18 (Adapted for 1D/Spectrograms)
* **Framework:** TensorFlow -> TFLite
* **Optimization Target:** Latency & Battery Life
* **Result:** ~15ms inference time on Pixel hardware

