"""
Mock Emotion Detector
Simulates emotion detection from audio files
In production, this would be replaced with a real ML model or API
"""
import random
import time
from pathlib import Path


class EmotionDetector:
    """Mock emotion detection for voice recordings"""
    
    # Available emotions with associated colors and emojis
    EMOTIONS = {
        'happy': {'color': '#4CAF50', 'emoji': '😊', 'icon': 'sentiment_satisfied'},
        'sad': {'color': '#2196F3', 'emoji': '😢', 'icon': 'sentiment_dissatisfied'},
        'angry': {'color': '#F44336', 'emoji': '😠', 'icon': 'sentiment_very_dissatisfied'},
        'calm': {'color': '#00BCD4', 'emoji': '😌', 'icon': 'spa'},
        'anxious': {'color': '#FF9800', 'emoji': '😰', 'icon': 'psychology'},
        'neutral': {'color': '#9E9E9E', 'emoji': '😐', 'icon': 'sentiment_neutral'},
        'excited': {'color': '#E91E63', 'emoji': '🤩', 'icon': 'celebration'},
    }
    
    def __init__(self):
        """Initialize the emotion detector"""
        self.processing_time = 1.5  # Simulate processing delay
    
    def analyze_audio(self, audio_file_path):
        """
        Analyze an audio file and return detected emotion
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            dict: Emotion analysis result with emotion, confidence, and metadata
        """
        # Simulate processing time
        time.sleep(self.processing_time)
        
        # Check if file exists
        if not Path(audio_file_path).exists():
            return {
                'emotion': 'neutral',
                'confidence': 0.5,
                'error': 'File not found'
            }
        
        # Mock analysis: randomly select an emotion
        # In production, this would analyze audio features:
        # - Pitch variations
        # - Speaking rate
        # - Voice intensity
        # - Spectral features
        # - etc.
        
        emotion = random.choice(list(self.EMOTIONS.keys()))
        confidence = random.uniform(0.65, 0.95)  # Realistic confidence range
        
        result = {
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'color': self.EMOTIONS[emotion]['color'],
            'emoji': self.EMOTIONS[emotion]['emoji'],
            'icon': self.EMOTIONS[emotion]['icon'],
            'timestamp': time.time(),
        }
        
        return result
    
    def get_emotion_info(self, emotion_name):
        """
        Get metadata for a specific emotion
        
        Args:
            emotion_name: Name of the emotion
            
        Returns:
            dict: Emotion metadata (color, emoji, icon)
        """
        return self.EMOTIONS.get(emotion_name.lower(), self.EMOTIONS['neutral'])
    
    def get_all_emotions(self):
        """Get list of all available emotions"""
        return list(self.EMOTIONS.keys())


# Example usage and testing
if __name__ == "__main__":
    detector = EmotionDetector()
    
    print("Mock Emotion Detector - Test Run")
    print("=" * 50)
    print("\nAvailable emotions:")
    for emotion, info in detector.EMOTIONS.items():
        print(f"  {info['emoji']} {emotion.capitalize()}: {info['color']}")
    
    print("\n" + "=" * 50)
    print("Simulating emotion detection...")
    print("=" * 50)
    
    # Simulate analyzing a few recordings
    for i in range(5):
        print(f"\nRecording {i+1}:")
        result = detector.analyze_audio("mock_recording.wav")
        print(f"  Emotion: {result['emoji']} {result['emotion'].capitalize()}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Color: {result['color']}")
