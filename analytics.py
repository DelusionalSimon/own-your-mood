"""
Analytics Module
Calculates emotion statistics and generates data for visualizations
"""
from collections import Counter
from typing import List, Dict, Any


class EmotionAnalytics:
    """Analyzes emotion data from recordings"""
    
    def __init__(self, recordings: List[Dict[str, Any]]):
        """
        Initialize analytics with recordings data
        
        Args:
            recordings: List of recording dictionaries with emotion data
        """
        self.recordings = recordings
        self.emotions_with_data = [r for r in recordings if r.get('emotion')]
    
    def get_total_recordings(self) -> int:
        """Get total number of recordings"""
        return len(self.recordings)
    
    def get_total_with_emotions(self) -> int:
        """Get total number of recordings with emotion data"""
        return len(self.emotions_with_data)
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        Get count of each emotion
        
        Returns:
            Dictionary mapping emotion names to counts
        """
        emotions = [r['emotion'] for r in self.emotions_with_data]
        return dict(Counter(emotions))
    
    def get_emotion_percentages(self) -> Dict[str, float]:
        """
        Get percentage distribution of emotions
        
        Returns:
            Dictionary mapping emotion names to percentages
        """
        distribution = self.get_emotion_distribution()
        total = self.get_total_with_emotions()
        
        if total == 0:
            return {}
        
        return {
            emotion: (count / total) * 100
            for emotion, count in distribution.items()
        }
    
    def get_most_common_emotion(self) -> tuple:
        """
        Get the most common emotion
        
        Returns:
            Tuple of (emotion_name, count) or (None, 0) if no data
        """
        distribution = self.get_emotion_distribution()
        if not distribution:
            return (None, 0)
        
        most_common = max(distribution.items(), key=lambda x: x[1])
        return most_common
    
    def get_intensity_distribution(self) -> Dict[str, int]:
        """
        Get count of each intensity level
        
        Returns:
            Dictionary mapping intensity levels to counts
        """
        intensities = [r.get('emotion_intensity') for r in self.emotions_with_data 
                      if r.get('emotion_intensity')]
        return dict(Counter(intensities))
    
    def get_emotion_by_intensity(self, emotion: str) -> Dict[str, int]:
        """
        Get intensity breakdown for a specific emotion
        
        Args:
            emotion: Name of the emotion
            
        Returns:
            Dictionary mapping intensity levels to counts
        """
        emotion_records = [r for r in self.emotions_with_data 
                          if r.get('emotion') == emotion]
        intensities = [r.get('emotion_intensity') for r in emotion_records
                      if r.get('emotion_intensity')]
        return dict(Counter(intensities))
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics
        
        Returns:
            Dictionary with various statistics
        """
        most_common_emotion, most_common_count = self.get_most_common_emotion()
        
        return {
            'total_recordings': self.get_total_recordings(),
            'recordings_with_emotions': self.get_total_with_emotions(),
            'most_common_emotion': most_common_emotion,
            'most_common_count': most_common_count,
            'emotion_distribution': self.get_emotion_distribution(),
            'emotion_percentages': self.get_emotion_percentages(),
            'intensity_distribution': self.get_intensity_distribution(),
        }


# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_recordings = [
        {'emotion': 'happiness', 'emotion_intensity': 'high'},
        {'emotion': 'sadness', 'emotion_intensity': 'low'},
        {'emotion': 'happiness', 'emotion_intensity': 'medium'},
        {'emotion': 'anger', 'emotion_intensity': 'high'},
        {'emotion': 'happiness', 'emotion_intensity': 'low'},
        {'emotion': 'fear', 'emotion_intensity': 'medium'},
    ]
    
    analytics = EmotionAnalytics(sample_recordings)
    stats = analytics.get_summary_stats()
    
    print("Emotion Analytics Summary")
    print("=" * 50)
    print(f"Total recordings: {stats['total_recordings']}")
    print(f"Most common emotion: {stats['most_common_emotion']} ({stats['most_common_count']} times)")
    print(f"\nEmotion distribution:")
    for emotion, count in stats['emotion_distribution'].items():
        percentage = stats['emotion_percentages'][emotion]
        print(f"  {emotion.capitalize()}: {count} ({percentage:.1f}%)")
    print(f"\nIntensity distribution:")
    for intensity, count in stats['intensity_distribution'].items():
        print(f"  {intensity.capitalize()}: {count}")
