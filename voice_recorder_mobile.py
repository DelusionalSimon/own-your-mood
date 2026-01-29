"""
Voice Recorder Module (Mobile Version)
Strictly handles file management and metadata.
Actual audio capture is delegated to Flet's mobile-native recorder.
"""
import shutil
import os
import json
import time
from datetime import datetime
from pathlib import Path

class VoiceRecorder:
    """Handles file management for mobile recordings"""
    
    def __init__(self):
        # On Android, this creates a folder in the app's private storage
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
    def save_recording(self, temp_path):
        """
        Moves the temporary file from Flet to our permanent recordings folder.
        Returns the new permanent path.
        """
        if not temp_path or not os.path.exists(temp_path):
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"recording_{timestamp}.wav"
        dest_path = self.recordings_dir / new_filename
        
        try:
            # Copy/Move the temp file to our permanent directory
            shutil.copy(temp_path, dest_path)
            return str(dest_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def get_recordings(self):
        """Get list of all saved recordings"""
        recordings = []
        if not self.recordings_dir.exists():
            return []

        for file in self.recordings_dir.glob("*.wav"):
            emotion_data = self.load_emotion_metadata(str(file))
            try:
                stat = file.stat()
                recordings.append({
                    'filename': file.name,
                    'path': str(file),
                    'timestamp': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size,
                    'emotion': emotion_data.get('emotion'),
                    'emotion_intensity': emotion_data.get('intensity'),
                    'emotion_color': emotion_data.get('color'),
                    'emotion_emoji': emotion_data.get('emoji'),
                })
            except Exception as e:
                print(f"Skipping file {file}: {e}")
                
        # Sort by timestamp, newest first
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        return recordings
    
    def delete_recording(self, filepath):
        """Delete a recording file and its emotion metadata"""
        try:
            path_obj = Path(filepath)
            if path_obj.exists():
                path_obj.unlink()
            
            # Also delete emotion metadata file if it exists
            emotion_file = path_obj.with_suffix('.json')
            if emotion_file.exists():
                emotion_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting recording: {e}")
            return False
    
    def save_emotion_metadata(self, audio_filepath, emotion_data):
        """Save emotion metadata to a JSON file"""
        try:
            emotion_file = Path(audio_filepath).with_suffix('.json')
            with open(emotion_file, 'w') as f:
                json.dump(emotion_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving emotion metadata: {e}")
            return False
    
    def load_emotion_metadata(self, audio_filepath):
        """Load emotion metadata from JSON file"""
        try:
            emotion_file = Path(audio_filepath).with_suffix('.json')
            if emotion_file.exists():
                with open(emotion_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            pass 
        return {}