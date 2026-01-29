"""
Voice Recorder Module (Mobile Version)
Handles file management and metadata. 
"""
import shutil
import os
import json
from datetime import datetime
from pathlib import Path

class VoiceRecorder:
    """Handles file management for mobile recordings"""
    
    def __init__(self):
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
    def save_recording(self, temp_path):
        """
        Finalizes the recording.
        If temp_path is already in our folder, rename it.
        If it's elsewhere, move it here.
        """
        if not temp_path:
            return None
            
        src_path = Path(temp_path)
        if not src_path.exists():
            return None

        # Generate final filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"recording_{timestamp}.wav"
        dest_path = self.recordings_dir / new_filename
        
        try:
            # Check if we are just renaming a file inside the same folder
            if src_path.parent.resolve() == self.recordings_dir.resolve():
                src_path.rename(dest_path)
            else:
                # Move from system temp to our folder
                shutil.move(str(src_path), str(dest_path))
                
            return str(dest_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def get_recordings(self):
        """Get list of all saved recordings"""
        recordings = []
        if not self.recordings_dir.exists():
            return []

        # Filter out temp files so they don't show up in the list
        for file in self.recordings_dir.glob("recording_*.wav"):
            emotion_data = self.load_emotion_metadata(str(file))
            try:
                stat = file.stat()
                recordings.append({
                    'filename': file.name,
                    'path': str(file),
                    'timestamp': datetime.fromtimestamp(stat.st_mtime),
                    'size': stat.st_size,
                    'emotion': emotion_data.get('emotion'),
                    'emotion_emoji': emotion_data.get('emoji'),
                    'emotion_color': emotion_data.get('color'),
                    'emotion_confidence': emotion_data.get('confidence', 0), # Added for percentage
                })
            except Exception as e:
                print(f"Skipping file {file}: {e}")
                
        # Sort by timestamp, newest first
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        return recordings
    
    def delete_recording(self, filepath):
        try:
            path_obj = Path(filepath)
            if path_obj.exists():
                path_obj.unlink()
            
            emotion_file = path_obj.with_suffix('.json')
            if emotion_file.exists():
                emotion_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting recording: {e}")
            return False
    
    def save_emotion_metadata(self, audio_filepath, emotion_data):
        try:
            emotion_file = Path(audio_filepath).with_suffix('.json')
            with open(emotion_file, 'w') as f:
                json.dump(emotion_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving emotion metadata: {e}")
            return False
    
    def load_emotion_metadata(self, audio_filepath):
        try:
            emotion_file = Path(audio_filepath).with_suffix('.json')
            if emotion_file.exists():
                with open(emotion_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            pass 
        return {}