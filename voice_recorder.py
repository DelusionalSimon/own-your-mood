"""
Voice Recorder Module
Handles audio recording, playback, and file management
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import time
import json


class VoiceRecorder:
    """Handles audio recording and playback functionality"""
    
    def __init__(self, sample_rate=44100, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.paused = False
        self.audio_data = []
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        self.recording_thread = None
        self.start_time = None
        self.pause_time = None
        self.total_pause_duration = 0
        
    def start_recording(self):
        """Start recording audio"""
        if self.recording:
            return False
            
        self.recording = True
        self.paused = False
        self.audio_data = []
        self.start_time = time.time()
        self.total_pause_duration = 0
        
        def record_audio():
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback
            ):
                while self.recording:
                    sd.sleep(100)
        
        self.recording_thread = threading.Thread(target=record_audio, daemon=True)
        self.recording_thread.start()
        return True
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio input stream"""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording and not self.paused:
            self.audio_data.append(indata.copy())
    
    def pause_recording(self):
        """Pause the current recording"""
        if self.recording and not self.paused:
            self.paused = True
            self.pause_time = time.time()
            return True
        return False
    
    def resume_recording(self):
        """Resume a paused recording"""
        if self.recording and self.paused:
            self.paused = False
            if self.pause_time:
                self.total_pause_duration += time.time() - self.pause_time
                self.pause_time = None
            return True
        return False
    
    def stop_recording(self):
        """Stop recording and return the filename"""
        if not self.recording:
            return None
            
        self.recording = False
        self.paused = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        if not self.audio_data:
            return None
        
        # Concatenate all audio chunks
        audio_array = np.concatenate(self.audio_data, axis=0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.recordings_dir / f"recording_{timestamp}.wav"
        
        # Save the audio file
        sf.write(filename, audio_array, self.sample_rate)
        
        return str(filename)
    
    def get_recording_duration(self):
        """Get the current recording duration in seconds"""
        if not self.recording or not self.start_time:
            return 0
        
        current_time = time.time()
        if self.paused and self.pause_time:
            return self.pause_time - self.start_time - self.total_pause_duration
        else:
            pause_offset = self.total_pause_duration
            if self.paused and self.pause_time:
                pause_offset += current_time - self.pause_time
            return current_time - self.start_time - pause_offset
    
    def play_recording(self, filename):
        """Play a recorded audio file"""
        try:
            data, sample_rate = sf.read(filename)
            sd.play(data, sample_rate)
            return True
        except Exception as e:
            print(f"Error playing recording: {e}")
            return False
    
    def stop_playback(self):
        """Stop any currently playing audio"""
        sd.stop()
    
    def get_recordings(self):
        """Get list of all recordings"""
        recordings = []
        for file in self.recordings_dir.glob("*.wav"):
            emotion_data = self.load_emotion_metadata(str(file))
            recordings.append({
                'filename': file.name,
                'path': str(file),
                'timestamp': datetime.fromtimestamp(file.stat().st_mtime),
                'size': file.stat().st_size,
                'emotion': emotion_data.get('emotion'),
                'emotion_intensity': emotion_data.get('intensity'),
                'emotion_color': emotion_data.get('color'),
                'emotion_emoji': emotion_data.get('emoji'),
            })
        # Sort by timestamp, newest first
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        return recordings
    
    def delete_recording(self, filepath):
        """Delete a recording file and its emotion metadata"""
        try:
            Path(filepath).unlink()
            # Also delete emotion metadata file if it exists
            emotion_file = Path(filepath).with_suffix('.json')
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
            print(f"Error loading emotion metadata: {e}")
        return {}
