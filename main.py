"""
Voice Recorder App - Main UI
A modern voice recording application built with Flet
"""
import flet as ft
from voice_recorder import VoiceRecorder
from emotion_detector import EmotionDetector
from analytics import EmotionAnalytics
import threading
import time


class VoiceRecorderApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.recorder = VoiceRecorder()
        self.emotion_detector = EmotionDetector()
        self.is_recording = False
        self.is_paused = False
        self.timer_running = False
        
        # UI Components
        self.record_button = None
        self.pause_button = None
        self.stop_button = None
        self.timer_text = None
        self.recording_indicator = None
        self.recordings_list = None
        self.tabs = None
        self.analytics_content = None
        
        self.setup_page()
        self.build_ui()
    
    def setup_page(self):
        """Configure page settings"""
        self.page.title = "Voice Recorder"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        self.page.window.width = 600
        self.page.window.height = 750
        self.page.window.resizable = True
    
    def build_ui(self):
        """Build the user interface with navigation"""
        # Header
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.MIC, size=40, color=ft.Colors.BLUE_400),
                    ft.Text("Voice Recorder", size=32, weight=ft.FontWeight.BOLD),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            margin=ft.Margin(0, 0, 0, 20),
        )
        
        # Navigation buttons
        self.recorder_btn = ft.ElevatedButton(
            "Recorder",
            icon=ft.Icons.MIC,
            on_click=lambda e: self.switch_view(0),
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400),
        )
        
        self.analytics_btn = ft.ElevatedButton(
            "Analytics",
            icon=ft.Icons.BAR_CHART,
            on_click=lambda e: self.switch_view(1),
        )
        
        nav_row = ft.Row(
            [self.recorder_btn, self.analytics_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        )
        
        # Content container
        self.recorder_content = self.build_recorder_tab()
        self.content_container = ft.Container(
            content=self.recorder_content,
            expand=True,
        )
        
        # Main layout
        self.page.add(
            ft.Column(
                [
                    header,
                    nav_row,
                    ft.Container(height=10),
                    self.content_container,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            )
        )
        
        # Load existing recordings
        self.refresh_recordings_list()
    
    def switch_view(self, index):
        """Switch between recorder and analytics views"""
        if index == 0:
            # Recorder view
            self.recorder_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400)
            self.analytics_btn.style = ft.ButtonStyle()
            # Don't rebuild recorder tab - it has active components
            # Just make sure it's visible
            if self.content_container.content != self.recorder_content:
                self.content_container.content = self.recorder_content
        else:
            # Analytics view
            self.recorder_btn.style = ft.ButtonStyle()
            self.analytics_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400)
            # Rebuild analytics to get fresh data
            self.content_container.content = self.build_analytics_tab()
        
        self.content_container.update()
        self.page.update()
    
    def build_recorder_tab(self):
        """Build the recorder tab content"""
        # Recording indicator
        self.recording_indicator = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color=ft.Colors.RED, size=20),
                    ft.Text("Recording", size=16, color=ft.Colors.RED),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            visible=False,
            animate_opacity=300,
        )
        
        # Timer display
        self.timer_text = ft.Text(
            "00:00",
            size=48,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )
        
        timer_container = ft.Container(
            content=self.timer_text,
            alignment=ft.alignment.Alignment(0, 0),
            padding=20,
        )
        
        # Control buttons
        self.record_button = ft.ElevatedButton(
            "Record",
            icon=ft.Icons.MIC,
            on_click=self.start_recording,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.RED_400,
                padding=20,
            ),
            width=140,
            height=60,
        )
        
        self.pause_button = ft.ElevatedButton(
            "Pause",
            icon=ft.Icons.PAUSE,
            on_click=self.toggle_pause,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.ORANGE_400,
                padding=20,
            ),
            width=140,
            height=60,
            visible=False,
        )
        
        self.stop_button = ft.ElevatedButton(
            "Stop",
            icon=ft.Icons.STOP,
            on_click=self.stop_recording,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_GREY_700,
                padding=20,
            ),
            width=140,
            height=60,
            visible=False,
        )
        
        controls_row = ft.Row(
            [
                self.record_button,
                self.pause_button,
                self.stop_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        )
        
        # Recordings list
        self.recordings_list = ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            height=300,
        )
        
        recordings_header = ft.Container(
            content=ft.Text("Saved Recordings", size=20, weight=ft.FontWeight.BOLD),
            margin=ft.Margin(0, 30, 0, 10),
        )
        
        return ft.Container(
            content=ft.Column(
                [
                    self.recording_indicator,
                    timer_container,
                    controls_row,
                    recordings_header,
                    ft.Container(
                        content=self.recordings_list,
                        border=ft.border.all(1, ft.Colors.BLUE_GREY_700),
                        border_radius=10,
                        padding=10,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=10,
        )
    
    def build_analytics_tab(self):
        """Build the analytics tab content"""
        recordings = self.recorder.get_recordings()
        analytics = EmotionAnalytics(recordings)
        stats = analytics.get_summary_stats()
        
        # Summary cards
        total_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Total Recordings", size=14, color=ft.Colors.GREY_500),
                    ft.Text(str(stats['total_recordings']), size=32, weight=ft.FontWeight.BOLD),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=5,
            ),
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=10,
            padding=20,
            expand=True,
        )
        
        emotions_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("With Emotions", size=14, color=ft.Colors.GREY_500),
                    ft.Text(str(stats['recordings_with_emotions']), size=32, weight=ft.FontWeight.BOLD),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=5,
            ),
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=10,
            padding=20,
            expand=True,
        )
        
        most_common_emotion = stats['most_common_emotion']
        if most_common_emotion:
            emotion_info = self.emotion_detector.get_emotion_info(most_common_emotion)
            most_common_text = f"{emotion_info['emoji']} {most_common_emotion.capitalize()}"
        else:
            most_common_text = "N/A"
        
        most_common_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Most Common", size=14, color=ft.Colors.GREY_500),
                    ft.Text(most_common_text, size=20, weight=ft.FontWeight.BOLD),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=5,
            ),
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=10,
            padding=20,
            expand=True,
        )
        
        summary_row = ft.Row(
            [total_card, emotions_card, most_common_card],
            spacing=10,
        )
        
        # Emotion distribution
        emotion_items = []
        for emotion, percentage in stats['emotion_percentages'].items():
            count = stats['emotion_distribution'][emotion]
            emotion_info = self.emotion_detector.get_emotion_info(emotion)
            
            emotion_items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Container(
                                content=ft.Row(
                                    [
                                        ft.Text(emotion_info['emoji'], size=20),
                                        ft.Text(emotion.capitalize(), size=16, weight=ft.FontWeight.BOLD),
                                    ],
                                    spacing=8,
                                ),
                                expand=True,
                            ),
                            ft.Text(f"{count}x", size=14, color=ft.Colors.GREY_500),
                            ft.Text(f"{percentage:.1f}%", size=16, weight=ft.FontWeight.BOLD),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    bgcolor=emotion_info['color'],
                    border_radius=8,
                    padding=15,
                    margin=ft.Margin(0, 0, 0, 8),
                )
            )
        
        if not emotion_items:
            emotion_items.append(
                ft.Container(
                    content=ft.Text(
                        "No emotion data yet. Record some audio to see analytics!",
                        color=ft.Colors.GREY_500,
                        italic=True,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=30,
                )
            )
        
        emotion_distribution = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Emotion Distribution", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    ft.Column(emotion_items, scroll=ft.ScrollMode.AUTO, height=250),
                ],
            ),
            border=ft.border.all(1, ft.Colors.BLUE_GREY_700),
            border_radius=10,
            padding=15,
        )
        
        # Intensity breakdown
        intensity_items = []
        for intensity, count in stats['intensity_distribution'].items():
            total_with_emotions = stats['recordings_with_emotions']
            percentage = (count / total_with_emotions * 100) if total_with_emotions > 0 else 0
            
            intensity_items.append(
                ft.Container(
                    content=ft.Row(
                        [
                            ft.Text(intensity.capitalize(), size=16, weight=ft.FontWeight.BOLD, expand=True),
                            ft.Text(f"{count}x", size=14, color=ft.Colors.GREY_500),
                            ft.Text(f"{percentage:.1f}%", size=16, weight=ft.FontWeight.BOLD),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    bgcolor=ft.Colors.BLUE_GREY_900,
                    border_radius=8,
                    padding=15,
                    margin=ft.Margin(0, 0, 0, 8),
                )
            )
        
        if not intensity_items:
            intensity_items.append(
                ft.Container(
                    content=ft.Text(
                        "No intensity data yet",
                        color=ft.Colors.GREY_500,
                        italic=True,
                    ),
                    padding=20,
                )
            )
        
        intensity_breakdown = ft.Container(
            content=ft.Column(
                [
                    ft.Text("Intensity Levels", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    ft.Column(intensity_items),
                ],
            ),
            border=ft.border.all(1, ft.Colors.BLUE_GREY_700),
            border_radius=10,
            padding=15,
        )
        
        return ft.Container(
            content=ft.Column(
                [
                    summary_row,
                    ft.Container(height=20),
                    emotion_distribution,
                    ft.Container(height=20),
                    intensity_breakdown,
                ],
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=10,
        )
    
    def start_recording(self, e):
        """Start recording audio"""
        if self.recorder.start_recording():
            self.is_recording = True
            self.is_paused = False
            
            # Update UI
            self.record_button.visible = False
            self.pause_button.visible = True
            self.stop_button.visible = True
            self.recording_indicator.visible = True
            
            # Start timer
            self.start_timer()
            
            self.page.update()
    
    def toggle_pause(self, e):
        """Pause or resume recording"""
        if self.is_paused:
            # Resume
            if self.recorder.resume_recording():
                self.is_paused = False
                self.pause_button.text = "Pause"
                self.pause_button.icon = ft.Icons.PAUSE
                self.recording_indicator.visible = True
        else:
            # Pause
            if self.recorder.pause_recording():
                self.is_paused = True
                self.pause_button.text = "Resume"
                self.pause_button.icon = ft.Icons.PLAY_ARROW
                self.recording_indicator.visible = False
        
        self.page.update()
    
    def stop_recording(self, e):
        """Stop recording and save file"""
        filename = self.recorder.stop_recording()
        self.is_recording = False
        self.is_paused = False
        self.timer_running = False
        
        # Reset UI
        self.record_button.visible = True
        self.pause_button.visible = False
        self.pause_button.text = "Pause"
        self.pause_button.icon = ft.Icons.PAUSE
        self.stop_button.visible = False
        self.recording_indicator.visible = False
        self.timer_text.value = "00:00"
        
        self.page.update()
        
        if filename:
            # Show saving message
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("Recording saved! Detecting emotion..."),
                bgcolor=ft.Colors.GREEN_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
            
            # Detect emotion in background thread
            def detect_emotion():
                emotion_result = self.emotion_detector.analyze_audio(filename)
                self.recorder.save_emotion_metadata(filename, emotion_result)
                
                # Refresh UI on main thread
                self.refresh_recordings_list()
                
                # Show emotion detected message
                self.page.snack_bar = ft.SnackBar(
                    content=ft.Text(f"Emotion detected: {emotion_result['emoji']} {emotion_result['emotion'].capitalize()}"),
                    bgcolor=ft.Colors.BLUE_700,
                )
                self.page.snack_bar.open = True
                self.page.update()
            
            emotion_thread = threading.Thread(target=detect_emotion, daemon=True)
            emotion_thread.start()
        else:
            # Refresh recordings list anyway
            self.refresh_recordings_list()
            self.page.update()
    
    def start_timer(self):
        """Start the recording timer"""
        self.timer_running = True
        
        def update_timer():
            while self.timer_running and self.is_recording:
                duration = int(self.recorder.get_recording_duration())
                minutes = duration // 60
                seconds = duration % 60
                self.timer_text.value = f"{minutes:02d}:{seconds:02d}"
                self.page.update()
                time.sleep(1)
        
        timer_thread = threading.Thread(target=update_timer, daemon=True)
        timer_thread.start()
    
    def refresh_recordings_list(self):
        """Refresh the list of recordings"""
        self.recordings_list.controls.clear()
        
        recordings = self.recorder.get_recordings()
        
        if not recordings:
            self.recordings_list.controls.append(
                ft.Container(
                    content=ft.Text(
                        "No recordings yet",
                        color=ft.Colors.GREY_500,
                        italic=True,
                    ),
                    alignment=ft.alignment.Alignment(0, 0),
                    padding=20,
                )
            )
        else:
            for recording in recordings:
                self.recordings_list.controls.append(
                    self.create_recording_item(recording)
                )
        
        self.page.update()
    
    def create_recording_item(self, recording):
        """Create a UI item for a recording"""
        timestamp_str = recording['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        size_kb = recording['size'] / 1024
        
        # Create emotion badge if emotion data exists
        emotion_badge = None
        if recording.get('emotion'):
            # Format emotion text with intensity
            intensity = recording.get('emotion_intensity', '')
            emotion_text = recording['emotion'].capitalize()
            if intensity:
                emotion_text = f"{emotion_text} ({intensity})"
            
            emotion_badge = ft.Container(
                content=ft.Row(
                    [
                        ft.Text(recording['emotion_emoji'], size=14),
                        ft.Text(
                            emotion_text,
                            size=12,
                            weight=ft.FontWeight.BOLD,
                        ),
                    ],
                    spacing=4,
                ),
                bgcolor=recording.get('emotion_color', '#9E9E9E'),
                border_radius=12,
                padding=ft.padding.symmetric(horizontal=8, vertical=4),
            )
        
        # Build the info column
        info_items = [
            ft.Text(recording['filename'], weight=ft.FontWeight.BOLD),
            ft.Text(
                f"{timestamp_str} • {size_kb:.1f} KB",
                size=12,
                color=ft.Colors.GREY_500,
            ),
        ]
        
        # Add emotion badge to info if it exists
        if emotion_badge:
            info_items.append(emotion_badge)
        
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.AUDIO_FILE, color=ft.Colors.BLUE_400),
                    ft.Column(
                        info_items,
                        spacing=4,
                        expand=True,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.PLAY_ARROW,
                        icon_color=ft.Colors.GREEN_400,
                        tooltip="Play",
                        on_click=lambda e, path=recording['path']: self.play_recording(path),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE,
                        icon_color=ft.Colors.RED_400,
                        tooltip="Delete",
                        on_click=lambda e, path=recording['path']: self.delete_recording(path),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=8,
            padding=10,
        )
    
    def play_recording(self, filepath):
        """Play a recording"""
        if self.recorder.play_recording(filepath):
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("Playing recording..."),
                bgcolor=ft.Colors.BLUE_700,
            )
            self.page.snack_bar.open = True
            self.page.update()
    
    def delete_recording(self, filepath):
        """Delete a recording"""
        if self.recorder.delete_recording(filepath):
            self.refresh_recordings_list()
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text("Recording deleted"),
                bgcolor=ft.Colors.ORANGE_700,
            )
            self.page.snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    VoiceRecorderApp(page)


if __name__ == "__main__":
    ft.app(main)
