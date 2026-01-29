"""
Voice Recorder App - Mobile Version
A modern voice recording application built with Flet for Android/iOS.
"""
import flet as ft
import flet_audio_recorder as far
from voice_recorder_mobile import VoiceRecorder
from emotion_detector import EmotionDetector
from analytics import EmotionAnalytics
import threading
import time
import asyncio

class VoiceRecorderApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.recorder_manager = VoiceRecorder()
        self.emotion_detector = EmotionDetector()
        
        self.is_recording = False
        self.timer_running = False
        
        # --- AUDIO RECORDER (Mobile Native) ---
        self.audio_recorder = far.AudioRecorder(
            audio_encoder=far.AudioEncoder.WAV,
            on_state_changed=self.handle_recorder_state
        )
        self.page.overlay.append(self.audio_recorder)
        
        # UI Components
        self.record_button = None
        self.stop_button = None
        self.timer_text = None
        self.recording_indicator = None
        self.recordings_list = None
        self.content_container = None
        
        self.setup_page()
        self.build_ui()
    
    def setup_page(self):
        """Configure page settings"""
        self.page.title = "Own Your Mood"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        # Mobile-friendly dimensions for testing
        self.page.window.width = 400 
        self.page.window.height = 800

    def build_ui(self):
        """Build the user interface"""
        # Header
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.PSYCHOLOGY, size=40, color=ft.Colors.BLUE_400),
                    ft.Text("Own Your Mood", size=28, weight=ft.FontWeight.BOLD),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            margin=ft.Margin(0, 0, 0, 20),
        )
        
        # Navigation Buttons
        self.recorder_btn = ft.ElevatedButton(
            "Recorder",
            icon=ft.Icons.MIC,
            on_click=lambda e: self.switch_view(0),
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE),
        )
        
        self.analytics_btn = ft.ElevatedButton(
            "Analytics",
            icon=ft.Icons.BAR_CHART,
            on_click=lambda e: self.switch_view(1),
            style=ft.ButtonStyle(color=ft.Colors.WHITE),
        )
        
        nav_row = ft.Row(
            [self.recorder_btn, self.analytics_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        )
        
        # Content Container
        self.recorder_content = self.build_recorder_tab()
        self.content_container = ft.Container(
            content=self.recorder_content,
            expand=True,
        )
        
        self.page.add(
            ft.Column(
                [
                    header,
                    nav_row,
                    ft.Divider(height=20, color="transparent"),
                    self.content_container,
                ],
                expand=True,
            )
        )
        self.refresh_recordings_list()

    def switch_view(self, index):
        if index == 0:
            self.recorder_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE)
            self.analytics_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.SURFACE_VARIANT, color=ft.Colors.WHITE)
            self.content_container.content = self.recorder_content
        else:
            self.recorder_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.SURFACE_VARIANT, color=ft.Colors.WHITE)
            self.analytics_btn.style = ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE)
            self.content_container.content = self.build_analytics_tab()
        self.page.update()

    # --- RECORDER TAB ---
    def build_recorder_tab(self):
        self.recording_indicator = ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, color=ft.Colors.RED, size=20),
                    ft.Text("Listening...", size=16, color=ft.Colors.RED),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            visible=False,
        )
        
        self.timer_text = ft.Text("00:00", size=48, weight=ft.FontWeight.BOLD)
        
        self.record_button = ft.ElevatedButton(
            "Start Recording",
            icon=ft.Icons.MIC,
            on_click=self.start_recording,
            style=ft.ButtonStyle(bgcolor=ft.Colors.RED_400, color=ft.Colors.WHITE),
            height=60, width=160
        )
        
        self.stop_button = ft.ElevatedButton(
            "Stop Analysis",
            icon=ft.Icons.STOP,
            on_click=self.stop_recording_click,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREY_700, color=ft.Colors.WHITE),
            height=60, width=160,
            visible=False
        )
        
        self.recordings_list = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)

        return ft.Container(
            content=ft.Column(
                [
                    self.recording_indicator,
                    ft.Container(self.timer_text, alignment=ft.alignment.center, padding=10),
                    ft.Row([self.record_button, self.stop_button], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Divider(),
                    ft.Text("Recent Analyses", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(self.recordings_list, expand=True)
                ],
            ),
            padding=10,
        )

    # --- ANALYTICS TAB ---
    def build_analytics_tab(self):
        recordings = self.recorder_manager.get_recordings()
        analytics = EmotionAnalytics(recordings)
        stats = analytics.get_summary_stats()
        
        def stat_card(title, value):
            return ft.Container(
                content=ft.Column([
                    ft.Text(title, size=12, color=ft.Colors.GREY_400),
                    ft.Text(str(value), size=24, weight=ft.FontWeight.BOLD)
                ], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                bgcolor=ft.Colors.BLUE_GREY_900, padding=15, border_radius=10, expand=True
            )

        row1 = ft.Row([
            stat_card("Total", stats['total_recordings']),
            stat_card("Analyzed", stats['recordings_with_emotions']),
        ])
        
        dist_col = ft.Column()
        for emotion, pct in stats['emotion_percentages'].items():
            meta = self.emotion_detector.get_emotion_info(emotion)
            dist_col.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"{meta['emoji']} {emotion.capitalize()}", size=16),
                        ft.Text(f"{pct:.1f}%", weight=ft.FontWeight.BOLD)
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=10, border_radius=5, bgcolor=meta['color']
                )
            )

        return ft.Container(
            content=ft.Column([
                row1, 
                ft.Divider(height=20),
                ft.Text("Emotional Profile", size=18, weight=ft.FontWeight.BOLD),
                dist_col
            ], scroll=ft.ScrollMode.AUTO),
            padding=10
        )

    # --- RECORDING LOGIC ---
    # --- RECORDING LOGIC ---
    def start_recording(self, e):
        # Generate a temp filename in the recordings folder
        # We use the manager's directory just to get a safe path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_recording_{timestamp}.wav"
        
        # Construct full path. 
        # Note: on Android strict paths matter, but flet handles local paths well.
        # We will use the 'recordings' folder we created in the manager.
        save_path = str(self.recorder_manager.recordings_dir / temp_filename)

        # Start the recorder with the explicit path
        self.audio_recorder.start_recording(output_path=save_path)
        self.is_recording = True
        
        self.record_button.visible = False
        self.stop_button.visible = True
        self.recording_indicator.visible = True
        self.start_timer()
        self.page.update()

    async def stop_recording_click(self, e):
        """Stops recording and retrieves the file path"""
        self.is_recording = False
        self.timer_running = False
        
        # Stop recording.
        # This returns the path we set in start_recording
        output_path = await self.audio_recorder.stop_recording_async()
        
        # Reset UI
        self.record_button.visible = True
        self.stop_button.visible = False
        self.recording_indicator.visible = False
        self.timer_text.value = "00:00"
        self.page.update()

        if output_path:
            # 1. Finalize the file 
            # (We use the manager to copy it to the clean timestamped filename)
            final_path = self.recorder_manager.save_recording(output_path)
            
            # 2. Run AI Analysis
            if final_path:
                self.run_analysis(final_path)

    def handle_recorder_state(self, e):
        # Useful for debugging state changes (recording/stopped/etc)
        print(f"Recorder State: {e.data}")

    def run_analysis(self, filepath):
        self.page.snack_bar = ft.SnackBar(content=ft.Text("Analyzing voice biometrics..."), bgcolor=ft.Colors.BLUE_GREY_700)
        self.page.snack_bar.open = True
        self.page.update()

        def analyze():
            # Run inference in a background thread to keep UI responsive
            result = self.emotion_detector.analyze_audio(filepath)
            
            # Save results
            self.recorder_manager.save_emotion_metadata(filepath, result)
            
            # Update UI (Must be done on main thread conceptually, but Flet handles threaded updates well)
            self.refresh_recordings_list()
            
            # Show Result
            msg = f"Detected: {result['emoji']} {result['emotion'].capitalize()}"
            self.page.snack_bar = ft.SnackBar(content=ft.Text(msg), bgcolor=result.get('color', 'green'))
            self.page.snack_bar.open = True
            self.page.update()

        threading.Thread(target=analyze, daemon=True).start()

    def refresh_recordings_list(self):
        self.recordings_list.controls.clear()
        recs = self.recorder_manager.get_recordings()
        
        if not recs:
            self.recordings_list.controls.append(ft.Text("No recordings yet.", italic=True, color=ft.Colors.GREY))
        
        for r in recs:
            self.recordings_list.controls.append(self.create_recording_item(r))
        self.page.update()

    def create_recording_item(self, r):
        return ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.AUDIO_FILE, color=ft.Colors.BLUE_200),
                ft.Column([
                    ft.Text(r['filename'], weight=ft.FontWeight.BOLD),
                    ft.Text(f"{r['emotion_emoji'] or ''} {r['emotion'] or 'Unknown'}", size=12)
                ], expand=True),
                
                # Play Button (Uses Flet Audio)
                ft.IconButton(ft.Icons.PLAY_ARROW, on_click=lambda e: self.play_audio(r['path'])),
                
                # Delete Button
                ft.IconButton(ft.Icons.DELETE, icon_color="red", on_click=lambda e: self.delete_rec(r['path']))
            ]),
            bgcolor=ft.Colors.BLUE_GREY_900, padding=10, border_radius=5
        )

    def play_audio(self, path):
        # Flet Audio overlay for mobile playback
        audio = ft.Audio(src=path, autoplay=True)
        self.page.overlay.append(audio)
        self.page.update()

    def delete_rec(self, path):
        self.recorder_manager.delete_recording(path)
        self.refresh_recordings_list()

    def start_timer(self):
        self.timer_running = True
        start = time.time()
        def loop():
            while self.timer_running:
                elapsed = int(time.time() - start)
                self.timer_text.value = f"{elapsed//60:02d}:{elapsed%60:02d}"
                self.page.update()
                time.sleep(0.2)
        threading.Thread(target=loop, daemon=True).start()

def main(page: ft.Page):
    VoiceRecorderApp(page)

if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets")