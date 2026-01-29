"""
Voice Recorder App - Mobile Version
Stable, clean, and bug-free.
"""
import flet as ft
import flet_audio_recorder as far
from voice_recorder import VoiceRecorder
from emotion_detector import EmotionDetector
from analytics import EmotionAnalytics
import threading
import time
from datetime import datetime

class VoiceRecorderApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.recorder_manager = VoiceRecorder()
        self.emotion_detector = EmotionDetector()
        
        self.is_recording = False
        self.timer_running = False
        
        # Track which files are currently being analyzed (set of paths)
        self.processing_files = set()
        
        # --- AUDIO RECORDER ---
        self.audio_recorder = far.AudioRecorder(
            audio_encoder=far.AudioEncoder.WAV,
            on_state_changed=self.handle_recorder_state
        )
        self.page.overlay.append(self.audio_recorder)
        
        # UI Components
        self.streak_text = None 
        self.streak_fire_row = None 
        self.record_button = None
        self.stop_button = None
        self.timer_text = None
        self.recording_indicator = None
        self.recordings_list = None
        self.content_container = None
        
        self.setup_page()
        self.build_ui()
    
    def setup_page(self):
        self.page.title = "Own Your Mood"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = "#1a1625" 
        self.page.padding = 20
        self.page.window.width = 400 
        self.page.window.height = 800

    # --- STREAK LOGIC ---
    def calculate_streak(self):
        """Calculates current streak of daily usage"""
        recordings = self.recorder_manager.get_recordings()
        if not recordings:
            return 0
            
        # Get unique dates from recordings
        dates = sorted(list(set(r['timestamp'].date() for r in recordings)), reverse=True)
        
        if not dates:
            return 0

        today = datetime.now().date()
        last_recording = dates[0]
        
        # If last recording was before yesterday, streak is broken
        if (today - last_recording).days > 1:
            return 0
            
        streak = 1
        current_date = last_recording
        for prev_date in dates[1:]:
            if (current_date - prev_date).days == 1:
                streak += 1
                current_date = prev_date
            else:
                break
                
        return streak

    def get_fire_count(self, streak):
        """Returns the number of fires based on streak length"""
        if streak == 0:
            return 1 # Show 1 grey/dim fire if 0
        elif streak < 7:
            return 1
        elif streak < 30:
            return 2
        else:
            return 3

    # --- UI BUILDER ---
    def build_ui(self):
        """Build the user interface"""
        
        # --- HEADER WITH DYNAMIC STREAK ---
        self.streak_text = ft.Text("0", size=20, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
        
        # We use a Row with negative spacing to push fires closer
        self.streak_fire_row = ft.Row(
            controls=[ft.Text("🔥", size=20)], 
            spacing=-5, # Negative spacing pushes them together
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER
        )
        
        streak_badge = ft.Container(
            content=ft.Row([
                self.streak_fire_row,
                self.streak_text,
                ft.Text("Day Streak", size=14, color=ft.Colors.WHITE70)
            ], spacing=5, alignment=ft.MainAxisAlignment.CENTER),
            padding=ft.padding.symmetric(horizontal=15, vertical=8),
        )

        header = ft.Container(
            content=ft.Column([
                ft.Text("Own Your Mood!", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                ft.Container(height=5),
                streak_badge
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            margin=ft.Margin(0, 20, 0, 20),
        )
        
        # Nav Buttons
        self.recorder_btn = ft.ElevatedButton(
            "Record", icon=ft.Icons.MIC,
            on_click=lambda e: self.switch_view(0),
            style=ft.ButtonStyle(bgcolor=ft.Colors.PINK_400, color=ft.Colors.WHITE),
            width=130
        )
        self.analytics_btn = ft.ElevatedButton(
            "Insights", icon=ft.Icons.PIE_CHART,
            on_click=lambda e: self.switch_view(1),
            style=ft.ButtonStyle(color=ft.Colors.WHITE, bgcolor=ft.Colors.WHITE10),
            width=130
        )
        
        nav_row = ft.Row([self.recorder_btn, self.analytics_btn], alignment=ft.MainAxisAlignment.CENTER, spacing=10)
        
        # Content
        self.recorder_content = self.build_recorder_tab()
        self.content_container = ft.Container(content=self.recorder_content, expand=True)
        
        self.page.add(ft.Column([header, nav_row, ft.Divider(height=20, color="transparent"), self.content_container], expand=True))
        self.refresh_recordings_list()

    def switch_view(self, index):
        if index == 0:
            self.recorder_btn.style.bgcolor = ft.Colors.PINK_400
            self.analytics_btn.style.bgcolor = ft.Colors.WHITE10
            self.content_container.content = self.recorder_content
        else:
            self.recorder_btn.style.bgcolor = ft.Colors.WHITE10
            self.analytics_btn.style.bgcolor = ft.Colors.PINK_400
            self.content_container.content = self.build_analytics_tab()
        self.page.update()

    def build_recorder_tab(self):
        self.recording_indicator = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.CIRCLE, color=ft.Colors.PINK_400, size=12),
                ft.Text(" Listening...", size=16, color=ft.Colors.PINK_100),
            ], alignment=ft.MainAxisAlignment.CENTER),
            visible=False,
        )
        
        self.timer_text = ft.Text("00:00", size=50, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
        
        self.record_button = ft.Container(
            content=ft.Icon(ft.Icons.MIC, size=40, color=ft.Colors.WHITE),
            width=90, height=90,
            bgcolor=ft.Colors.PINK_500,
            border_radius=45,
            alignment=ft.alignment.center,
            on_click=self.start_recording,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.PINK_900, offset=ft.Offset(0, 4))
        )
        
        self.stop_button = ft.Container(
            content=ft.Icon(ft.Icons.STOP_ROUNDED, size=40, color=ft.Colors.WHITE),
            width=90, height=90,
            bgcolor=ft.Colors.PURPLE_400,
            border_radius=45,
            alignment=ft.alignment.center,
            on_click=self.stop_recording_click,
            visible=False,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.PURPLE_900, offset=ft.Offset(0, 4))
        )
        
        self.recordings_list = ft.Column(spacing=10, scroll=ft.ScrollMode.HIDDEN)

        return ft.Container(
            content=ft.Column([
                ft.Container(height=10),
                self.recording_indicator,
                ft.Container(self.timer_text, alignment=ft.alignment.center, padding=10),
                ft.Container(height=10),
                ft.Row([self.record_button, self.stop_button], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(height=30),
                ft.Text("Past Moods", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE70),
                ft.Container(self.recordings_list, expand=True)
            ]),
            padding=10,
        )

    def build_analytics_tab(self):
        recordings = self.recorder_manager.get_recordings()
        analytics = EmotionAnalytics(recordings)
        stats = analytics.get_summary_stats()
        
        def stat_card(title, value):
            return ft.Container(
                content=ft.Column([
                    ft.Text(title, size=12, color=ft.Colors.WHITE54),
                    ft.Text(str(value), size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
                ], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                bgcolor=ft.Colors.WHITE10, padding=15, border_radius=20, expand=True
            )

        row1 = ft.Row([
            stat_card("Total Entries", stats['total_recordings']),
            stat_card("Moods Found", stats['recordings_with_emotions']),
        ])
        
        dist_col = ft.Column(spacing=10)
        for emotion, pct in stats['emotion_percentages'].items():
            meta = self.emotion_detector.get_emotion_info(emotion)
            dist_col.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"{meta['emoji']} {emotion.capitalize()}", size=16, color=ft.Colors.WHITE),
                        ft.Text(f"{pct:.0f}%", weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE70)
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=15, border_radius=15, bgcolor=meta['color'] 
                )
            )

        return ft.Container(
            content=ft.Column([
                row1, ft.Divider(height=20, color="transparent"),
                ft.Text("Emotional Profile", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE70),
                ft.Container(height=10), dist_col
            ], scroll=ft.ScrollMode.HIDDEN),
            padding=10
        )

    # --- LOGIC ---
    def start_recording(self, e):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_{timestamp}.wav"
        save_path = str(self.recorder_manager.recordings_dir / temp_filename)

        if self.page.web: self.audio_recorder.start_recording()
        else: self.audio_recorder.start_recording(output_path=save_path)
             
        self.is_recording = True
        self.record_button.visible = False
        self.stop_button.visible = True
        self.recording_indicator.visible = True
        self.start_timer()
        self.page.update()

    async def stop_recording_click(self, e):
        self.is_recording = False
        self.timer_running = False
        
        output_path = await self.audio_recorder.stop_recording_async()
        
        self.record_button.visible = True
        self.stop_button.visible = False
        self.recording_indicator.visible = False
        self.timer_text.value = "00:00"
        self.page.update()

        if output_path:
            final_path = self.recorder_manager.save_recording(output_path)
            if final_path:
                # Add to processing set so it spins
                self.processing_files.add(final_path)
                self.refresh_recordings_list()
                
                self.run_analysis(final_path)

    def handle_recorder_state(self, e):
        pass

    def run_analysis(self, filepath):
        self.page.snack_bar = ft.SnackBar(content=ft.Text("Reading vibes..."), bgcolor=ft.Colors.PURPLE_700)
        self.page.snack_bar.open = True
        self.page.update()

        def analyze():
            # Run AI
            result = self.emotion_detector.analyze_audio(filepath)
            self.recorder_manager.save_emotion_metadata(filepath, result)
            
            # Remove from processing set (done)
            if filepath in self.processing_files:
                self.processing_files.remove(filepath)
            
            # Update UI
            self.refresh_recordings_list()
            
            msg = f"Vibe: {result['emoji']} {result['emotion'].capitalize()}"
            self.page.snack_bar = ft.SnackBar(content=ft.Text(msg), bgcolor=result.get('color', 'green'))
            self.page.snack_bar.open = True
            self.page.update()

        threading.Thread(target=analyze, daemon=True).start()

    def refresh_recordings_list(self):
        self.recordings_list.controls.clear()
        recs = self.recorder_manager.get_recordings()
        
        if not recs:
            self.recordings_list.controls.append(
                ft.Container(content=ft.Text("No vibes yet.", italic=True, color=ft.Colors.WHITE24), padding=20, alignment=ft.alignment.center)
            )
        
        for r in recs:
            self.recordings_list.controls.append(self.create_recording_item(r))
            
        # Update Streak
        streak = self.calculate_streak()
        if self.streak_text: self.streak_text.value = str(streak)
        count = self.get_fire_count(streak)
        if self.streak_fire_row: self.streak_fire_row.controls = [ft.Text("🔥", size=20) for _ in range(count)]
        self.page.update()

    def create_recording_item(self, r):
        date_str = r['timestamp'].strftime("%b %d, %H:%M")
        
        # Determine State
        is_processing = r['path'] in self.processing_files
        has_result = r['emotion'] is not None

        if is_processing:
            # --- SPINNING STATE (Active) ---
            return ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.ProgressRing(width=20, height=20, stroke_width=2, color=ft.Colors.PINK_400),
                        padding=12, bgcolor=ft.Colors.WHITE10, border_radius=15,
                        width=44, height=44, alignment=ft.alignment.center
                    ),
                    ft.Column([
                        ft.Text("Analyzing Vibes...", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.PINK_100, italic=True),
                        ft.Text(date_str, size=12, color=ft.Colors.WHITE24)
                    ], expand=True),
                    ft.IconButton(ft.Icons.PLAY_ARROW_ROUNDED, icon_color=ft.Colors.WHITE12, disabled=True),
                ]),
                bgcolor=ft.Colors.BLACK12, padding=10, border_radius=20, border=ft.border.all(1, ft.Colors.PINK_900)
            )
        elif has_result:
            # --- DONE STATE ---
            confidence_val = r.get('emotion_confidence', 0)
            conf_str = f"{int(confidence_val * 100)}%" # Format as percentage

            return ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Text(r['emotion_emoji'] or "✨", size=24),
                        padding=10, bgcolor=ft.Colors.WHITE10, border_radius=15
                    ),
                    ft.Column([
                        # Row to hold Emotion Name + Percentage
                        ft.Row([
                            ft.Text(f"{r['emotion'].capitalize()}", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.WHITE),
                            ft.Text(conf_str, size=12, color=ft.Colors.WHITE54)
                        ], spacing=6, alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.END),
                        
                        ft.Text(date_str, size=12, color=ft.Colors.WHITE54)
                    ], expand=True),
                    ft.IconButton(ft.Icons.PLAY_ARROW_ROUNDED, icon_color=ft.Colors.PINK_200, on_click=lambda e: self.play_audio(r['path'])),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE_ROUNDED, icon_color=ft.Colors.WHITE24, on_click=lambda e: self.delete_rec(r['path']))
                ]),
                bgcolor=ft.Colors.WHITE10, padding=10, border_radius=20
            )
        else:
            # --- UNKNOWN STATE ---
            return ft.Container(
                content=ft.Row([
                    ft.Container(content=ft.Icon(ft.Icons.ERROR_OUTLINE, color=ft.Colors.GREY), padding=10, bgcolor=ft.Colors.WHITE10, border_radius=15),
                    ft.Column([
                        ft.Text("Unknown", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.GREY),
                        ft.Text(date_str, size=12, color=ft.Colors.WHITE24)
                    ], expand=True),
                    ft.IconButton(ft.Icons.DELETE_OUTLINE_ROUNDED, icon_color=ft.Colors.WHITE24, on_click=lambda e: self.delete_rec(r['path']))
                ]),
                bgcolor=ft.Colors.WHITE10, padding=10, border_radius=20
            )

    def play_audio(self, path):
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