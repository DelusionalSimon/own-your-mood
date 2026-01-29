import flet as ft
from emotion_detector import EmotionDetector
import time

class OwnYourMoodApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.detector = EmotionDetector()
        
        # Audio setup
        self.audio_file = "vibe_check.wav"
        self.recorder = ft.AudioRecorder(
            audio_encoder=ft.AudioEncoder.WAV,
        )
        self.page.overlay.append(self.recorder)
        
        self.configure_page()
        self.build_ui()

    def configure_page(self):
        self.page.title = "Own Your Mood"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 40
        self.page.window.width = 400
        self.page.window.height = 700
        self.page.bgcolor = "#1A1A2E" # Deep midnight blue

    def build_ui(self):
        # Result Emoji with a soft glow effect
        self.emoji_display = ft.Text("✨", size=100)
        self.label_display = ft.Text("Ready for a vibe check?", size=20, color="#A0A0B8")
        
        # A cute, rounded card for results
        self.result_card = ft.Container(
            content=ft.Column([
                self.emoji_display,
                self.label_display,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=40,
            border_radius=30,
            bgcolor="#16213E",
            border=ft.border.all(2, "#0F3460"),
            alignment=ft.alignment.center,
        )

        # The big pulse button
        self.record_btn = ft.FloatingActionButton(
            content=ft.Icon(ft.Icons.MIC_ROUNDED, color=ft.Colors.WHITE, size=30),
            bgcolor=ft.Colors.PINK_600,
            on_click=self.toggle_recording,
            width=80,
            height=80,
        )

        self.status_bar = ft.Text("Tap to record", size=12, color=ft.Colors.PINK_200)

        # Layout
        self.page.add(
            ft.Column([
                ft.Text("OWN YOUR MOOD", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.PINK_600),
                ft.Container(height=40),
                self.result_card,
                ft.Container(height=40),
                self.record_btn,
                ft.Container(height=10),
                self.status_bar,
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        )

    def toggle_recording(self, e):
        if self.recorder.is_recording():
            self.stop_and_analyze()
        else:
            self.start_recording()

    def start_recording(self):
        self.recorder.start_recording(self.audio_file)
        self.record_btn.bgcolor = ft.Colors.RED_400
        self.status_bar.value = "Recording... talk to me!"
        self.page.update()

    def stop_and_analyze(self):
        self.recorder.stop_recording()
        self.record_btn.bgcolor = ft.Colors.PINK_600
        self.status_bar.value = "Reading your soul..."
        self.page.update()
        
        # Give the OS a millisecond to finish the file write
        time.sleep(0.5)
        
        res = self.detector.analyze_audio(self.audio_file)
        
        # Update Display
        self.emoji_display.value = res.get('emoji', "✨")
        self.label_display.value = f"Feeling {res.get('emotion', 'Unknown').capitalize()}"
        self.status_bar.value = f"Confidence: {int(res.get('confidence', 0)*100)}%"
        self.page.update()

def main(page: ft.Page):
    OwnYourMoodApp(page)

if __name__ == "__main__":
    ft.app(target=main)