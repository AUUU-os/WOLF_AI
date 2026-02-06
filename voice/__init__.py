"""
WOLF_AI Voice Control

"Hey Wolf!" voice assistant for controlling the pack.

Platforms:
- Desktop (Windows/Mac/Linux) - requires SpeechRecognition + pyaudio
- Android (Termux) - uses termux-speech-to-text/termux-tts-speak
- Web - uses browser Web Speech API (via dashboard)
"""

from .voice_control import (
    WolfVoiceAssistant,
    VoiceCommand,
    TextToSpeech,
    SpeechToText,
    VoicePlatform
)

__all__ = [
    "WolfVoiceAssistant",
    "VoiceCommand",
    "TextToSpeech",
    "SpeechToText",
    "VoicePlatform"
]
