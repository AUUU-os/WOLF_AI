"""
WOLF_AI Termux Integration

Android integration via Termux:API for Samsung S24 Ultra.
The phone becomes THE FOREST - the pack's mobile command center.
"""

from .termux_bridge import (
    TermuxWolf,
    TermuxNotification,
    TermuxSMS,
    TermuxDevice,
    TermuxClipboard,
    TermuxSpeech,
    TermuxSensors,
    TermuxMedia,
    get_termux,
    notify,
    speak,
    battery,
    vibrate
)

__all__ = [
    "TermuxWolf",
    "TermuxNotification", "TermuxSMS", "TermuxDevice",
    "TermuxClipboard", "TermuxSpeech", "TermuxSensors", "TermuxMedia",
    "get_termux", "notify", "speak", "battery", "vibrate"
]
