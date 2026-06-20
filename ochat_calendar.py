"""ochat_calendar: macOS Calendar.app I/O via AppleScript (osascript).

No model calls, no prompts, no CLI -- a pure I/O layer that ochat.py's
orchestration code decides when and why to call.
"""

import platform
import subprocess
from datetime import datetime


class CalendarError(Exception):
    """Raised when an osascript call to Calendar.app fails."""


def is_macos() -> bool:
    return platform.system() == "Darwin"
