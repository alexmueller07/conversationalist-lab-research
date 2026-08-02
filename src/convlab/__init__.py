"""convlab -- multimodal measurement of dyadic conversation quality.

The package turns three synchronized video recordings of a two-person
conversation into a documented table of behavioral measures: who spoke
when, how quickly each replied, what they looked at, when they nodded,
smiled and laughed, how their speech and movement tracked one another, and
how all of that changed over the course of the conversation.
"""

from __future__ import annotations

import os

# MediaPipe pulls in TensorFlow's logging on some installs; silence the
# banner before any transitive import can emit it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

__version__ = "0.1.0"

from convlab.config import Config
from convlab.session import PERSONS, Session, discover_sessions, iter_sessions

__all__ = [
    "__version__",
    "Config",
    "Session",
    "PERSONS",
    "discover_sessions",
    "iter_sessions",
]
