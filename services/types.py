"""Definierar datatyper (Word, Segment) som används i hela projektet."""

from dataclasses import dataclass
from typing import List

@dataclass
class Word:
    start: float
    end: float
    text: str

@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str
    words: List[Word]
