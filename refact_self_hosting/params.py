from fastapi import Query
from pydantic import BaseModel, Required

from typing import Union, Dict, List


class SamplingParams(BaseModel):
    model: str = Query(default="", regex="^[a-z/A-Z0-9_\.\-]+$")
    max_tokens: int = 500
    temperature: float = 0.2
    top_p: float = 1.0
    top_n: int = 0
    stop: Union[List[str], str] = []
    stream: bool = False

    def clamp(self):
        def _clamp(a, b, x):
            return max(a, min(b, x))
        self.temperature = _clamp(0, 4, self.temperature)
        self.top_p = _clamp(0.0, 1.0, self.top_p)
        self.top_n = _clamp(0, 1000, self.top_n)
        self.max_tokens = _clamp(0, 8192, self.max_tokens)
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_n": self.top_n,
            "max_tokens": self.max_tokens,
            "stop_tokens": self.stop,
        }


class TextSamplingParams(SamplingParams):
    prompt: str
    echo: bool = False


class DiffSamplingParams(SamplingParams):
    intent: str
    sources: Dict[str, str]
    cursor_file: str
    cursor0: int
    cursor1: int
    function: str = Query(
        default=Required,
        regex="^([a-z0-9\.\-]+)$"
    )
    max_edits: int = 4


class ChatSamplingParams(BaseModel):
    model: str = Query(default="", regex="^[a-z/A-Z0-9_\.\-]+$")
    messages: List[Dict[str, str]]
    max_tokens: int = 500
    temperature: float = 0.2
    stop: Union[List[str], str] = []
    stream: bool = False

    def clamp(self):
        def _clamp(a, b, x):
            return max(a, min(b, x))
        self.temperature = _clamp(0, 4, self.temperature)
        self.max_tokens = _clamp(0, 8192, self.max_tokens)
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
