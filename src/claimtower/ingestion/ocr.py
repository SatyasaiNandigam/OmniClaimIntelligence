"""OCR path for pages without a text layer (T2.1). Pluggable: the store depends only on the `OcrEngine` protocol.

Implemented engine: `VisionLLMOcr`, the plan's "vision-LLM fallback". It sends the page image to the model behind core/llm.py and
returns the transcription. A local engine (Tesseract, Azure Document Intelligence) can be added later by implementing the same protocol.

Honest limits: OCR text is NOT verbatim ground truth. The store marks such pages `source="ocr"` so later stages (T2.4) can lower
extraction confidence for facts taken from them; the original image is never changed. With record/replay cassettes (core.llm) a
vision OCR run is deterministic in tests.
"""
from __future__ import annotations

import re
from typing import Protocol

from claimtower.core.llm import Image, LLMClient

PROMPT = (
    "You are transcribing a scanned page of an insurance claim document. Transcribe ALL visible text exactly as written, "
    "line by line, in reading order. Do not summarise, translate, correct spelling, or add anything. "
    "If part of the page is unreadable write [illegible] in its place. Output only the transcription."
)


class OcrEngine(Protocol):
    name: str

    def recognise(self, image: bytes, mime: str) -> str: ...


class VisionLLMOcr:
    name = "vision-llm"

    def __init__(self, llm: LLMClient, prompt: str = PROMPT) -> None:
        self._llm, self._prompt = llm, prompt

    def recognise(self, image: bytes, mime: str) -> str:
        text = self._llm.vision(self._prompt, [Image(image, mime)])
        assert isinstance(text, str)
        return re.sub(r"^```(?:text)?\s*|\s*```$", "", text.strip())
