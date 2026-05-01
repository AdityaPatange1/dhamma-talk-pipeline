"""Open-source TTS generation with selectable voices.

This implementation uses pyttsx3, which works on Python 3.12 and supports
system-installed voices (macOS, Windows, Linux).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TTSResult:
    output_path: Path
    voice: str
    model_name: str


class TTSError(RuntimeError):
    """Raised when TTS generation fails."""


class CoquiTTSEngine:
    """TTS wrapper keeping existing interface for pipeline compatibility."""

    def __init__(self, model_name: str, sample_rate: int = 22050) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        try:
            import pyttsx3  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise TTSError(
                "pyttsx3 dependency is missing. Install with `pip install -r requirements.txt`."
            ) from exc
        self._pyttsx3: Any = pyttsx3
        self._engine: Any = pyttsx3.init()
        self._voices: list[Any] = list(self._engine.getProperty("voices") or [])

    def list_voices(self) -> list[str]:
        if not self._voices:
            return ["default"]
        labels: list[str] = []
        for voice in self._voices:
            voice_id = str(getattr(voice, "id", "")).strip()
            voice_name = str(getattr(voice, "name", "")).strip()
            if voice_name and voice_id:
                labels.append(f"{voice_name} | {voice_id}")
            elif voice_name:
                labels.append(voice_name)
            elif voice_id:
                labels.append(voice_id)
        return labels or ["default"]

    def _resolve_voice_id(self, requested_voice: str) -> str | None:
        query = requested_voice.strip().lower()
        if not query:
            return None
        for voice in self._voices:
            voice_id = str(getattr(voice, "id", ""))
            voice_name = str(getattr(voice, "name", ""))
            if query == voice_id.lower() or query == voice_name.lower():
                return voice_id
            if query in voice_id.lower() or query in voice_name.lower():
                return voice_id
        return None

    def synthesize(
        self,
        text: str,
        voice: str,
        output_path: Path,
        *,
        speech_rate_wpm: int,
    ) -> TTSResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_voice_id = self._resolve_voice_id(voice)
        if selected_voice_id:
            self._engine.setProperty("voice", selected_voice_id)
        elif voice and voice.lower() != "default":
            raise TTSError(
                f"Voice '{voice}' is unavailable. Use --list-voices to inspect supported voices."
            )

        # pyttsx3: rate is roughly words-per-minute; lower = slower, clearer speech.
        self._engine.setProperty("rate", int(speech_rate_wpm))

        # pyttsx3 writes using platform synth engines; WAV works well on macOS.
        self._engine.save_to_file(text, str(output_path))
        self._engine.runAndWait()
        return TTSResult(output_path=output_path, voice=voice, model_name=self.model_name)
