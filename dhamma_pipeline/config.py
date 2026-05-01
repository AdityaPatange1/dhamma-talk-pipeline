"""Configuration helpers for the Dhamma Talk Pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    openai_api_key: str
    openai_model: str
    output_dir: Path
    tts_model_name: str
    sample_rate: int
    speech_rate_wpm: int
    master_audio: bool
    audio_best_fit: bool


def load_config(
    output_dir: str = "outputs",
    openai_model: str = "gpt-4.1-mini",
    tts_model_name: str = "pyttsx3-system",
    sample_rate: int = 22050,
    speech_rate_wpm: int = 115,
    master_audio: bool = True,
    audio_best_fit: bool = False,
) -> PipelineConfig:
    """Load configuration from environment + CLI defaults."""
    # Load environment variables from .env if present.
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in .env or export it before running."
        )

    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    return PipelineConfig(
        openai_api_key=api_key,
        openai_model=openai_model,
        output_dir=resolved_output,
        tts_model_name=tts_model_name,
        sample_rate=sample_rate,
        speech_rate_wpm=speech_rate_wpm,
        master_audio=master_audio,
        audio_best_fit=audio_best_fit,
    )
