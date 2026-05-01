"""End-to-end orchestration for Dhamma talk generation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dhamma_pipeline.audio_post import (
    finalize_wav_for_playback,
    master_audio_with_rubric,
    write_raw_normalized_wav,
)
from dhamma_pipeline.config import PipelineConfig
from dhamma_pipeline.llm import generate_guidelines, generate_transcript_text
from dhamma_pipeline.models import TalkTranscript
from dhamma_pipeline.transcript import (
    build_output_paths,
    write_json_transcript,
    write_markdown_transcript,
)
from dhamma_pipeline.tts_engine import CoquiTTSEngine


@dataclass
class PipelineResult:
    title: str
    voice: str
    json_path: Path
    markdown_path: Path
    audio_path: Path
    word_count: int
    audio_mode: str


def run_pipeline(
    *,
    config: PipelineConfig,
    prompt: str,
    voice: str,
    duration_minutes: int,
) -> PipelineResult:
    guidelines = generate_guidelines(
        api_key=config.openai_api_key,
        model=config.openai_model,
        prompt=prompt,
        target_minutes=duration_minutes,
    )
    transcript_text = generate_transcript_text(
        api_key=config.openai_api_key,
        model=config.openai_model,
        prompt=prompt,
        guidelines=guidelines,
    )

    title = guidelines.topic.strip() or "Dhamma Talk"
    transcript = TalkTranscript.create(
        title=title,
        prompt=prompt,
        guidelines=guidelines,
        full_text=transcript_text,
        llm_model=config.openai_model,
        voice=voice,
    )

    paths = build_output_paths(config.output_dir, title)
    write_json_transcript(paths["json"], transcript)
    write_markdown_transcript(paths["markdown"], transcript)

    tts = CoquiTTSEngine(model_name=config.tts_model_name, sample_rate=config.sample_rate)
    fd, raw_name = tempfile.mkstemp(
        suffix="-tts-raw.wav",
        prefix=".",
        dir=str(config.output_dir),
    )
    os.close(fd)
    raw_path = Path(raw_name)
    audio_mode = "pending"
    try:
        tts.synthesize(
            text=transcript.full_text,
            voice=voice,
            output_path=raw_path,
            speech_rate_wpm=config.speech_rate_wpm,
        )
        if config.master_audio:
            audio_mode, _meta = master_audio_with_rubric(
                raw_path,
                paths["audio"],
                best_fit=config.audio_best_fit,
                max_iterations=10,
            )
        else:
            write_raw_normalized_wav(raw_path, paths["audio"])
            audio_mode = "no-master"
    finally:
        raw_path.unlink(missing_ok=True)

    # NSSpeech / soundfile can emit WAV variants Core Audio rejects; normalize for afplay.
    finalize_wav_for_playback(paths["audio"])

    return PipelineResult(
        title=title,
        voice=voice,
        json_path=paths["json"],
        markdown_path=paths["markdown"],
        audio_path=paths["audio"],
        word_count=len(transcript.full_text.split()),
        audio_mode=audio_mode,
    )
