"""Transcript serialization utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path

from dhamma_pipeline.models import TalkTranscript


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "dhamma-talk"


def build_output_paths(output_dir: Path, title: str) -> dict[str, Path]:
    safe = _slugify(title)
    return {
        "json": output_dir / f"{safe}.json",
        "markdown": output_dir / f"{safe}.md",
        "audio": output_dir / f"{safe}.wav",
    }


def write_json_transcript(path: Path, transcript: TalkTranscript) -> None:
    path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def write_markdown_transcript(path: Path, transcript: TalkTranscript) -> None:
    lines = [
        f"# {transcript.title}",
        "",
        f"- Prompt: {transcript.prompt}",
        f"- Generated At: {transcript.generated_at_iso}",
        f"- OpenAI Model: {transcript.llm_model}",
        f"- Voice: {transcript.voice}",
        "",
        "## Guidelines",
        "",
        f"- Topic: {transcript.guidelines.topic}",
        f"- Presentation Style: {transcript.guidelines.presentation_style}",
        f"- Audience: {transcript.guidelines.audience_profile}",
        f"- Tone: {transcript.guidelines.tone}",
        f"- Duration (minutes): {transcript.guidelines.duration_target_minutes}",
        "",
        "### Topics Covered",
        "",
    ]
    lines.extend([f"- {topic}" for topic in transcript.guidelines.topics_covered])
    lines.extend(["", "### Precepts Answered", ""])
    lines.extend([f"- {precept}" for precept in transcript.guidelines.precepts_answered])
    lines.extend(["", "### Structure Outline", ""])
    lines.extend([f"- {point}" for point in transcript.guidelines.structure_outline])
    lines.extend(["", "## Full Talk Transcript", "", transcript.full_text.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")
