"""Data models used across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class TalkGuidelines:
    topic: str
    presentation_style: str
    topics_covered: list[str]
    precepts_answered: list[str]
    audience_profile: str
    duration_target_minutes: int
    tone: str
    structure_outline: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TalkTranscript:
    title: str
    prompt: str
    guidelines: TalkGuidelines
    full_text: str
    generated_at_iso: str
    llm_model: str
    voice: str

    @classmethod
    def create(
        cls,
        title: str,
        prompt: str,
        guidelines: TalkGuidelines,
        full_text: str,
        llm_model: str,
        voice: str,
    ) -> "TalkTranscript":
        return cls(
            title=title,
            prompt=prompt,
            guidelines=guidelines,
            full_text=full_text,
            generated_at_iso=datetime.now(UTC).isoformat(),
            llm_model=llm_model,
            voice=voice,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["guidelines"] = self.guidelines.to_dict()
        return payload
