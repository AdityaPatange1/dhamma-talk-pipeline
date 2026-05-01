"""OpenAI-backed text generation for Dhamma talks."""

from __future__ import annotations

import json

from dhamma_pipeline.models import TalkGuidelines


class TalkGenerationError(RuntimeError):
    """Raised when the LLM response cannot be parsed/used."""


def _client(api_key: str):
    try:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise TalkGenerationError(
            "OpenAI dependency is missing. Install with `pip install -r requirements.txt`."
        ) from exc
    return OpenAI(api_key=api_key)


def generate_guidelines(
    *,
    api_key: str,
    model: str,
    prompt: str,
    target_minutes: int,
) -> TalkGuidelines:
    """Generate structured Dhamma talk guidelines from user prompt."""
    system_prompt = (
        "You are an expert Theravada Dhamma teacher and discourse architect. "
        "Return ONLY valid JSON with the required schema."
    )
    user_prompt = f"""
Create a detailed plan for a Dhamma talk.
User prompt: "{prompt}"
Target duration: {target_minutes} minutes.

JSON schema:
{{
  "topic": "string",
  "presentation_style": "string",
  "topics_covered": ["string", "..."],
  "precepts_answered": ["string", "..."],
  "audience_profile": "string",
  "duration_target_minutes": 0,
  "tone": "string",
  "structure_outline": ["string", "..."]
}}
""".strip()

    completion = _client(api_key).chat.completions.create(
        model=model,
        temperature=0.6,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = completion.choices[0].message.content or ""
    try:
        payload = json.loads(text)
        return TalkGuidelines(
            topic=payload["topic"],
            presentation_style=payload["presentation_style"],
            topics_covered=list(payload["topics_covered"]),
            precepts_answered=list(payload["precepts_answered"]),
            audience_profile=payload["audience_profile"],
            duration_target_minutes=int(payload["duration_target_minutes"]),
            tone=payload["tone"],
            structure_outline=list(payload["structure_outline"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise TalkGenerationError(
            "Failed to parse guidelines from OpenAI response."
        ) from exc


def generate_transcript_text(
    *,
    api_key: str,
    model: str,
    prompt: str,
    guidelines: TalkGuidelines,
) -> str:
    """Generate a complete Dhamma talk transcript in markdown-friendly prose."""
    system_prompt = (
        "You are a wise Dhamma teacher. Produce a compassionate, practical, and "
        "authentic talk with clear sections and smooth flow."
    )
    user_prompt = f"""
Generate a complete Dhamma talk transcript.

Prompt:
{prompt}

Guidelines (JSON):
{json.dumps(guidelines.to_dict(), ensure_ascii=True, indent=2)}

Requirements:
- 1200-1800 words.
- Warm and clear spoken style.
- Include opening reflection, doctrinal exposition, practical exercises, and closing blessings.
- Include how precepts are practically applied in daily life.
- No markdown headers; plain prose paragraphs separated by blank lines.
""".strip()

    completion = _client(api_key).chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = completion.choices[0].message.content or ""
    if not text.strip():
        raise TalkGenerationError("OpenAI returned an empty transcript.")
    return text.strip()
