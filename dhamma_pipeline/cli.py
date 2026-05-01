"""CLI for Dhamma Talk Pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from rich.traceback import install

from dhamma_pipeline.config import load_config
from dhamma_pipeline.pipeline import run_pipeline
from dhamma_pipeline.tts_engine import CoquiTTSEngine
from dhamma_pipeline.ui import console, render_banner, render_markdown, render_result

install(show_locals=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dhamma_talk_pipeline.py",
        description=(
            "Generate complete Dhamma talks from a prompt using OpenAI for transcript "
            "creation and open-source system TTS for audio synthesis."
        ),
        epilog=(
            "Example:\n"
            "  python dhamma_talk_pipeline.py --prompt \"Dhamma talk on mindfulness of mind\" "
            "--voice default\n\n"
            "Tip: run --list-voices to inspect all voices available in your installed "
            "speech engine."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Talk request prompt. Example: 'Dhamma talk on mindfulness of mind'.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="default",
        help="Voice name or voice ID from your system TTS voices (default: default).",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=20,
        help="Target talk duration in minutes for guideline generation (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for JSON, Markdown, and audio artifacts.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model used for guideline and transcript generation.",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="pyttsx3-system",
        help="TTS backend label (default: pyttsx3-system).",
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=115,
        help=(
            "TTS speech rate in words-per-minute (pyttsx3). Lower is slower and usually "
            "easier to follow (default: 115; typical range 80–170)."
        ),
    )
    parser.add_argument(
        "--no-master",
        action="store_true",
        help="Skip mixing/mastering (EQ, smoothing, saturation, humanization); keep raw engine WAV.",
    )
    parser.add_argument(
        "--best-fit",
        action="store_true",
        help=(
            "If the Dhamma-talk rubric never passes within 10 blend iterations, keep the "
            "lowest-penalty processed mix. Without this flag, fall back to peak-normalized "
            "raw TTS only (no wet mastering)."
        ),
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List voices supported by your local system speech engine and exit.",
    )
    return parser


def _show_voices(tts_model: str) -> int:
    with console.status("[bold cyan]Loading speech engine and fetching voices...[/bold cyan]"):
        engine = CoquiTTSEngine(model_name=tts_model)
        voices = engine.list_voices()
    render_markdown(
        "## Supported Voices\n\n"
        + "\n".join([f"- `{voice}`" for voice in voices])
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    render_banner()

    if args.list_voices:
        return _show_voices(args.tts_model)

    if not args.prompt:
        parser.error("--prompt is required unless --list-voices is used.")

    if not 60 <= args.speech_rate <= 220:
        parser.error("--speech-rate must be between 60 and 220.")

    try:
        config = load_config(
            output_dir=args.output_dir,
            openai_model=args.openai_model,
            tts_model_name=args.tts_model,
            speech_rate_wpm=args.speech_rate,
            master_audio=not args.no_master,
            audio_best_fit=args.best_fit,
        )
    except ValueError as exc:
        console.print(f"[bold red]Configuration Error:[/bold red] {exc}")
        return 2

    with console.status(
        "[bold cyan]Generating guidelines, transcript, speech, and mastering audio...[/bold cyan]"
    ):
        result = run_pipeline(
            config=config,
            prompt=args.prompt,
            voice=args.voice,
            duration_minutes=args.duration_minutes,
        )

    render_result(result)
    console.print("[bold green]Completed successfully.[/bold green]")
    return 0
