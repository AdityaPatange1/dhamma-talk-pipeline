"""Rich-based terminal rendering helpers."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from dhamma_pipeline.pipeline import PipelineResult

console = Console()


def render_banner() -> None:
    console.print(
        Panel.fit(
            "[bold green]Dhamma Talk Pipeline[/bold green]\n"
            "[dim]OpenAI text generation + Open-source voice synthesis[/dim]"
        )
    )


def render_markdown(text: str) -> None:
    console.print(Markdown(text))


def render_result(result: PipelineResult) -> None:
    table = Table(title="Pipeline Output")
    table.add_column("Artifact", style="cyan", no_wrap=True)
    table.add_column("Path", style="green")
    table.add_row("Title", result.title)
    table.add_row("Voice", result.voice)
    table.add_row("Word Count", str(result.word_count))
    table.add_row("Audio mode", result.audio_mode)
    table.add_row("JSON", str(result.json_path))
    table.add_row("Markdown", str(result.markdown_path))
    table.add_row("Audio", str(result.audio_path))
    console.print(table)
