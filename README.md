# Dhamma Talk Pipeline

A production-oriented Python project that generates complete Dhamma talks from a prompt, writes structured transcript artifacts, and synthesizes full audio recitations using an open-source text-to-speech model.

This project is intentionally provided as a **normal Python project layout** (not packaged as a pip-distributed module): one main executable at repository root and implementation code inside a dedicated folder.

## Project Highlights

- Prompt-driven Dhamma talk generation with OpenAI
- Structured guideline generation before transcript generation
- Transcript export in both JSON and Markdown
- Open-source TTS audio synthesis (`pyttsx3` system voices)
- Multi-voice support via `--voice`
- Rich terminal UI (banners, markdown, spinner, output table)
- Detailed CLI help and usage examples

## Folder Structure

```text
dhamma-talk-pipeline/
├── dhamma_talk_pipeline.py        # Main entry point (the single root-level app file)
├── dhamma_pipeline/               # Implementation package directory
│   ├── __init__.py
│   ├── cli.py                     # Argument parsing and CLI flow
│   ├── config.py                  # Environment and runtime config
│   ├── llm.py                     # OpenAI integration for generation
│   ├── models.py                  # Shared dataclasses
│   ├── pipeline.py                # End-to-end orchestration
│   ├── transcript.py              # JSON/Markdown writing helpers
│   ├── tts_engine.py              # Open-source TTS integration
│   └── ui.py                      # Rich rendering helpers
├── outputs/                       # Generated artifacts (created automatically)
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## Functional Specification Coverage

### 1) Prompt-based CLI generation into `outputs/`

You can run:

```bash
python dhamma_talk_pipeline.py --prompt "Dhamma talk on mindfulness of mind"
```

This generates:

- A full talk transcript in JSON
- A full talk transcript in Markdown
- A complete audio recitation file (`.wav`)

All artifacts are saved in the configured output directory (`outputs/` by default).

### 2) Required pipeline order

The project follows this execution flow:

1. Generate **guidelines/model plan** containing:
   - topic
   - presentation style
   - topics covered
   - precepts answered
   - audience profile / tone / outline / duration target
2. Generate full transcript text from the produced guideline plan
3. Save transcript as JSON + Markdown
4. Synthesize entire transcript as audio using open-source TTS

### 3) Voice selection via `--voice`

Use:

```bash
python dhamma_talk_pipeline.py --prompt "..." --voice default
```

Voice values correspond to local system voice names/IDs exposed by `pyttsx3`.  
Use `--list-voices` to inspect available voices.

### 4) Detailed help command

Use:

```bash
python dhamma_talk_pipeline.py --help
```

The CLI includes detailed option descriptions, examples, and usage notes.

## Prerequisites

- Python 3.10+ (3.12 supported)
- Internet connection (OpenAI API access)
- OpenAI API key
- System dependencies for audio/model tooling (platform-dependent)

## Installation

### 1. Clone and enter repository

```bash
git clone <your-repo-url>
cd dhamma-talk-pipeline
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Set:

```bash
export OPENAI_API_KEY="sk-..."
```

You can also source your `.env` using your preferred shell tooling.

## Usage

## Generate a Dhamma talk

```bash
python dhamma_talk_pipeline.py \
  --prompt "Dhamma talk on mindfulness of mind" \
  --voice default \
  --duration-minutes 20
```

## List available voices

```bash
python dhamma_talk_pipeline.py --list-voices
```

## Use custom models/directories

```bash
python dhamma_talk_pipeline.py \
  --prompt "Dhamma talk on loving-kindness and non-harming" \
  --openai-model gpt-4.1-mini \
  --tts-model pyttsx3-system \
  --output-dir outputs
```

## CLI Reference

| Option | Description |
|---|---|
| `--prompt` | Input prompt for the Dhamma talk request |
| `--voice` | Local system voice name or ID |
| `--duration-minutes` | Target talk duration used in planning |
| `--output-dir` | Output directory for artifacts |
| `--openai-model` | OpenAI model for planning + transcript generation |
| `--tts-model` | TTS backend label (default `pyttsx3-system`) |
| `--list-voices` | Print available voices and exit |
| `--help` | Detailed usage help |

## Output Artifacts

For a topic slug like `mindfulness-of-mind`, generated files are:

- `outputs/mindfulness-of-mind.json`
- `outputs/mindfulness-of-mind.md`
- `outputs/mindfulness-of-mind.wav`

### JSON includes

- Prompt metadata
- Guidelines model details
- Full transcript text
- Model and voice metadata
- Generation timestamp

### Markdown includes

- Human-readable metadata
- Guideline sections
- Full transcript narrative

### Audio includes

- Full recitation of generated transcript in selected voice

## Architecture Overview

### `dhamma_pipeline/llm.py`

- Integrates OpenAI chat completions
- Stage 1: structured guidelines JSON
- Stage 2: full transcript generation from guidelines
- Raises explicit generation errors on invalid/empty outputs

### `dhamma_pipeline/tts_engine.py`

- Uses `pyttsx3` (open source) with local system voices
- Supports voice listing and voice selection
- Writes audio output via platform speech engine

### `dhamma_pipeline/pipeline.py`

- Orchestrates complete generation flow
- Guarantees output writes in deterministic locations
- Returns structured result object for UI rendering

### `dhamma_pipeline/cli.py`

- Clean argparse-based interface
- Rich spinner/status output and result table
- Validation for required prompt/config

## Reliability and Production Considerations

This project is production-oriented in structure, but you should still apply environment-level hardening for your deployment context:

- Secret management: never hardcode API keys
- Runtime observability: add logs and telemetry where required
- Retries/backoff: add network retry strategy for API calls
- Input safety: enforce prompt policies appropriate to your use case
- Cost control: constrain model use, temperature, and token lengths
- Deterministic outputs: version-pin models where possible

## Troubleshooting

### `OPENAI_API_KEY is missing`

Set your environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### Voice not found

Run:

```bash
python dhamma_talk_pipeline.py --list-voices
```

Then pass a valid voice with `--voice`.

### TTS install issues

`pyttsx3` uses OS speech services. If voices are missing, install/add system voices in your OS settings and rerun `--list-voices`.

## Security and Ethical Notes

- Generated spiritual content should be reviewed by qualified teachers before formal publication or instruction use.
- Ensure you comply with OpenAI usage policies.
- Respect community traditions and cultural context in generated discourse.

## Maintenance Guide

### Update dependencies

```bash
pip install -U -r requirements.txt
```

### Validate CLI quickly

```bash
python dhamma_talk_pipeline.py --help
python dhamma_talk_pipeline.py --list-voices
```

### Suggested extensions

- Add SSML controls where supported by selected TTS backend
- Add chapter timestamps in transcript markdown
- Add caching for repeated prompts
- Add test suite with mocked LLM and TTS calls
- Add CI checks (format/lint/type/test)

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make focused commits with clear messages
4. Add or update tests where relevant
5. Open a pull request with rationale and sample outputs

Recommended local quality checks:

```bash
python -m compileall .
```

## License

This project is licensed under the terms of the included `LICENSE` file.

## Quick Start Recap

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python dhamma_talk_pipeline.py --prompt "Dhamma talk on mindfulness of mind" --voice default
```
