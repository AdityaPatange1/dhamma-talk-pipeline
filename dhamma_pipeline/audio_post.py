"""Mixing, mastering, and rubric-driven selection for TTS (Dhamma talk friendly)."""

from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal


class AudioMasteringError(RuntimeError):
    """Raised when mastering cannot read or write audio."""


@dataclass(frozen=True)
class MasteringProfile:
    """Legacy profile holder (gentle chain uses fixed internal values)."""

    ceiling: float = 0.97


@dataclass
class DhammaTalkRubricResult:
    """Programmatic checks for calm, intelligible speech (no listening required)."""

    passed: bool
    score: float
    details: dict[str, float | bool] = field(default_factory=dict)


def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return np.mean(x, axis=1)
    raise AudioMasteringError("Unsupported channel layout for mastering.")


def peak_normalize(y: np.ndarray, target_peak: float = 0.97) -> np.ndarray:
    """Scale so max |y| == target_peak (preserves dynamics below clipping)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y))) + 1e-12
    return np.clip((y / peak) * target_peak, -1.0, 1.0)


def _biquad_lowshelf(
    fc: float, gain_db: float, s: float, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ Audio EQ Cookbook: lowshelf."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / s - 1.0) + 2.0)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
    a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a_coef = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a_coef


def _biquad_peaking(
    fc: float, gain_db: float, q: float, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ Audio EQ Cookbook: peaking EQ."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sample_rate
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a_lin

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a_coef = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a_coef


def gentle_master_mono(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Light-touch chain: clarity without mangling timbre (no vibrato / comb / heavy squash)."""
    x = _to_mono_float(samples)
    if x.size == 0:
        return x.astype(np.float32)
    if x.size < 512:
        return peak_normalize(x, 0.97).astype(np.float32)

    peak_in = float(np.max(np.abs(x))) + 1e-12
    x = x / peak_in
    nyq = 0.5 * sample_rate

    # Mild HPF — rumble only
    hp_norm = min(0.499, 58.0 / nyq)
    sos_hp = signal.butter(2, hp_norm, btype="highpass", output="sos")
    y = signal.sosfiltfilt(sos_hp, x)

    # Small warmth + small presence (speech intelligibility)
    b_ls, a_ls = _biquad_lowshelf(125.0, 1.1, 0.71, sample_rate)
    sos_ls = signal.tf2sos(b_ls, a_ls)
    y = signal.sosfiltfilt(sos_ls, y)

    b_pk, a_pk = _biquad_peaking(2200.0, 0.75, 1.05, sample_rate)
    sos_pk = signal.tf2sos(b_pk, a_pk)
    y = signal.sosfiltfilt(sos_pk, y)

    # Near-linear warmth only
    y = np.tanh(1.04 * y) / np.tanh(1.04)

    return peak_normalize(y, 0.97).astype(np.float32)


def _analysis_segment(y: np.ndarray, sample_rate: int) -> np.ndarray:
    """Use up to ~90s for rubric metrics (stable, fast enough)."""
    y = _to_mono_float(np.asarray(y, dtype=np.float64))
    max_n = min(y.size, int(sample_rate * 90))
    return y[:max_n]


def spectral_centroid_hz(y: np.ndarray, sample_rate: int) -> float:
    seg = _analysis_segment(y, sample_rate)
    if seg.size < 512:
        return 1500.0
    nperseg = min(4096, max(256, seg.size // 8))
    f, pxx = signal.welch(seg, sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    pxx = np.maximum(pxx, 1e-20)
    return float(np.sum(f * pxx) / np.sum(pxx))


def hf_energy_ratio(y: np.ndarray, sample_rate: int) -> float:
    """Share of PSD above 7 kHz (harsh / artifact detector)."""
    seg = _analysis_segment(y, sample_rate)
    if seg.size < 512:
        return 0.0
    nperseg = min(4096, max(256, seg.size // 8))
    f, pxx = signal.welch(seg, sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    pxx = np.maximum(pxx, 1e-20)
    total = float(np.sum(pxx)) + 1e-20
    hf = float(np.sum(pxx[f >= 7000.0]))
    return hf / total


def evaluate_dhamma_talk_rubric(y: np.ndarray, sample_rate: int) -> DhammaTalkRubricResult:
    """Heuristic rubric for recorded Dhamma-style speech (calm, clear, natural)."""
    y = _to_mono_float(np.asarray(y, dtype=np.float64))
    details: dict[str, float | bool] = {}
    penalties: list[float] = []

    if y.size < 64:
        details["too_short"] = True
        return DhammaTalkRubricResult(False, 1e6, details)

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    details["peak"] = peak
    details["rms"] = rms

    # Loudness: not inaudible, not smashed
    if peak < 0.18:
        penalties.append((0.18 - peak) * 80.0)
        details["peak_ok"] = False
    else:
        details["peak_ok"] = True

    if rms < 0.028:
        penalties.append((0.028 - rms) * 120.0)
        details["rms_ok"] = False
    elif rms > 0.32:
        penalties.append((rms - 0.32) * 40.0)
        details["rms_ok"] = False
    else:
        details["rms_ok"] = True

    crest_db = 20.0 * np.log10((peak + 1e-12) / (rms + 1e-12))
    details["crest_db"] = float(crest_db)
    # Too low = over-limited / mushy; too high = odd or very dynamic (less common for TTS)
    if crest_db < 6.5:
        penalties.append((6.5 - crest_db) * 2.5)
        details["crest_ok"] = False
    elif crest_db > 26.0:
        penalties.append((crest_db - 26.0) * 1.0)
        details["crest_ok"] = False
    else:
        details["crest_ok"] = True

    sc = spectral_centroid_hz(y, sample_rate)
    details["spectral_centroid_hz"] = sc
    if sc < 320.0:
        penalties.append((320.0 - sc) * 0.05)
        details["centroid_ok"] = False
    elif sc > 7200.0:
        penalties.append((sc - 7200.0) * 0.02)
        details["centroid_ok"] = False
    else:
        details["centroid_ok"] = True

    hf = hf_energy_ratio(y, sample_rate)
    details["hf_ratio"] = hf
    if hf > 0.28:
        penalties.append((hf - 0.28) * 60.0)
        details["hf_ok"] = False
    else:
        details["hf_ok"] = True

    clip_frac = float(np.mean(np.abs(y) >= 0.998))
    details["clip_fraction"] = clip_frac
    if clip_frac > 2e-4:
        penalties.append(clip_frac * 5000.0)
        details["clip_ok"] = False
    else:
        details["clip_ok"] = True

    score = float(sum(p * p for p in penalties))
    passed = len(penalties) == 0
    details["passed"] = passed
    return DhammaTalkRubricResult(passed=passed, score=score, details=details)


def write_riff_pcm16_mono_wav(
    output_path: Path,
    samples_mono: np.ndarray,
    sample_rate: int,
) -> None:
    """Write classic Microsoft PCM WAV (format 1) for maximum player compatibility (e.g. macOS afplay)."""
    y = np.asarray(samples_mono, dtype=np.float64).ravel()
    if y.size == 0:
        raise AudioMasteringError("Cannot write empty WAV.")
    y = np.clip(y, -1.0, 1.0)
    pcm = np.round(y * 32767.0).astype("<i2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def finalize_wav_for_playback(path: Path) -> None:
    """Re-read any supported WAV and rewrite as RIFF PCM16 mono (fixes afplay 'wht?' on extensible WAV)."""
    try:
        import soundfile as sf  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise AudioMasteringError(
            "soundfile is required. Install with `pip install -r requirements.txt`."
        ) from exc

    data, sr = sf.read(str(path), always_2d=False)
    sr = int(sr)
    mono = _to_mono_float(np.asarray(data, dtype=np.float64))
    peak = float(np.max(np.abs(mono))) + 1e-12
    if peak > 1.0:
        mono = mono / peak
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix=".afplay-", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        write_riff_pcm16_mono_wav(tmp_path, mono, sr)
        os.replace(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def master_audio_with_rubric(
    raw_wav_path: Path,
    output_wav_path: Path,
    *,
    best_fit: bool = True,
    max_iterations: int = 10,
) -> tuple[str, dict[str, float | int | str | bool]]:
    """Blend dry (normalized TTS) with gentle master; pick first rubric pass or best-fit / raw."""
    try:
        import soundfile as sf  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise AudioMasteringError(
            "soundfile is required for mastering. Install with `pip install -r requirements.txt`."
        ) from exc

    data, sr = sf.read(str(raw_wav_path), always_2d=False)
    sr = int(sr)
    mono = _to_mono_float(np.asarray(data, dtype=np.float64))

    dry = peak_normalize(mono, 0.97).astype(np.float64)
    wet = gentle_master_mono(mono, sr).astype(np.float64)

    best_score = float("inf")
    best_blend = 0.0
    best_wave: np.ndarray | None = None
    meta: dict[str, float | int | str | bool] = {}

    n = max(1, min(max_iterations, 50))
    for k in range(n):
        alpha = k / max(n - 1, 1)
        mixed = dry * (1.0 - alpha) + wet * float(alpha)
        mixed = peak_normalize(mixed, 0.97)

        rubric = evaluate_dhamma_talk_rubric(mixed, sr)
        if rubric.score < best_score:
            best_score = rubric.score
            best_blend = alpha
            best_wave = mixed.copy()

        if rubric.passed:
            write_riff_pcm16_mono_wav(output_wav_path, mixed, sr)
            meta.update(
                {
                    "mode": "rubric-pass",
                    "wet_blend": float(alpha),
                    "iteration": k + 1,
                    "rubric_score": rubric.score,
                }
            )
            return "rubric-pass", meta

    assert best_wave is not None

    if best_fit:
        out = peak_normalize(best_wave, 0.97)
        write_riff_pcm16_mono_wav(output_wav_path, out, sr)
        meta.update(
            {
                "mode": "best-fit",
                "wet_blend": float(best_blend),
                "rubric_score": float(best_score),
                "iterations": n,
            }
        )
        return "best-fit", meta

    # Raw fallback: normalized dry only (no EQ chain)
    out = dry.astype(np.float64)
    write_riff_pcm16_mono_wav(output_wav_path, out, sr)
    meta.update({"mode": "raw-fallback", "wet_blend": 0.0, "rubric_score": float(best_score)})
    return "raw-fallback", meta


def write_raw_normalized_wav(raw_wav_path: Path, output_wav_path: Path) -> None:
    """No mastering: peak-normalized dry speech only."""
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    data, sr = sf.read(str(raw_wav_path), always_2d=False)
    sr = int(sr)
    mono = _to_mono_float(np.asarray(data, dtype=np.float64))
    out = peak_normalize(mono, 0.97)
    write_riff_pcm16_mono_wav(output_wav_path, out, sr)


def master_wav_file(
    input_path: Path,
    output_path: Path,
    profile: MasteringProfile | None = None,
) -> None:
    """Single-shot gentle master (legacy API; prefer master_audio_with_rubric in pipeline)."""
    _ = profile
    try:
        import soundfile as sf  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise AudioMasteringError(
            "soundfile is required for mastering. Install with `pip install -r requirements.txt`."
        ) from exc

    data, sr = sf.read(str(input_path), always_2d=False)
    sr = int(sr)
    mastered = gentle_master_mono(data, sr)
    write_riff_pcm16_mono_wav(output_path, mastered, sr)


# Legacy name used by tests / imports
def master_mono_samples(
    samples: np.ndarray,
    sample_rate: int,
    profile: MasteringProfile | None = None,
) -> np.ndarray:
    _ = profile
    return gentle_master_mono(samples, sample_rate)
