"""Access the prepared Ballroom development and confirmed strong-vocal test sets."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BallroomTrack:
    track_id: str
    dance_style: str
    duplicate_group: str
    split: str
    original_path: Path
    instrumental_path: Path
    vocals_path: Path
    reference_bpm: float
    duration_seconds: float
    estimated_vocal_seconds: float
    estimated_vocal_fraction: float
    human_vocals: str


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_selection() -> dict:
    path = PROJECT_ROOT / "metadata/vocal_selection.json"
    return json.loads(path.read_text(encoding="utf-8"))


def vocal_activity(original_path: Path, vocals_path: Path, selection: dict) -> dict:
    """Estimate active duration; windows partition the audio, including its tail."""
    import numpy as np
    import soundfile as sf

    original, sample_rate = sf.read(original_path, dtype="float64", always_2d=True)
    vocals, vocal_rate = sf.read(vocals_path, dtype="float64", always_2d=True)
    if sample_rate != vocal_rate or len(original) != len(vocals) or not len(original):
        raise ValueError(f"Original and vocal stem must have matching sample rates and lengths: {original_path}")
    original_energy = float(np.mean(original ** 2))
    window = max(1, round(sample_rate * selection["window_seconds"]))
    active_samples = 0
    for start in range(0, len(original), window):
        end = min(start + window, len(original))
        mixture_energy = float(np.mean(original[start:end] ** 2))
        if mixture_energy == 0 or mixture_energy < selection["minimum_original_window_energy_fraction"] * original_energy:
            continue
        vocal_energy = float(np.mean(vocals[start:end] ** 2))
        if vocal_energy / mixture_energy >= selection["minimum_window_energy_ratio"]:
            active_samples += end - start
    fraction = active_samples / len(original)
    return {
        "estimated_vocal_seconds": active_samples / sample_rate,
        "estimated_vocal_fraction": fraction,
        "strong_vocal_candidate": fraction >= selection["minimum_vocal_fraction"],
    }


def _get_set(split: str) -> list[BallroomTrack]:
    manifest = PROJECT_ROOT / "metadata" / f"{split}_set.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}. Run: uv run --locked python scripts/prepare_dataset.py"
        )
    tracks = []
    for row in read_csv(manifest):
        track = BallroomTrack(
            track_id=row["track_id"],
            dance_style=row["dance_style"],
            duplicate_group=row["duplicate_group"],
            split=row["split"],
            original_path=PROJECT_ROOT / row["original_audio"],
            instrumental_path=PROJECT_ROOT / row["instrumental_audio"],
            vocals_path=PROJECT_ROOT / row["vocals_audio"],
            reference_bpm=float(row["reference_bpm"]),
            duration_seconds=float(row["duration_seconds"]),
            estimated_vocal_seconds=float(row["estimated_vocal_seconds"]),
            estimated_vocal_fraction=float(row["estimated_vocal_fraction"]),
            human_vocals=row["human_vocals"],
        )
        for path in (track.original_path, track.instrumental_path, track.vocals_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run scripts/prepare_dataset.py locally.")
        tracks.append(track)
    return tracks


def get_development_set() -> list[BallroomTrack]:
    """Return all 140 development excerpts."""
    return _get_set("development")


def get_test_set() -> list[BallroomTrack]:
    """Return manually confirmed substantial-vocal test excerpts."""
    return _get_set("test")
