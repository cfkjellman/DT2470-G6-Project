"""Run Demucs on the 30-song pilot or the 20-song validation sample."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from time import perf_counter

import soundfile as sf

from ballroom_dataset import read_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "track_id", "dance_style", "original_audio", "vocals_audio", "instrumental_audio",
    "sample_rate", "duration_seconds", "original_rms", "vocals_rms",
    "vocal_energy_ratio", "vocal_energy_db", "separation_seconds", "human_vocals", "notes",
]


def fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def separation_config() -> dict:
    return {
        "model": "htdemucs", "device": "cpu", "threads": 4,
        "shifts": 0, "overlap": 0.25, "segment_seconds": 7.8,
        "instrumental_method": "sum_non_vocal_stems", "wav_subtype": "FLOAT",
        "demucs_version": version("demucs"), "torch_version": version("torch"),
        "numpy_version": version("numpy"), "soundfile_version": version("soundfile"),
    }


def load_model():
    import torch
    from demucs.pretrained import get_model

    torch.set_num_threads(4)
    torch.hub.set_dir(str(PROJECT_ROOT / "data/demucs_models"))
    model = get_model("htdemucs")
    model.eval()
    return model


def cached_result(directory: Path, config: dict, audio_hash: str) -> dict | None:
    receipt = directory / "result.json"
    if not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    if saved["config"] != config or saved["source_sha256"] != audio_hash:
        return None
    for stem in ("vocals", "instrumental"):
        path = directory / f"{stem}.wav"
        if not path.exists() or fingerprint(path) != saved["stem_sha256"][stem]:
            return None
    return saved["row"]


def save_result(directory: Path, row: dict, config: dict, audio_hash: str):
    receipt = {
        "config": config, "source_sha256": audio_hash, "row": row,
        "stem_sha256": {
            stem: fingerprint(directory / f"{stem}.wav")
            for stem in ("vocals", "instrumental")
        },
    }
    temporary = directory / "result.json.part"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(directory / "result.json")


def write_review(path: Path, rows: list[dict], labels: dict):
    for row in rows:
        previous = labels.get(row["track_id"], {})
        row["human_vocals"] = previous.get("human_vocals", "")
        row["notes"] = previous.get("notes", "")
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def separate(model, audio: Path, directory: Path, config: dict) -> dict:
    import torch
    from demucs.apply import apply_model
    from demucs.audio import convert_audio

    original, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
    wav = convert_audio(
        torch.from_numpy(original.T.copy()), sample_rate, model.samplerate, model.audio_channels,
    )
    mixture = wav.clone()
    reference = wav.mean(0)
    offset, scale = reference.mean(), reference.std()
    if scale <= 1e-8:
        raise ValueError(f"Cannot separate silent audio: {audio}")
    wav = (wav - offset) / scale
    started = perf_counter()
    with torch.inference_mode():
        sources = apply_model(
            model, wav[None], device="cpu", shifts=0, split=True,
            overlap=config["overlap"], segment=config["segment_seconds"],
            num_workers=0, progress=False,
        )[0]
    elapsed = perf_counter() - started
    sources = sources * scale + offset
    vocal_index = model.sources.index("vocals")
    vocals = sources[vocal_index]
    instrumental = sources[[i for i in range(len(model.sources)) if i != vocal_index]].sum(0)

    # Float WAV files keep the stem levels unchanged for energy measurements.
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(directory / "vocals.wav", vocals.T.numpy(), model.samplerate, subtype="FLOAT")
    sf.write(directory / "instrumental.wav", instrumental.T.numpy(), model.samplerate, subtype="FLOAT")
    original_energy = float(mixture.double().square().mean())
    vocal_energy = float(vocals.double().square().mean())
    ratio = vocal_energy / original_energy
    return {
        "original_audio": relative(audio),
        "vocals_audio": relative(directory / "vocals.wav"),
        "instrumental_audio": relative(directory / "instrumental.wav"),
        "sample_rate": model.samplerate,
        "duration_seconds": mixture.shape[-1] / model.samplerate,
        "original_rms": math.sqrt(original_energy),
        "vocals_rms": math.sqrt(vocal_energy),
        "vocal_energy_ratio": ratio,
        "vocal_energy_db": 10 * math.log10(ratio) if ratio > 0 else float("-inf"),
        "separation_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true", help="Use the additional 20-song sample.")
    args = parser.parse_args()
    filename = "ballroom_pilot_validation.csv" if args.validation else "ballroom_pilot.csv"
    folder = "demucs_validation" if args.validation else "demucs_pilot"
    pilot = read_csv(PROJECT_ROOT / "metadata" / filename)
    splits = {row["track_id"]: row for row in read_csv(PROJECT_ROOT / "metadata/ballroom_splits.csv")}
    if any(splits[row["track_id"]]["split"] != "development" for row in pilot):
        raise ValueError("The pilot must only contain development excerpts.")
    metadata_path = PROJECT_ROOT / "data/ballroom/metadata.csv"
    metadata = {row["track_id"]: row for row in read_csv(metadata_path)}
    output = PROJECT_ROOT / "results" / folder
    output.mkdir(parents=True, exist_ok=True)
    review = output / "review.csv"
    results = {row["track_id"]: row for row in read_csv(review)} if review.exists() else {}
    labels = {row["track_id"]: row for row in read_csv(PROJECT_ROOT / "metadata/ballroom_vocal_reviews.csv")}
    labels.update(results)
    config = separation_config()
    model = None

    for index, excerpt in enumerate(pilot, start=1):
        track_id = excerpt["track_id"]
        audio = metadata_path.parent / metadata[track_id]["audio_path"]
        directory = output / track_id
        audio_hash = fingerprint(audio)
        row = cached_result(directory, config, audio_hash)
        if row is None:
            if model is None:
                model = load_model()
            row = {"track_id": track_id, "dance_style": excerpt["dance_style"]}
            row.update(separate(model, audio, directory, config))
            save_result(directory, row, config, audio_hash)
        results[track_id] = row
        write_review(review, list(results.values()), labels)
        print(f"[{index}/{len(pilot)}] {track_id}", flush=True)
    print(f"Saved listening scores and labels to {review}")


if __name__ == "__main__":
    main()
