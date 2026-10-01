"""Download and validate Ballroom, then write metadata for the experiment."""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import subprocess

import mirdata
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ("audio", "tempo", "beats")


def download_archive(remote, data_dir: Path) -> Path:
    """Resume a transfer, checking its checksum before accepting the archive."""
    archive = data_dir / (remote.destination_dir or "") / remote.filename
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    if not archive.exists():
        subprocess.run(
            [
                "curl", "-fsSL", "--continue-at", "-", "--retry", "3",
                "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "30",
                "--output", str(partial), remote.url,
            ],
            check=True,
        )
    candidate = archive if archive.exists() else partial
    with candidate.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "md5").hexdigest()
    if checksum != remote.checksum:
        raise ValueError(
            f"Archive checksum mismatch: {candidate}. "
            "Remove this corrupted archive and rerun to download it again."
        )
    if candidate == partial:
        partial.replace(archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ballroom",
        help="Dataset directory (default: <project>/data/ballroom).",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Validate without downloading audio or annotations."
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    dataset = mirdata.initialize("ballroom", data_home=str(data_dir), version="1.0")

    if not args.verify_only:
        # mirdata uses the versioned index and published MD5 checksums.
        dataset.download(partial_download=["index"])
        missing, invalid = dataset.validate(verbose=False)
        if any(invalid.values()):
            dataset.validate()
            raise SystemExit("Existing dataset files are corrupted; validation failed.")
        missing_paths = [
            Path(path)
            for paths in missing.get("tracks", {}).values()
            for path in paths
        ]
        for resource in RESOURCES:
            destination = data_dir / dataset.remotes[resource].destination_dir
            if any(destination in path.parents for path in missing_paths):
                print(f"Downloading missing {resource} to {destination}", flush=True)
                download_archive(dataset.remotes[resource], data_dir)
                # Keep archives so an interrupted extraction can be rerun.
                dataset.download(partial_download=[resource], cleanup=False)

    missing, invalid = dataset.validate(verbose=False)
    if any(missing.values()) or any(invalid.values()):
        dataset.validate()
        raise SystemExit("Dataset is incomplete or corrupted; validation failed.")

    tracks = dataset.load_tracks()
    if len(tracks) != 698:
        raise SystemExit(f"Expected 698 excerpts, found {len(tracks)}.")

    rows = []
    for track_id, track in sorted(tracks.items()):
        tempo = track.tempo
        if tempo is None or not math.isfinite(tempo) or tempo <= 0:
            raise SystemExit(f"Invalid reference tempo for {track_id}: {tempo}")
        info = sf.info(track.audio_path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise SystemExit(f"Empty or invalid audio for {track_id}.")
        rows.append(
            {
                "track_id": track_id,
                "genre": track.genre,
                "audio_path": Path(track.audio_path).relative_to(data_dir).as_posix(),
                "tempo_path": Path(track.tempo_path).relative_to(data_dir).as_posix(),
                "beats_path": Path(track.beats_path).relative_to(data_dir).as_posix(),
                "reference_bpm": tempo,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.duration, 6),
            }
        )

    manifest = data_dir / "metadata.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verified {len(rows)} excerpts and their tempo/beat annotations.")
    print(f"Dataset: {data_dir}")
    print(f"Metadata: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Dataset preparation failed: {error}") from error
