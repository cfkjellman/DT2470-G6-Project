"""Create a reproducible Ballroom split, keeping duplicate recordings together."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path
import random

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DANCE_STYLES = {
    "chachacha", "jive", "quickstep", "rumba", "samba", "tango",
    "viennesewaltz", "waltz",
}


def audio_fingerprint(path: Path) -> str:
    """Hash decoded audio, including sample rate and channel count."""
    digest = hashlib.sha256()
    with sf.SoundFile(path) as audio:
        digest.update(f"{audio.samplerate}:{audio.channels}:".encode())
        for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
            digest.update(block.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def create_splits(metadata: Path, duplicate_pairs: Path, development_size: int, seed: int):
    with metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tracks = {row["track_id"]: row for row in rows}
    if len(rows) != 698 or len(tracks) != 698:
        raise ValueError("Expected 698 distinct excerpt IDs in the metadata.")
    if not 0 < development_size < len(tracks):
        raise ValueError("Development size must be between 1 and 697.")

    styles = {}
    for track_id, row in tracks.items():
        genre = row["genre"].lower()
        style = "rumba" if genre.startswith("rumba-") else genre
        if style not in DANCE_STYLES:
            raise ValueError(f"Unknown dance style for {track_id}: {genre}")
        styles[track_id] = style

    # Connected components keep every related pair together, including chains.
    parents = {track_id: track_id for track_id in tracks}

    def find(track_id):
        while parents[track_id] != track_id:
            parents[track_id] = parents[parents[track_id]]
            track_id = parents[track_id]
        return track_id

    def join(first, second):
        if first not in tracks or second not in tracks:
            raise ValueError(f"Duplicate pair has an unknown ID: {first}, {second}")
        first_root, second_root = find(first), find(second)
        parents[max(first_root, second_root)] = min(first_root, second_root)

    with duplicate_pairs.open(newline="", encoding="utf-8") as handle:
        for pair in csv.DictReader(handle):
            join(pair["track_id_a"], pair["track_id_b"])

    fingerprints = {}
    for track_id in sorted(tracks):
        path = Path(tracks[track_id]["audio_path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Audio path must be relative to the dataset: {path}")
        fingerprint = audio_fingerprint(metadata.parent / path)
        if fingerprint in fingerprints:
            join(track_id, fingerprints[fingerprint])
        else:
            fingerprints[fingerprint] = track_id

    groups = defaultdict(list)
    for track_id in sorted(tracks):
        groups[find(track_id)].append(track_id)

    # Largest-remainder allocation preserves proportions and totals exactly.
    totals = Counter(styles.values())
    quotas = {
        style: count * development_size // len(tracks)
        for style, count in totals.items()
    }
    remainder_order = sorted(
        totals,
        key=lambda style: (-(totals[style] * development_size % len(tracks)), style),
    )
    for style in remainder_order[:development_size - sum(quotas.values())]:
        quotas[style] += 1

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    development = set()
    selected_counts = Counter()
    for group_id in group_ids:
        members = groups[group_id]
        counts = Counter(styles[track_id] for track_id in members)
        if all(selected_counts[style] + count <= quotas[style] for style, count in counts.items()):
            development.update(members)
            selected_counts.update(counts)
    if selected_counts != Counter(quotas):
        raise ValueError(
            "Cannot meet the requested style quotas without separating duplicates. "
            "Choose a different development size."
        )

    assignments = [
        {
            "track_id": track_id,
            "split": "development" if track_id in development else "test",
            "dance_style": styles[track_id],
            "duplicate_group": find(track_id),
        }
        for track_id in sorted(tracks)
    ]
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/ballroom/metadata.csv")
    parser.add_argument("--duplicate-pairs", type=Path, default=PROJECT_ROOT / "metadata/ballroom_duplicate_pairs.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "metadata/ballroom_splits.csv")
    parser.add_argument("--development-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assignments = create_splits(
        args.metadata.expanduser().resolve(),
        args.duplicate_pairs.expanduser().resolve(),
        args.development_size,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)

    counts = Counter(row["split"] for row in assignments)
    group_sizes = Counter(row["duplicate_group"] for row in assignments)
    print(f"Development: {counts['development']}; test: {counts['test']}; seed: {args.seed}")
    print(f"Duplicate groups: {sum(size > 1 for size in group_sizes.values())}")
    for style in sorted(DANCE_STYLES):
        style_counts = Counter(row["split"] for row in assignments if row["dance_style"] == style)
        print(f"{style}: {style_counts['development']} development, {style_counts['test']} test")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Split creation failed: {error}") from error
