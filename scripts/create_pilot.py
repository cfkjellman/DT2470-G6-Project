"""Select a reproducible, style-stratified Demucs pilot from development only."""

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def select_pilot(size: int, seed: int, excluded_groups: set) -> list[dict]:
    with (PROJECT_ROOT / "metadata/ballroom_splits.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    development = [
        row for row in rows
        if row["split"] == "development" and row["duplicate_group"] not in excluded_groups
    ]
    totals = Counter(row["dance_style"] for row in development)
    quotas = {style: count * size // len(development) for style, count in totals.items()}
    remainder_order = sorted(
        totals,
        key=lambda style: (-(totals[style] * size % len(development)), style),
    )
    for style in remainder_order[:size - sum(quotas.values())]:
        quotas[style] += 1

    by_style = defaultdict(list)
    for row in sorted(development, key=lambda row: row["track_id"]):
        by_style[row["dance_style"]].append(row)
    rng = random.Random(seed)
    selected = []
    seen_groups = set()
    for style in sorted(by_style):
        rng.shuffle(by_style[style])
        count = 0
        for row in by_style[style]:
            if count == quotas[style]:
                break
            if row["duplicate_group"] not in seen_groups:
                selected.append(row)
                seen_groups.add(row["duplicate_group"])
                count += 1
        if count != quotas[style]:
            raise ValueError("Not enough unique recordings to fill the pilot.")

    # Interleave styles so even a short timing run covers several styles.
    queues = defaultdict(list)
    for row in selected:
        queues[row["dance_style"]].append(row)
    pilot = []
    while len(pilot) < size:
        for style in sorted(queues):
            if queues[style]:
                pilot.append(queues[style].pop(0))
    return pilot


def save_pilot(filename: str, pilot: list[dict]) -> None:
    output = PROJECT_ROOT / "metadata" / filename
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "split", "dance_style", "duplicate_group"])
        writer.writeheader()
        writer.writerows(pilot)
    print(f"Saved {len(pilot)} development excerpts to {output.name}")


def main() -> None:
    pilot = select_pilot(30, 42, set())
    excluded_groups = {row["duplicate_group"] for row in pilot}
    validation = select_pilot(20, 43, excluded_groups)
    save_pilot("ballroom_pilot.csv", pilot)
    save_pilot("ballroom_pilot_validation.csv", validation)


if __name__ == "__main__":
    main()
