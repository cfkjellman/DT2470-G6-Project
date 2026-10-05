"""Separate Ballroom audio and save the development and test set CSVs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import shutil

from ballroom_dataset import load_selection, read_csv, vocal_activity
from demucs_pilot import cached_result, fingerprint, load_model, relative, save_result, separate, separation_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA = PROJECT_ROOT / "metadata"
OUTPUT = PROJECT_ROOT / "data/ballroom/separated"
PILOT_FOLDERS = [PROJECT_ROOT / "results/demucs_pilot", PROJECT_ROOT / "results/demucs_validation"]
CONFIG = separation_config()
SELECTION = load_selection()
MANIFEST_FIELDS = [
    "track_id", "split", "dance_style", "duplicate_group", "original_audio",
    "instrumental_audio", "vocals_audio", "reference_bpm", "duration_seconds",
    "estimated_vocal_seconds", "estimated_vocal_fraction", "human_vocals",
]
_MODEL = None


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_excerpts():
    metadata_path = PROJECT_ROOT / "data/ballroom/metadata.csv"
    source = {row["track_id"]: row for row in read_csv(metadata_path)}
    splits = read_csv(METADATA / "ballroom_splits.csv")
    group_splits = {}
    excerpts = []
    for row in splits:
        group = row["duplicate_group"]
        if group in group_splits and group_splits[group] != row["split"]:
            raise ValueError(f"Recording group crosses development and test: {group}")
        group_splits[group] = row["split"]
        audio = metadata_path.parent / source[row["track_id"]]["audio_path"]
        excerpts.append({**row, "audio": audio})
    return excerpts, source


def prepare_track(excerpt):
    global _MODEL
    audio = excerpt["audio"]
    track_id = excerpt["track_id"]
    directory = OUTPUT / track_id
    audio_hash = fingerprint(audio)
    row = cached_result(directory, CONFIG, audio_hash)

    # Reuse pilot stems so the same excerpts do not need separating again.
    if row is None:
        for folder in PILOT_FOLDERS:
            previous = folder / track_id
            row = cached_result(previous, CONFIG, audio_hash)
            if row is None:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            for stem in ("vocals", "instrumental"):
                destination = directory / f"{stem}.wav"
                if destination.exists():
                    if fingerprint(destination) != fingerprint(previous / destination.name):
                        shutil.copyfile(previous / destination.name, destination)
                else:
                    try:
                        os.link(previous / destination.name, destination)
                    except OSError:
                        shutil.copy2(previous / destination.name, destination)
            break

    if row is None:
        if _MODEL is None:
            _MODEL = load_model()
        row = {"track_id": track_id, "dance_style": excerpt["dance_style"]}
        row.update(separate(_MODEL, audio, directory, CONFIG))

    row.update({
        "original_audio": relative(audio),
        "vocals_audio": relative(directory / "vocals.wav"),
        "instrumental_audio": relative(directory / "instrumental.wav"),
    })
    save_result(directory, row, CONFIG, audio_hash)
    row.update(vocal_activity(audio, directory / "vocals.wav", SELECTION))
    row.update({"split": excerpt["split"], "duplicate_group": excerpt["duplicate_group"]})
    return row


def save_development_check(completed, reviews):
    checks = []
    for track_id in sorted(reviews):
        row = completed[track_id]
        checks.append({
            "track_id": track_id,
            "human_vocals": reviews[track_id]["human_vocals"],
            "estimated_vocal_seconds": row["estimated_vocal_seconds"],
            "estimated_vocal_fraction": row["estimated_vocal_fraction"],
            "strong_vocal_candidate": row["strong_vocal_candidate"],
        })
    fields = ["track_id", "human_vocals", "estimated_vocal_seconds",
              "estimated_vocal_fraction", "strong_vocal_candidate"]
    write_csv(METADATA / "vocal_selection_development_check.csv", fields, checks)
    selected = [row for row in checks if row["strong_vocal_candidate"]]
    return {
        "reviewed_excerpts": len(checks),
        "selected_candidates": len(selected),
        "known_non_vocal_candidates": [row["track_id"] for row in selected if row["human_vocals"] == "no"],
    }


def save_datasets(completed, source, reviews):
    development, test, audit = [], [], []
    for track_id in sorted(completed):
        row = completed[track_id]
        row["reference_bpm"] = source[track_id]["reference_bpm"]
        row["human_vocals"] = reviews.get(track_id, {}).get("human_vocals", "")
        record = {field: row[field] for field in MANIFEST_FIELDS}
        if row["split"] == "development":
            development.append(record)
        else:
            audit.append({
                "track_id": track_id,
                "dance_style": row["dance_style"],
                "duplicate_group": row["duplicate_group"],
                "estimated_vocal_seconds": row["estimated_vocal_seconds"],
                "estimated_vocal_fraction": row["estimated_vocal_fraction"],
                "selected": row["strong_vocal_candidate"],
            })
            if row["strong_vocal_candidate"]:
                test.append(record)
    write_csv(METADATA / "development_set.csv", MANIFEST_FIELDS, development)
    write_csv(METADATA / "test_set.csv", MANIFEST_FIELDS, test)
    write_csv(METADATA / "test_selection.csv", list(audit[0]), audit)

    # Keep listening labels and notes when regenerating the review file.
    review_path = PROJECT_ROOT / "results/test_vocal_review/review.csv"
    old_reviews = {row["track_id"]: row for row in read_csv(review_path)} if review_path.exists() else {}
    review_rows = []
    for row in test:
        old = old_reviews.get(row["track_id"], {})
        review_rows.append({
            "track_id": row["track_id"], "dance_style": row["dance_style"],
            "original_audio": row["original_audio"], "vocals_audio": row["vocals_audio"],
            "human_vocals": old.get("human_vocals", ""), "notes": old.get("notes", ""),
        })
    write_csv(review_path, ["track_id", "dance_style", "original_audio", "vocals_audio", "human_vocals", "notes"], review_rows)
    summary = {
        "selection": SELECTION, "separation": CONFIG,
        "development_check": save_development_check(completed, reviews),
        "development_count": len(development), "test_pool_count": len(audit), "test_count": len(test),
        "test_status": "automatically selected substantial-vocal candidates; human verification is incomplete",
    }
    temporary = METADATA / "prepared_dataset.json.part"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(METADATA / "prepared_dataset.json")
    print(f"Saved {len(development)} development excerpts and {len(test)} test candidates.")


def main():
    summary_path = METADATA / "prepared_dataset.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["selection"] != SELECTION:
        raise ValueError("The vocal-selection rule changed. Keep the rule fixed for this experiment.")
    excerpts, source = load_excerpts()
    reviews = {row["track_id"]: row for row in read_csv(METADATA / "ballroom_vocal_reviews.csv")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_model()  # Download weights once before starting the two workers.
    completed = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_track, excerpt) for excerpt in excerpts]
        try:
            for future in as_completed(futures):
                row = future.result()
                completed[row["track_id"]] = row
                print(f"[{len(completed)}/{len(excerpts)}] {row['track_id']}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    save_datasets(completed, source, reviews)


if __name__ == "__main__":
    main()
