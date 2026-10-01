# DT2470-G6-Project
G6 project for course DT2470 Music Informatics at KTH

## Project
The goal of this project is to investigate whether removing vocals from a song improves tempo estimation. We will use the Ballroom dataset, which contains 30-second music excerpts with reference tempo annotations. Using the pretrained Demucs model to remove vocals and our own tempo estimation algorithm, we will then compare tempo estimates from the original audio and the instrumental versions against the reference tempos. This way we can see if removing the vocals have any effect on tempo estimation

## Setup
Follow the steps below to download the dataset and set up python environment.
### Set up Python with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). On macOS
with:

```bash
brew install uv
```

From the repository directory:

```bash
uv sync --locked
```

### Download the Ballroom dataset

```bash
bash scripts/download_dataset.sh
```

This command will download the dataset in the following folder structure.

```text
data/ballroom/
├── metadata.csv
└── B_1.0/
    ├── audio/                  # WAV files grouped by dance style
    └── annotations/
        ├── tempo/              # Reference BPM files
        └── beats/              # Beat timestamps and bar positions
```

To check an existing download and regenerate its metadata without downloading
audio or annotations:

```bash
bash scripts/download_dataset.sh --verify-only
```

The Bash wrapper works from any working directory. For a shell without Bash,
run the Python helper directly from the repository:

```bash
uv run --locked python scripts/download_dataset.py
```

`data/`, `results/`, and `.venv/` are ignored by Git. Store audio locally and
share the downloader and configuration through Git.

## Development and test sets

The fixed split is saved in `metadata/ballroom_splits.csv`, which is tracked in
Git. It assigns all 698 excerpts to **140 development** and **558 test** excerpts
using random seed **42**. No audio files are moved or copied.

The split approximately preserves the proportions of the eight dance styles.
The three Rumba folders are treated as one style for sampling. Duplicate
recordings are kept in the same split, even when their dance labels differ.
`metadata/ballroom_duplicate_pairs.csv` records the 13 pairs published by the
[Ballroom annotation maintainers](https://github.com/CPJKU/BallroomAnnotations#description).
The generator also groups identical decoded audio. This covers known recording
replicas and exact matches; unidentified overlapping recordings could remain.

To reproduce the split after downloading the dataset:

```bash
uv run --locked python scripts/create_splits.py
```

Join `metadata/ballroom_splits.csv` to `data/ballroom/metadata.csv` using
`track_id` to find the audio and reference annotations. The split CSV includes
`dance_style` and `duplicate_group` so related excerpts remain identifiable.