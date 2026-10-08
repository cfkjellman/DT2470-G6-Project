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

`data/`, `results/`, and `.venv/` are ignored by Git.

## Development and test sets

We split the 698 excerpts into 140 development excerpts and 558 reserved test
excerpts, using seed 42. The development set contains different dance styles
and songs both with and without vocals. Copies of the same recording stay in
the same split to avoid overlap. The known duplicate pairs come from the
[Ballroom annotation maintainers](https://github.com/CPJKU/BallroomAnnotations#description).

The split is saved in `metadata/ballroom_splits.csv`. To recreate it:

```bash
uv run --locked python scripts/create_splits.py
```

## Demucs development pilot

We first ran Demucs on 30 development excerpts, then another 20. We listened
to the originals and vocal stems to check whether vocal-stem energy could help
us find songs with vocals. The selections and our 50 listening labels are
saved in `metadata/`.

To reproduce both pilot batches:

```bash
uv run --locked python scripts/create_pilot.py
uv run --locked python scripts/demucs_pilot.py
uv run --locked python scripts/demucs_pilot.py --validation
```

## Prepare the project datasets

```bash
uv run --locked python scripts/prepare_dataset.py
```

This runs Demucs on the full dataset and saves the vocal and instrumental
audio in `data/ballroom/separated/`. It uses the CPU and reuses completed stems,
so an interrupted run can be resumed. Our full run took about 1.5 hours.

We use one-second windows to estimate how much of each excerpt contains
vocals. A window counts when vocal-stem energy is at least 18% of the original
energy, while very quiet windows are ignored. The settings are in
`metadata/vocal_selection.json` and stay fixed when selecting test excerpts.

The resulting sets are:

- `metadata/development_set.csv`: all 140 development excerpts.
- `metadata/test_set.csv`: 208 manually confirmed songs with strong vocals.
- `metadata/test_selection.csv`: the automatic selection and final inclusion
  for all 558 reserved test excerpts.

Demucs sometimes puts instruments in the vocal stem. We listened to all 219
candidates and kept songs with clear human vocals for roughly one-third of the
excerpt. We rejected nine violin excerpts and left out two uncertain songs with
very little singing. The labels and notes are in `metadata/test_vocal_reviews.csv`.
Preparation uses these labels, so rerunning it keeps the same confirmed test set.

## Load the datasets in Python

After `uv sync --locked`, use:

```python
from ballroom_dataset import get_development_set, get_test_set

development = get_development_set()
test = get_test_set()

track = test[0]
print(track.original_path, track.instrumental_path, track.reference_bpm)
```

Both functions return track information and audio paths, ready to use in our
Python code.
