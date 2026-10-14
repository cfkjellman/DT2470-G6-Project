"""Tempo estimation for the vocal-removal project.

Pipeline for one excerpt (same steps as tempo_est_dev.ipynb):

    audio -> spectral novelty -> tempogram -> global tempo (BPM)

The novelty function follows the notebook exactly (STFT with 50 ms windows
and 25 ms hop, log compression with gamma = 1000, positive differences summed
over frequency, local-average subtraction over +-0.1 s). Original and
instrumental audio always go through the same parameters.

Use from Python:

    from tempo_estimation import estimate_tempo, evaluate
    bpm = estimate_tempo(track.original_path)
    evaluate(bpm, track.reference_bpm)

Run the whole comparison (original vs. Demucs instrumental):

    uv run --locked python tempo_estimation.py --split development
    uv run --locked python tempo_estimation.py --split test
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


WINDOW_SEC = 0.050          # STFT window length (notebook: 50 ms)
OVERLAP_SEC = 0.025         # STFT overlap (notebook) -> hop = window - overlap
GAMMA = 1000                # log-compression strength (notebook, Eq. 6.5)
LOCAL_AVG_SEC = 0.1         # half-width M of the local average (notebook)
TEMPO_WIN_SEC = 8.0         # tempogram window length, 4-12 is reasonable according to FMP
TEMPO_HOP_SEC = 0.5         # tempogram hop
BPM_AXIS = np.arange(40, 251, 1.0)


def load_audio(path):
    """Read a .wav file as mono float. Returns (data, sample_rate)."""
    data, sample_rate = sf.read(path, dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sample_rate


# Onset detection: spectral novelty (FMP Eq. 6.5-6.8)
def compute_local_average(x, M):
    """Local average over a centred window of 2M+1 samples.

    Same result as the loop version in the notebook (zero padding at the
    edges and always dividing by 2M+1), but vectorised.
    """
    return uniform_filter1d(x.astype(np.float64), size=2 * M + 1,
                            mode="constant", cval=0.0)


def compute_novelty(data, sample_rate, window_sec=WINDOW_SEC, overlap_sec=OVERLAP_SEC,
                    gamma=GAMMA, local_avg_sec=LOCAL_AVG_SEC):
    """Spectral-based novelty curve.

    Returns (novelty, fs_novelty) where fs_novelty is the novelty sampling
    rate in Hz.
    """
    window_len = int(window_sec * sample_rate)
    window_overlap = int(overlap_sec * sample_rate)
    hop_len = window_len - window_overlap
    _, _, Z = signal.stft(data, sample_rate, nperseg=window_len,
                          noverlap=window_overlap)
    magnitude = np.abs(Z)

    mag_compressed = np.log(1 + gamma * magnitude)           # Eq. 6.5
    diff = np.diff(mag_compressed, n=1, axis=1)              # Eq. 6.6
    diff[diff < 0] = 0
    novelty = np.sum(diff, axis=0)

    fs_novelty = sample_rate / hop_len
    M = int(np.ceil(local_avg_sec * fs_novelty))
    novelty = novelty - compute_local_average(novelty, M)   # Eq. 6.7-6.8
    novelty[novelty < 0] = 0
    if novelty.max() > 0:
        novelty = novelty / novelty.max()
    return novelty, fs_novelty


# Tempograms
def _frames(novelty, win_len, hop_len):
    """Overlapping windows of the novelty curve, zero-padded at the ends."""
    padded = np.pad(novelty, win_len // 2)
    n_frames = 1 + (len(padded) - win_len) // hop_len
    idx = np.arange(win_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return padded[idx]


def tempogram_fourier(novelty, fs_novelty, win_sec=TEMPO_WIN_SEC,
                      hop_sec=TEMPO_HOP_SEC, bpm=BPM_AXIS):
    """Fourier tempogram |T(n, tau)|, shape (len(bpm), number of frames).

    Compares each window of the novelty curve with a windowed complex
    sinusoid per tempo. Tends to show harmonics (2x, 3x the true tempo).

    FMP Eq. (6.25) & (6.26)
    """
    win_len = int(round(win_sec * fs_novelty))
    hop_len = max(1, int(round(hop_sec * fs_novelty)))
    frames = _frames(novelty, win_len, hop_len)
    n = np.arange(win_len)
    kernel = np.exp(-2j * np.pi * (bpm / 60.0)[:, None] * n[None, :] / fs_novelty)
    kernel *= np.hanning(win_len)[None, :]
    return np.abs(kernel @ frames.T)


def tempogram_autocorr(novelty, fs_novelty, win_sec=TEMPO_WIN_SEC,
                       hop_sec=TEMPO_HOP_SEC, bpm=BPM_AXIS, window_type="hanning"):
    """Autocorrelation tempogram, shape (len(bpm), number of frames).

    Local autocorrelation with lag converted to BPM = 60 * fs / lag.
    Tends to show subharmonics (1/2, 1/3 of the true tempo).
    """
    win_len = int(round(win_sec * fs_novelty))
    hop_len = max(1, int(round(hop_sec * fs_novelty)))

    if window_type.lower() == "hanning":
        frames = _frames(novelty, win_len, hop_len) * np.hanning(win_len)[None, :]
    else: # elif window_type.lower() == "rectangular": , won't be accessed by user anyway
        frames = _frames(novelty, win_len, hop_len) * np.ones(win_len)[None, :] 
    n_fft = 2 ** int(np.ceil(np.log2(2 * win_len)))
    F = np.fft.rfft(frames, n_fft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n_fft, axis=1)[:, :win_len]   # wiener-khinchin: ac = invFFT(|x|^2)
    ac /= (win_len - np.arange(win_len))[None, :]
    ac = np.maximum(ac, 0)
    lags = 60.0 * fs_novelty / bpm
    return np.stack([np.interp(lags, np.arange(win_len), row) for row in ac], axis=1)


# Global tempo
def _normalise(v):
    return v / v.max() if v.max() > 0 else v


def tempo_prior(bpm=BPM_AXIS, center=120.0, octave_width=1.0):
    """Log-Gaussian weight around `center` BPM (std in octaves)."""
    return np.exp(-0.5 * (np.log2(bpm / center) / octave_width) ** 2)


def global_tempo_curve(novelty, fs_novelty, method="combined", bpm=BPM_AXIS):
    """Tempogram averaged over time, normalised to max 1.

    method: "fourier", "autocorr" or "combined" (product of both, so the
    Fourier harmonics and the autocorrelation subharmonics cancel out).

    "Also, to alleviate the problem
    of tempo octave confusion, one may improve the result by a combined usage of the
    Fourier and autocorrelation tempograms." - Müller, FMP
    """
    if method == "fourier":
        return _normalise(tempogram_fourier(novelty, fs_novelty, bpm=bpm).mean(axis=1))
    if method == "autocorr":
        return _normalise(tempogram_autocorr(novelty, fs_novelty, bpm=bpm, window_type="hanning").mean(axis=1))
    # if method == "autocorr_rect":
    #     return _normalise(tempogram_autocorr(novelty, fs_novelty, bpm=bpm, window_type="rect").mean(axis=1))
    if method == "combined":
        f = global_tempo_curve(novelty, fs_novelty, "fourier", bpm)
        a = global_tempo_curve(novelty, fs_novelty, "autocorr", bpm)
        return _normalise(f * a)
    raise ValueError(f"Unknown method: {method}") 


def estimate_tempo_from_novelty(novelty, fs_novelty, method="combined",
                                use_prior=True, prior_center=120.0,
                                prior_octave_width=1.0, bpm=BPM_AXIS):
    """Pick the global tempo from a novelty curve. Returns (bpm, curve)."""
    curve = global_tempo_curve(novelty, fs_novelty, method, bpm)
    if use_prior:
        curve = curve * tempo_prior(bpm, prior_center, prior_octave_width)
    k = int(np.argmax(curve))

    # The autocorrelation lag grid is coarse (one novelty sample = 25 ms, with 
    # current default params), so once the metrical level is chosen, take the 
    # exact peak from the Fourier curve within +-6 % of it.
    if method != "autocorr":
        fourier = global_tempo_curve(novelty, fs_novelty, "fourier", bpm)
        near = np.abs(bpm - bpm[k]) <= 0.06 * bpm[k]
        k = int(np.flatnonzero(near)[np.argmax(fourier[near])])
        peak_curve = fourier
    else:
        peak_curve = curve

    # Parabolic interpolation for sub-BPM precision
    tempo = bpm[k]
    if 0 < k < len(bpm) - 1:
        a, b, c = peak_curve[k - 1], peak_curve[k], peak_curve[k + 1]
        if a - 2 * b + c != 0:
            tempo = bpm[k] + 0.5 * (a - c) / (a - 2 * b + c) * (bpm[1] - bpm[0])
    return float(tempo), curve


def estimate_tempo(audio, sample_rate=None, **kwargs):
    """Global tempo in BPM for a .wav path, or an array plus sample_rate."""
    if isinstance(audio, (str, Path)):
        audio, sample_rate = load_audio(audio)
    novelty, fs_novelty = compute_novelty(np.asarray(audio, dtype=np.float64), sample_rate)
    return estimate_tempo_from_novelty(novelty, fs_novelty, **kwargs)[0]


# Evaluation
def evaluate(est_bpm, ref_bpm, tol=0.04):
    """abs_error in BPM; acc1 = within +-4 % of the reference;
    acc2 = within +-4 % of reference x {1, 2, 3, 1/2, 1/3} (octave errors ok).
    
    see e.g. https://transactions.ismir.net/articles/10.5334/tismir.43 for more info
    """
    est, ref = float(est_bpm), float(ref_bpm)
    return {
        "abs_error": abs(est - ref),
        f"acc_1": abs(est - ref) <= tol * ref,
        f"acc_2": any(abs(est - ref * f) <= tol * ref * f for f in (1, 2, 3, 1 / 2, 1 / 3)),
    }


# Experiment: original vs. instrumental
def run_experiment(tracks, out_csv, **kwargs):
    """Estimate tempo for original and instrumental of every track, save a
    CSV and print a summary. `tracks` comes from ballroom_dataset."""
    rows = []
    for i, track in enumerate(tracks, 1):
        orig = estimate_tempo(track.original_path, **kwargs)
        inst = estimate_tempo(track.instrumental_path, **kwargs)
        ev_o = evaluate(orig, track.reference_bpm)
        ev_i = evaluate(inst, track.reference_bpm)
        rows.append({
            "track_id": track.track_id, "dance_style": track.dance_style,
            "human_vocals": track.human_vocals, "reference_bpm": track.reference_bpm,
            "orig_bpm": round(orig, 2), "inst_bpm": round(inst, 2),
            **{f"orig_{k}": v for k, v in ev_o.items()},
            **{f"inst_{k}": v for k, v in ev_i.items()},
        })
        print(f"[{i}/{len(tracks)}] {track.track_id}: ref {track.reference_bpm:.1f}  "
              f"original {orig:.1f}  instrumental {inst:.1f}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"\nSaved {len(rows)} rows to {out_csv}")
    return rows


def print_summary(rows):
    def stats(rs):
        o = np.array([r["orig_abs_error"] for r in rs])
        i = np.array([r["inst_abs_error"] for r in rs])
        return (len(rs), np.mean([r["orig_acc1"] for r in rs]), np.mean([r["inst_acc1"] for r in rs]),
                np.mean([r["orig_acc2"] for r in rs]), np.mean([r["inst_acc2"] for r in rs]),
                np.median(o), np.median(i))

    header = f"{'':16}{'n':>4}{'acc1 orig':>11}{'acc1 inst':>11}{'acc2 orig':>11}" \
             f"{'acc2 inst':>11}{'med err orig':>14}{'med err inst':>14}"
    print("\n" + header)
    for name, rs in [("ALL", rows)] + [
            (s, [r for r in rows if r["dance_style"] == s])
            for s in sorted({r["dance_style"] for r in rows})]:
        n, a1o, a1i, a2o, a2i, mo, mi = stats(rs)
        print(f"{name:16}{n:>4}{a1o:>11.2f}{a1i:>11.2f}{a2o:>11.2f}{a2i:>11.2f}{mo:>14.2f}{mi:>14.2f}")

    # report how many instrumental results are better/worse/same as original
    # avoid small numerical/floating point differences by handling diff manually
    diff = np.array([r["orig_abs_error"] - r["inst_abs_error"] for r in rows])
    diff = np.round(diff, decimals=6)
    print(f"\nInstrumental closer to reference: {(diff > 0.5).sum()}, "
          f"further away: {(diff < -0.5).sum()}, about equal (<0.5 BPM): "
          f"{(np.abs(diff) <= 0.5).sum()}")
    # statistical test to determine significance of results
    if np.any(diff != 0):
        from scipy.stats import wilcoxon
        print(f"Wilcoxon signed-rank test on absolute errors: p = {wilcoxon(diff).pvalue:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Tempo: original vs. instrumental")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--method", choices=["fourier", "autocorr", "combined"], default="combined")
    parser.add_argument("--no-prior", action="store_true")
    parser.add_argument("--out", help="CSV path (default results/tempo_<split>.csv)")
    args = parser.parse_args()

    from ballroom_dataset import get_development_set, get_test_set
    tracks = get_development_set() if args.split == "development" else get_test_set()
    run_experiment(tracks, args.out or f"results/tempo_{args.split}.csv",
                   method=args.method, use_prior=not args.no_prior)


if __name__ == "__main__":
    main()
