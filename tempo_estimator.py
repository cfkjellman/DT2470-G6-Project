"""The tempo estimators used by our development and test notebooks."""

import numpy as np
from scipy import signal


def compute_stft(data, sample_rate, window_size, hop_size):
    window_samples = round(window_size * sample_rate)
    hop_samples = round(hop_size * sample_rate)

    frequencies, times, stft = signal.stft(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=window_samples,
        noverlap=window_samples - hop_samples
    )

    stft = stft * signal.get_window("hann", window_samples).sum()
    return frequencies, times, stft


def log_compression(X: np.ndarray, gamma: float) -> np.ndarray:
    return np.log1p(gamma * np.abs(X))


def spectral_novelty_function(Y: np.ndarray) -> np.ndarray:
    diff = np.diff(Y, axis=1)
    positive_diff = np.maximum(diff, 0)
    return np.sum(positive_diff, axis=0)


def local_average_function(novelty: np.ndarray, M: int) -> np.ndarray:
    local_average = np.zeros_like(novelty, dtype=float)

    for n in range(len(novelty)):
        start = max(0, n - M)
        end = min(len(novelty), n + M + 1)

        local_average[n] = np.sum(novelty[start:end]) / (2 * M + 1)

    return local_average


def enhanced_novelty_function(novelty: np.ndarray, M: int) -> np.ndarray:
    local_average = local_average_function(novelty, M)
    return np.maximum(novelty - local_average, 0)


def fourier_tempogram(
    novelty: np.ndarray,
    novelty_rate: float,
    bpms: np.ndarray,
    window_seconds: float,
    hop_seconds: float,
):
    # Window and hop measured in novelty frames.
    N = round(window_seconds * novelty_rate)
    H = round(hop_seconds * novelty_rate)
    window = np.hanning(N)

    # Zero-padding lets us center windows at the beginning and end.
    padding = N // 2
    padded_novelty = np.pad(novelty, (padding, padding))
    m = np.arange(len(padded_novelty)) - padding

    centers = np.arange(0, len(novelty), H)
    coefficients = np.zeros(
        (len(bpms), len(centers)),
        dtype=complex,
    )

    for k, bpm in enumerate(bpms):
        omega = bpm / (60 * novelty_rate)
        sinusoid = np.exp(-2j * np.pi * omega * m)

        # Delta(m) multiplied by the complex sinusoid.
        modulated_novelty = padded_novelty * sinusoid

        for j, center in enumerate(centers):
            segment = modulated_novelty[center:center + N]

            # Equation 6.25: sum over the window.
            coefficients[k, j] = np.sum(segment * window)

    tempogram = np.abs(coefficients)
    times = centers / novelty_rate

    return tempogram, times


def autocorrelation_tempogram(
    novelty,
    novelty_rate,
    bpms,
    window_seconds,
    hop_seconds,
):
    N = round(window_seconds * novelty_rate)
    H = round(hop_seconds * novelty_rate)
    window = np.hanning(N)

    padding = N // 2
    padded_novelty = np.pad(novelty, (padding, padding))
    centers = np.arange(0, len(novelty), H)

    # Lag corresponding to each candidate BPM.
    candidate_lags = 60 * novelty_rate / bpms
    tempogram = np.zeros((len(bpms), len(centers)))

    for j, center in enumerate(centers):
        segment = padded_novelty[center:center + N] * window

        correlation = np.correlate(segment, segment, mode="full")

        # Keep zero and positive lags.
        correlation = correlation[N - 1:]

        # Candidate lags can fall between integer frame positions.
        tempogram[:, j] = np.interp(
            candidate_lags,
            np.arange(N),
            correlation,
        )

    return tempogram, centers / novelty_rate


def global_tempo_estimation(
    tempogram: np.ndarray,
    bpms: np.ndarray,
):
    # Equation 6.32: average each candidate tempo's strength across time.
    global_strength = tempogram.mean(axis=1)

    # Select the BPM corresponding to the maximum, following Eq. 6.33.
    best_index = np.argmax(global_strength)
    estimated_bpm = float(bpms[best_index])

    return estimated_bpm, global_strength


def combined_tempo_estimation(fourier_strength, autocorr_strength, bpms):
    """Multiply the normalized global strength curves and select a tempo."""
    # Leave an all-zero curve unchanged to avoid dividing by zero.
    if fourier_strength.max() > 0:
        fourier_strength = fourier_strength / fourier_strength.max()
    if autocorr_strength.max() > 0:
        autocorr_strength = autocorr_strength / autocorr_strength.max()

    combined_strength = fourier_strength * autocorr_strength
    estimated_bpm = float(bpms[np.argmax(combined_strength)])

    return estimated_bpm, combined_strength


def compute_novelty(data, sample_rate, parameters):
    _, _, stft = compute_stft(
        data, sample_rate, parameters["window_size"], parameters["hop_size"]
    )
    compressed = log_compression(stft, parameters["gamma"])
    novelty = spectral_novelty_function(compressed)

    hop_samples = round(parameters["hop_size"] * sample_rate)
    novelty_rate = sample_rate / hop_samples
    M = int(np.ceil(parameters["local_average_size"] * novelty_rate))
    enhanced = enhanced_novelty_function(novelty, M)

    return enhanced, novelty_rate


def estimate_tempo(data, sample_rate, parameters, method):
    """Return the estimated BPM and the strength curve used to select it."""
    novelty, novelty_rate = compute_novelty(data, sample_rate, parameters)
    bpms = np.arange(parameters["bpm_min"], parameters["bpm_max"] + 1)

    if method in ("fourier", "combined"):
        tempogram, _ = fourier_tempogram(
            novelty, novelty_rate, bpms,
            window_seconds=parameters["tempogram_window_size"],
            hop_seconds=parameters["tempogram_hop_size"],
        )
        fourier_bpm, fourier_strength = global_tempo_estimation(tempogram, bpms)

    if method in ("autocorr", "combined"):
        tempogram, _ = autocorrelation_tempogram(
            novelty, novelty_rate, bpms,
            window_seconds=parameters["tempogram_window_size"],
            hop_seconds=parameters["tempogram_hop_size"],
        )
        autocorr_bpm, autocorr_strength = global_tempo_estimation(tempogram, bpms)

    if method == "fourier":
        return fourier_bpm, fourier_strength
    if method == "autocorr":
        return autocorr_bpm, autocorr_strength
    if method == "combined":
        return combined_tempo_estimation(fourier_strength, autocorr_strength, bpms)

    raise ValueError(f"Unknown tempo method: {method}")
