import argparse
import glob
import os

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000      
N_MFCC = 13
RNG = np.random.default_rng(42)


def load_audio(path, sr=SR, trim=True):
    """Load as mono at a fixed sample rate and trim leading/trailing silence."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if trim:
        trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(trimmed) > sr * 0.2:
            y = trimmed
    return y


def discover_samples(audio_dir):
    """Return a list of dicts: member, phrase, recording_id, path."""
    rows = []
    for path in sorted(glob.glob(os.path.join(audio_dir, "*", "*.*"))):
        if not path.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            continue
        member = os.path.basename(os.path.dirname(path)).lower()
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        phrase = "_".join(p for p in stem.split("_") if not p.isdigit() and p != member) or stem
        rows.append({"member": member, "phrase": phrase,
                     "recording_id": f"{member}/{stem}", "path": path})
    if not rows:
        raise FileNotFoundError(f"No audio files found under {audio_dir}/<member>/")
    return rows


def plot_waveform_spectrogram(y, title, save_path=None, sr=SR):
    """Waveform (top) + log-mel spectrogram (bottom) for one sample."""
    fig, ax = plt.subplots(2, 1, figsize=(9, 5))
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title=f"Waveform - {title}", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set(title=f"Mel spectrogram - {title}")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    return fig


def plot_augmentation_comparison(y, augmented, title, save_path=None, sr=SR):
    """Original vs each augmentation: one row each, waveform on the left, spectrogram right."""
    items = [("original", y)] + list(augmented.items())
    fig, axes = plt.subplots(len(items), 2, figsize=(11, 2.2 * len(items)))
    for i, (name, sig) in enumerate(items):
        librosa.display.waveshow(sig, sr=sr, ax=axes[i, 0])
        axes[i, 0].set_title(f"{name} (waveform)", fontsize=9)
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=64), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", ax=axes[i, 1])
        axes[i, 1].set_title(f"{name} (spectrogram)", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110)
        plt.close(fig)
    return fig


def add_noise(y, snr_db=20):
    """Add white noise at a given signal-to-noise ratio."""
    sig_power = np.mean(y ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    return y + RNG.normal(0, np.sqrt(noise_power), size=y.shape)


def augment_all(y, sr=SR):
    return {
        "time_stretch": time_stretch(y, 1.1),   
        "noise": add_noise(y, 20),              
    }


def extract_features(y, sr=SR):
    """Fixed-length feature vector: MFCC (mean/std), spectral roll-off, energy, + a few extras."""
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    feats.update({
        "rolloff_mean": float(np.mean(rolloff)), "rolloff_std": float(np.std(rolloff)),
        "rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)),
        "energy_total": float(np.sum(y ** 2)),
        "centroid_mean": float(np.mean(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "zcr_mean": float(np.mean(zcr)),
        "duration_s": float(len(y) / sr),
    })
    return feats


FEATURE_PREFIXES = ("mfcc_", "rolloff_", "rms_", "energy_", "centroid_", "bandwidth_", "zcr_", "duration_")


def feature_columns(df):
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def run_pipeline(audio_dir, out_dir, save_augmented_wavs=True):
    plots_dir = os.path.join(out_dir, "audio_plots")
    aug_dir = os.path.join(out_dir, "augmented_audio")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(aug_dir, exist_ok=True)

    samples = discover_samples(audio_dir)
    print(f"Found {len(samples)} recordings from {len({s['member'] for s in samples})} member(s)")

    rows = []
    for s in samples:
        y = load_audio(s["path"])
        tag = s["recording_id"].replace("/", "_")

        plot_waveform_spectrogram(y, s["recording_id"], os.path.join(plots_dir, f"{tag}_original.png"))

        augmented = augment_all(y)
        plot_augmentation_comparison(y, augmented, s["recording_id"],
                                     os.path.join(plots_dir, f"{tag}_augmentations.png"))

        variants = {"original": y, **augmented}
        for aug_name, sig in variants.items():
            if save_augmented_wavs and aug_name != "original":
                sf.write(os.path.join(aug_dir, f"{tag}_{aug_name}.wav"), sig, SR)
            row = {"member": s["member"], "phrase": s["phrase"],
                   "recording_id": s["recording_id"], "augmentation": aug_name}
            row.update(extract_features(sig))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "audio_features.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows x {len(feature_columns(df))} features -> {csv_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    run_pipeline(args.audio_dir, args.out_dir)
