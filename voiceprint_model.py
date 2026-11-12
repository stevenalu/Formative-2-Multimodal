"""
Voiceprint verification model.

Train + evaluate (reads outputs/audio_features.csv from audio_pipeline.py):
    python voiceprint_model.py train --csv outputs/audio_features.csv --model_out models/voiceprint_model.joblib

Verify one recording (this is what the command-line demo app calls):
    python voiceprint_model.py verify --model models/voiceprint_model.joblib --audio path/to/clip.wav
"""
import argparse
import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, f1_score, log_loss)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_pipeline import extract_features, feature_columns, load_audio

DEFAULT_THRESHOLD = 0.60  # minimum confidence to accept a voice as an enrolled member


def candidate_models():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
        "random_forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42)),
    }


def cross_validate(model, X, y, groups, is_original, classes):
    """
    Leave-one-recording-out CV (right for tiny datasets): all augmented copies of one recording stay in the same fold, so the model
    is never tested on a pitch-shifted copy of something it trained on (that would leak).
    Train on originals + augmentations, test only on the original held-out recordings.
    """
    all_true, all_pred, all_proba = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        te = te[is_original[te]]
        if len(te) == 0:
            continue
        m = _clone(model).fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])
        # align columns to the full class list (a class can be absent from a training fold)
        full = np.zeros((len(te), len(classes)))
        for j, c in enumerate(m.classes_):
            full[:, list(classes).index(c)] = proba[:, j]
        all_proba.append(full)
        all_true.extend(y[te])
        all_pred.extend(np.array(classes)[full.argmax(1)])
    proba = np.vstack(all_proba)
    return np.array(all_true), np.array(all_pred), proba


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def train(csv_path, model_out, threshold=DEFAULT_THRESHOLD, report_dir="outputs"):
    df = pd.read_csv(csv_path)
    cols = feature_columns(df)
    X = df[cols].values
    y = df["member"].values
    groups = df["recording_id"].values
    is_original = (df["augmentation"] == "original").values
    classes = sorted(np.unique(y))
    print(f"{len(df)} rows | {len(cols)} features | members: {classes}")

    results, best_name, best_f1 = {}, None, -1
    for name, model in candidate_models().items():
        yt, yp, proba = cross_validate(model, X, y, groups, is_original, classes)
        res = {
            "accuracy": accuracy_score(yt, yp),
            "f1_macro": f1_score(yt, yp, average="macro"),
            "log_loss": log_loss(yt, proba, labels=classes),
        }
        results[name] = res
        print(f"{name:20s} acc={res['accuracy']:.3f}  f1={res['f1_macro']:.3f}  loss={res['log_loss']:.3f}")
        if res["f1_macro"] > best_f1 or (res["f1_macro"] == best_f1 and res["log_loss"] < results[best_name]["log_loss"]):
            best_name, best_f1, best_cm = name, res["f1_macro"], (yt, yp)

    print(f"Best model: {best_name}")
    os.makedirs(report_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(best_cm[0], best_cm[1], labels=classes, ax=ax, colorbar=False)
    ax.set_title(f"Voiceprint CV confusion matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "voiceprint_confusion_matrix.png"), dpi=120)
    plt.close(fig)
    with open(os.path.join(report_dir, "voiceprint_metrics.json"), "w") as f:
        json.dump({"cv_results": results, "best": best_name}, f, indent=2)

    final = candidate_models()[best_name].fit(X, y)  # refit on all data
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump({"model": final, "feature_cols": cols, "classes": classes,
                 "threshold": threshold, "name": best_name}, model_out)
    print(f"Saved model -> {model_out}")


def verify_voice(audio_path, model_path, claimed_member=None, threshold=None):
    """
    Returns dict: approved (bool), predicted_member, confidence.
    Approved only if the best-matching enrolled member has confidence >= threshold
    (and matches claimed_member when given, e.g. the user the face model identified).
    """
    bundle = joblib.load(model_path)
    threshold = bundle["threshold"] if threshold is None else threshold
    y = load_audio(audio_path)
    feats = extract_features(y)
    x = np.array([[feats[c] for c in bundle["feature_cols"]]])
    proba = bundle["model"].predict_proba(x)[0]
    classes = list(bundle["model"].classes_)
    i = int(np.argmax(proba))
    predicted, conf = classes[i], float(proba[i])
    approved = conf >= threshold and (claimed_member is None or predicted == claimed_member.lower())
    return {"approved": bool(approved), "predicted_member": predicted,
            "confidence": round(conf, 3), "threshold": threshold}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--csv", default="outputs/audio_features.csv")
    t.add_argument("--model_out", default="models/voiceprint_model.joblib")
    t.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    v = sub.add_parser("verify")
    v.add_argument("--model", default="models/voiceprint_model.joblib")
    v.add_argument("--audio", required=True)
    v.add_argument("--claimed", default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.csv, a.model_out, a.threshold)
    else:
        print(verify_voice(a.audio, a.model, a.claimed))
