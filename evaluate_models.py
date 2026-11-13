#!/usr/bin/env python3
"""Evaluate all three models with accuracy, F1-score and log loss.

Each model is re-fitted inside cross-validation, so every score comes from samples the
model did not see while fitting. Near-copies are always held out together:
  - face:    leave one photo out. All 6 versions of a photo (original + 5 augmentations)
             leave the training set together; the model is scored on the original.
  - voice:   leave one recording out, scored on the original (the same protocol as
             voiceprint_model.py).
  - product: 5-fold, with all transactions of one customer in the same fold, because
             they share identical features.

Usage:  python evaluate_models.py     -> prints a table and writes evaluation_metrics.csv
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_utils import FaceModel, ProductModel, VoiceModel, load_config


def cross_validate(name, estimator, X, y_fit, y_names, labels, groups, scheme, score_rows=None):
    """Return the metrics for one model.

    y_fit: labels in the form the estimator was trained on. y_names/labels: readable names.
    groups: samples sharing a group are held out together. score_rows: boolean mask of the
    rows to score (the unaugmented originals); training always uses every row.
    """
    X, y_fit, y_names, groups = map(np.asarray, (X, y_fit, y_names, groups))
    y_names = y_names.astype(str)
    labels = [str(c) for c in labels]
    score_rows = np.ones(len(y_names), bool) if score_rows is None else np.asarray(score_rows)
    fit_classes = list(np.unique(y_fit))            # same sorted order as `labels`
    if scheme == "leave-one-out":
        splits = LeaveOneGroupOut().split(X, y_fit, groups)
    else:
        splits = StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y_fit, groups)
    proba = np.zeros((len(y_names), len(labels)))
    for train, test in splits:
        m = clone(estimator).fit(X[train], y_fit[train])
        cols = [fit_classes.index(c) for c in m.classes_]
        proba[np.ix_(test, cols)] = m.predict_proba(X[test])
    y, proba = y_names[score_rows], proba[score_rows]
    pred = np.array(labels)[proba.argmax(axis=1)]
    return {
        "model": name,
        "scored_samples": len(y),
        "classes": len(labels),
        "accuracy": accuracy_score(y, pred),
        "f1_weighted": f1_score(y, pred, average="weighted"),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, proba, labels=labels),
        "chance_accuracy": 1 / len(labels),
    }


def report_separation(title, score, member, group, originals, best, threshold):
    """For each original: best score against the same member's other samples, and against other members."""
    same, other = [], []
    for i in np.where(originals)[0]:
        same.append(best(score[i, (member == member[i]) & (group != group[i])]))
        other.append(best(score[i, member != member[i]]))
    print(f"  {title}: app threshold {threshold}")
    print(f"    same member   : {min(same):.2f} to {max(same):.2f}")
    print(f"    other members : {min(other):.2f} to {max(other):.2f}")


def main():
    cfg = load_config()
    results = []

    face = FaceModel(cfg["face_model"], cfg["face_scaler"], cfg["image_features_csv"])
    img = pd.read_csv(cfg["image_features_csv"])
    cols = [c for c in img.columns if c.startswith(("r_hist", "g_hist", "b_hist", "emb_"))]
    results.append(cross_validate(
        "Facial recognition", make_pipeline(StandardScaler(), clone(face.model)),
        img[cols].values, img["member"], img["member"], face.classes,
        img["member"] + "/" + img["expression"], "leave-one-out", img["augmentation"] == "original"))

    voice = VoiceModel(cfg["voice_model"], cfg["audio_features_csv"])
    aud = pd.read_csv(cfg["audio_features_csv"])
    results.append(cross_validate(
        "Voiceprint verification", voice.model,
        aud[voice.features].values, aud["member"], aud["member"], voice.classes,
        aud["recording_id"], "leave-one-out", aud["augmentation"] == "original"))

    product = ProductModel(cfg["product_model"])
    data = pd.read_csv(cfg["merged_dataset"])
    results.append(cross_validate(
        "Product recommendation", product.model,
        product.matrix(data).values, product.label_encoder.transform(data["product_category"]),
        data["product_category"], product.classes, data["customer_id"], "5-fold"))

    out = pd.DataFrame(results)
    out.to_csv("evaluation_metrics.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("\nSaved evaluation_metrics.csv")

    print("\nEnrolment check used by app.py (each original compared with the member's OTHER samples)")
    photo = (img["member"] + "/" + img["expression"]).values
    emb = np.vstack([face.enrolled[m] for m in img["member"].unique()])  # same row order as img
    report_separation("Face cosine similarity (higher = more alike)", emb @ emb.T,
                      img["member"].values, photo, (img["augmentation"] == "original").values,
                      best=np.max, threshold=cfg["face_min_similarity"])
    z = np.vstack([voice.enrolled[m] for m in aud["member"].unique()])
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    report_separation("Voice feature distance (lower = more alike)", dist,
                      aud["member"].values, aud["recording_id"].values,
                      (aud["augmentation"] == "original").values,
                      best=np.min, threshold=cfg["voice_max_distance"])


if __name__ == "__main__":
    main()
