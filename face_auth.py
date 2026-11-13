"""Face authentication step of the multimodal pipeline.

Usage:
    python face_auth.py --image path/to/photo.jpg
Exit code 0 = access granted, 1 = access denied.
"""
import argparse
import os
import sys

import cv2
import joblib
import numpy as np
from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)
N_BINS = 16
FACE_THRESHOLD = 0.60

_detector = Cascade(lbp_frontal_face_cascade_filename())
_embedder = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(224, 224, 3))
_model = joblib.load(os.path.join(MODEL_DIR, "face_model_logreg.pkl"))
_scaler = joblib.load(os.path.join(MODEL_DIR, "face_scaler.pkl"))


def _detect(img):
    h, w = img.shape[:2]
    s = 800 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = _detector.detect_multi_scale(img=gray, scale_factor=1.1, step_ratio=1,
                                         min_size=(40, 40), max_size=(800, 800))
    return small, faces


def crop_face(img):
    small, faces = _detect(img)
    sh, sw = small.shape[:2]
    big = [f for f in faces if f["width"] >= 0.25 * min(sh, sw)]
    if big:
        def dist(f):
            cx, cy = f["c"] + f["width"] / 2, f["r"] + f["height"] / 2
            return (cx - sw / 2) ** 2 + (cy - sh / 2) ** 2
        f = min(big, key=dist)
        x, y, fw, fh = f["c"], f["r"], f["width"], f["height"]
        m = int(0.25 * max(fw, fh))
        crop = small[max(y - m, 0):min(y + fh + m, sh), max(x - m, 0):min(x + fw + m, sw)]
    else:
        s = min(sh, sw)
        y1, x1 = (sh - s) // 2, (sw - s) // 2
        crop = small[y1:y1 + s, x1:x1 + s]
    return cv2.resize(crop, IMG_SIZE)


def color_histogram(img):
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [N_BINS], [0, 256]).flatten()
        feats.extend(hist / hist.sum())
    return feats


def extract_features(face_img):
    x = preprocess_input(np.array([face_img]).astype("float32"))
    emb = _embedder.predict(x, verbose=0)[0]
    return np.concatenate([color_histogram(face_img), emb]).reshape(1, -1)


def predict_face(image_path):
    """Returns (name or None, confidence, message)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0, "ACCESS DENIED: could not read image."
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, faces = _detect(img)
    if len(faces) == 0:
        return None, 0.0, "ACCESS DENIED: no face detected."

    feats = _scaler.transform(extract_features(crop_face(img)))
    proba = _model.predict_proba(feats)[0]
    best = proba.argmax()
    name, conf = _model.classes_[best], float(proba[best])
    if conf < FACE_THRESHOLD:
        return None, conf, f"ACCESS DENIED: face not recognized (confidence {conf:.2f})."
    return name, conf, f"Face recognized: {name} (confidence {conf:.2f})."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face authentication")
    parser.add_argument("--image", required=True, help="path to a face image")
    args = parser.parse_args()
    name, conf, msg = predict_face(args.image)
    print(msg)
    sys.exit(0 if name else 1)
