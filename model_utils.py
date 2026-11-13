"""Load the three saved models and give them one common interface.

Each wrapper has:  .classes  (labels)  and  .predict_proba(x) -> [(label, probability)]
sorted from most to least likely.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def _ranked(labels, proba):
    return sorted(zip([str(c) for c in labels], map(float, proba)), key=lambda t: -t[1])


class FaceModel:
    """Logistic regression + the StandardScaler it was trained with (saved as two files)."""

    N_HIST = 48  # the vector is 48 histogram values followed by the 1280 embedding values

    def __init__(self, model_path, scaler_path, features_csv):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.classes = [str(c) for c in self.model.classes_]
        # enrolled embeddings per member, unit length, for the similarity check
        df = pd.read_csv(features_csv)
        emb = df[[c for c in df.columns if c.startswith("emb_")]].values
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.enrolled = {m: emb[(df["member"] == m).values] for m in df["member"].unique()}

    def similarity(self, vector, member):
        """Highest cosine similarity between this photo's embedding and the member's enrolled photos."""
        e = np.asarray(vector, dtype=float)[self.N_HIST:]
        return float((self.enrolled[member] @ (e / np.linalg.norm(e))).max())

    def predict_proba(self, vector):
        x = self.scaler.transform(np.asarray(vector, dtype=float).reshape(1, -1))
        return _ranked(self.model.classes_, self.model.predict_proba(x)[0])


class VoiceModel:
    """Bundle saved by voiceprint_model.py: {'model', 'feature_cols', 'classes', 'threshold'}."""

    def __init__(self, path, features_csv):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.features = list(bundle["feature_cols"])
        self.threshold = float(bundle.get("threshold", 0.6))
        self.classes = [str(c) for c in self.model.classes_]
        # enrolled recordings per member in standardised feature space, for the distance check
        df = pd.read_csv(features_csv)
        self._mean = df[self.features].values.mean(axis=0)
        self._std = df[self.features].values.std(axis=0)
        z = (df[self.features].values - self._mean) / self._std
        self.enrolled = {m: z[(df["member"] == m).values] for m in df["member"].unique()}

    def distance(self, feats, member):
        """Distance from this recording to the member's closest enrolled recording (0 = identical)."""
        z = (np.array([feats[c] for c in self.features]) - self._mean) / self._std
        return float(np.linalg.norm(self.enrolled[member] - z, axis=1).min())

    def predict_proba(self, feats):
        x = np.array([[feats[c] for c in self.features]])
        return _ranked(self.model.classes_, self.model.predict_proba(x)[0])


class ProductModel:
    """Bundle saved by the product notebook: {'model', 'label_encoder', 'features'}."""

    def __init__(self, path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.label_encoder = bundle["label_encoder"]
        self.features = list(bundle["features"])
        self.classes = [str(c) for c in self.label_encoder.classes_]

    def matrix(self, df):
        """Rebuild the training columns from raw columns.

        The notebook one-hot encoded main_platform and main_sentiment with pandas.get_dummies,
        so 'main_platform_Twitter' is 1 when the raw column 'main_platform' equals 'Twitter'.
        """
        out = {}
        for f in self.features:
            if f in df.columns:
                out[f] = df[f]
                continue
            raw = [c for c in df.columns if f.startswith(c + "_")]
            if not raw:
                raise KeyError(f"product model needs column '{f}', which the dataset does not have")
            c = max(raw, key=len)
            out[f] = (df[c].astype(str) == f[len(c) + 1:]).astype(int)
        return pd.DataFrame(out, index=df.index)[self.features]

    def predict_proba(self, record):
        proba = self.model.predict_proba(self.matrix(pd.DataFrame([record])))[0]
        labels = self.label_encoder.inverse_transform(np.asarray(self.model.classes_).astype(int))
        return _ranked(labels, proba)
