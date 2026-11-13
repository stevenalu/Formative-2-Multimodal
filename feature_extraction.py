"""Turn one new image or audio file into the feature vector its model was trained on.

Image: reproduces the steps of notebooks/Formative2_ImagePipeline_David (1).ipynb
       (face detection and crop, 16-bin RGB histograms, MobileNetV2 embedding).
Audio: calls load_audio and extract_features from audio_pipeline.py, the same code
       that produced audio_features.csv, so the two can never drift apart.
"""
import os
import shutil
import subprocess
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # keep TensorFlow start-up messages off the screen

import numpy as np

IMG_SIZE = (224, 224)
N_BINS = 16
HIST_COLS = [f"{c}_hist_{i}" for c in "rgb" for i in range(N_BINS)]

_detector = None
_embedder = None


class NoFaceError(Exception):
    """Raised when no face at all is detected in an image."""


def load_image_models():
    """Load the face detector and the MobileNetV2 embedder once (takes a few seconds)."""
    global _detector, _embedder
    if _detector is None:
        from skimage.data import lbp_frontal_face_cascade_filename
        from skimage.feature import Cascade
        _detector = Cascade(lbp_frontal_face_cascade_filename())
    if _embedder is None:
        from tensorflow.keras.applications import MobileNetV2
        _embedder = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                                input_shape=(224, 224, 3))


def _detect(img):
    import cv2
    h, w = img.shape[:2]
    s = 800 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    faces = _detector.detect_multi_scale(img=gray, scale_factor=1.1, step_ratio=1,
                                         min_size=(40, 40), max_size=(800, 800))
    return small, faces


def _crop_face(small, faces):
    """Keep the large face closest to the centre, add a margin, resize to 224x224."""
    import cv2
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
    else:  # only small detections: fall back to a centre crop, as in training
        s = min(sh, sw)
        y1, x1 = (sh - s) // 2, (sw - s) // 2
        crop = small[y1:y1 + s, x1:x1 + s]
    return cv2.resize(crop, IMG_SIZE)


def _color_histogram(img):
    import cv2
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [N_BINS], [0, 256]).flatten()
        feats.extend(hist / hist.sum())
    return feats


def extract_image_features(path):
    """Return the 1328-value vector (48 histogram bins + 1280 embedding values) for one photo."""
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    load_image_models()
    img = cv2.imread(path)
    if img is None:
        raise ValueError("not a readable image (use a .jpg or .png file)")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small, faces = _detect(img)
    if len(faces) == 0:
        raise NoFaceError("no face detected in the image")
    face = _crop_face(small, faces)
    emb = _embedder.predict(preprocess_input(np.array([face]).astype("float32")), verbose=0)[0]
    return np.concatenate([_color_histogram(face), emb])


def _ffmpeg_exe():
    """Path to an ffmpeg program: the system one, else the copy bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _to_wav(path):
    """Convert a recording the audio library cannot open itself (phone .m4a, .aac, .mp4) to a
    temporary .wav file. Works on Windows, macOS and Linux: imageio-ffmpeg (in requirements.txt)
    ships its own ffmpeg, so nothing has to be installed separately."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg = _ffmpeg_exe()
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", path, out]
    elif shutil.which("afconvert"):  # macOS built-in, last resort
        cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", path, out]
    else:
        os.remove(out)
        raise ValueError("this audio format needs ffmpeg: run 'pip install imageio-ffmpeg' or use a .wav recording")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or os.path.getsize(out) == 0:
        os.remove(out)
        raise ValueError("the recording could not be decoded (is it a valid audio file?)")
    return out


def extract_audio_features(path):
    """Return the feature dict for one recording, using the audio pipeline's own functions."""
    import soundfile as sf
    from audio_pipeline import SR, extract_features, load_audio

    converted = None
    try:
        sf.info(path)                 # formats read directly: .wav, .flac, .ogg, .mp3
    except Exception:
        converted = _to_wav(path)     # everything else, e.g. .m4a from a phone
    try:
        y = load_audio(converted or path)
    finally:
        if converted:
            os.remove(converted)
    if len(y) < SR * 0.2 or not np.any(y):
        raise ValueError("recording is silent or shorter than 0.2 seconds")
    return extract_features(y)
