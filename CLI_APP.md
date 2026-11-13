# System simulation: command-line app

Part D (Marie Honette Ihozo). Files: `app.py`, `feature_extraction.py`, `model_utils.py`,
`evaluate_models.py`, `config.json`, `requirements.txt`.

## Setup

Use Python 3.10 to 3.13, from the repository root:

    python3 -m venv .venv
    source .venv/bin/activate            # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

The first run downloads the MobileNetV2 weights (about 9 MB), so it needs internet once.

Notes:

- The app runs on Windows, macOS and Linux. Phone recordings (`.m4a`) are converted automatically
  by the ffmpeg that comes with the `imageio-ffmpeg` package, so nothing extra has to be installed.
- **macOS only:** XGBoost needs the OpenMP library. If the app stops with
  `libxgboost.dylib could not be loaded`, run `brew install libomp` once.
- Photos must be `.jpg` or `.png` (iPhone `.heic` files are not supported; export them as JPG).

## Run

    python app.py                                        # interactive menu
    python app.py --face photo.jpg --voice clip.wav      # one transaction, no prompts
    python evaluate_models.py                            # accuracy, F1 and log loss for all three models

In the menu, choose 1, then give the path to a face photo (`.jpg` or `.png`) and, if the face is
accepted, the path to a voice recording. Dragging a file into the terminal types its path.

## Multimodal decision logic

1. **Face recognition.** The photo goes through the same steps as the image notebook (face
   detection and crop, RGB histograms, MobileNetV2 embedding) and the facial recognition model
   names the best-matching member. Access continues only if
   - a face is detected,
   - the model's confidence is at least `face_threshold` (0.60), and
   - the photo's embedding has a cosine similarity of at least `face_min_similarity` (0.60) with
     that member's enrolled photos in `datasets/image_features.csv`.
2. **Product recommendation model.** A recognised face unlocks the product model. It runs on the
   member's customer record from `datasets/merged_customer_data.csv`, and the prediction is held
   without being shown.
3. **Voice verification.** The recording goes through `audio_pipeline.py` and the voiceprint model
   names the best-matching speaker. The held prediction is displayed only if
   - the speaker is the **same member** the face step identified,
   - the model's confidence is at least `voice_threshold` (0.60), and
   - the recording is within `voice_max_distance` (10.0) of that member's enrolled recordings in
     `audio_features.csv`, measured in standardised feature space.

A failed face check prints `ACCESS DENIED` and the product model is never called. A failed voice
check prints `ACCESS DENIED` and the held prediction is discarded without being shown. Every
attempt is appended to `logs/access_log.csv` (menu option 2 shows the last ten).

**Why the similarity and distance checks exist.** Both classifiers can only answer with one of the
enrolled members, so a stranger is always assigned to somebody, often with high confidence. In
testing, a photo of a person outside the group was classified as david at 98% confidence; its
similarity to David's enrolled photos was 0.43, so the similarity check denied it. The thresholds
come from `evaluate_models.py`, which prints how similar each member's samples are to their own
other samples (face 0.75 to 0.93, voice distance 3.8 to 6.7). The voice distance check only
catches recordings that are very unlike the enrolled ones, such as non-speech audio; with two
recordings per member it cannot separate a stranger's voice reliably.

## Demonstration

| Scenario | Inputs | Expected result |
|---|---|---|
| Full transaction | A member's photo, then the same member's recording | Approved, product shown |
| Unauthorized face | Photo of someone outside the group | Denied at the face step |
| Unauthorized voice | A member's photo, then another person's recording | Denied at the voice step |
| No face | Any image without a face | Denied at the face step |

## Who can log in

The facial recognition model was trained on david, faith and steven. The voiceprint model also
knows honette. A member needs both, so honette cannot pass the face step until the face model is
retrained with her photos. The app lists this when it starts.

## config.json

- `users`: each member's name mapped to a `customer_id` in the merged dataset. The IDs are examples
  from the product notebook.
- `face_threshold`, `voice_threshold`, `face_min_similarity`, `voice_max_distance`: the four limits
  described above. Raise the first three (or lower the last) to be stricter.
- The remaining entries are the paths of the models and feature tables.

## Evaluation

`evaluate_models.py` re-fits each saved model inside cross-validation and writes
`evaluation_metrics.csv`. Face and voice use leave-one-photo-out and leave-one-recording-out, so
augmented copies of the held-out sample never appear in training. The product model uses 5 folds
with each customer's transactions kept together, because they share identical features.

| Model | Scored samples | Accuracy | F1 (weighted) | Log loss | Chance accuracy |
|---|---|---|---|---|---|
| Facial recognition | 9 photos | 1.000 | 1.000 | 0.090 | 0.333 |
| Voiceprint verification | 8 recordings | 0.875 | 0.867 | 0.549 | 0.250 |
| Product recommendation | 117 transactions | 0.171 | 0.147 | 2.096 | 0.200 |

The face and voice scores rest on very few samples. The product score is stricter than the
notebook's split (which lets one customer appear in both training and test data) and is below
chance: the social features do not predict the product category in this dataset.
