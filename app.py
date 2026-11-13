#!/usr/bin/env python3
"""User Identity and Product Recommendation System - command-line app (Group 11).

Multimodal decision logic:
  1. FACE     image -> facial recognition model. The best match must be a registered
              user with confidence >= face_threshold, AND the photo must resemble that
              user's enrolled photos (similarity >= face_min_similarity). Otherwise
              ACCESS DENIED.
  2. VOICE    audio -> voiceprint model. The best match must be the SAME user the face
              step identified, with confidence >= voice_threshold, AND the recording must
              be close to that user's enrolled recordings (distance <= voice_max_distance).
              Otherwise ACCESS DENIED.
  3. PRODUCT  only after both checks pass, the product recommendation model runs on that
              user's customer record and the prediction is displayed.

Usage:
  python app.py                                     interactive menu
  python app.py --face photo.jpg --voice clip.wav   one transaction, no prompts
"""
import argparse
import csv
import os
import sys
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd

import feature_extraction as fx
from model_utils import FaceModel, ProductModel, VoiceModel, load_config

LOG_PATH = "logs/access_log.csv"
LINE = "=" * 64
NOT_A_USER = ("unknown", "unauthorized", "other", "stranger")


class AccessDenied(Exception):
    def __init__(self, stage, reason):
        super().__init__(reason)
        self.stage, self.reason = stage, reason


class System:
    def __init__(self, config_path="config.json"):
        self.cfg = load_config(config_path)
        self.face = FaceModel(self.cfg["face_model"], self.cfg["face_scaler"], self.cfg["image_features_csv"])
        self.voice = VoiceModel(self.cfg["voice_model"], self.cfg["audio_features_csv"])
        self.product = ProductModel(self.cfg["product_model"])
        self.customers = pd.read_csv(self.cfg["merged_dataset"])
        self.users = {str(k).lower(): v for k, v in self.cfg["users"].items()}
        self.face_threshold = float(self.cfg.get("face_threshold", 0.6))
        self.voice_threshold = float(self.cfg.get("voice_threshold", self.voice.threshold))
        self.face_min_similarity = float(self.cfg.get("face_min_similarity", 0.6))
        self.voice_max_distance = float(self.cfg.get("voice_max_distance", 10.0))
        fx.load_image_models()

    def enrolment(self):
        """Users who can complete a transaction, and users missing from one of the models."""
        full = [u for u in self.users if u in self.face.classes and u in self.voice.classes]
        partial = {u: [m for m, model in (("face", self.face), ("voice", self.voice))
                       if u not in model.classes] for u in self.users if u not in full}
        return full, partial

    # ---- step 1 ----
    def verify_face(self, image_path):
        if not os.path.isfile(image_path):
            raise AccessDenied("face", f"image file not found: {image_path}")
        try:
            vector = fx.extract_image_features(image_path)
            ranked = self.face.predict_proba(vector)
        except fx.NoFaceError as e:
            raise AccessDenied("face", str(e))
        except Exception as e:
            raise AccessDenied("face", f"could not process the image: {e}")
        user, conf = ranked[0]
        print(f"  Best match : {user} (confidence {conf:.0%}, required {self.face_threshold:.0%})")
        if user.lower() in NOT_A_USER or conf < self.face_threshold:
            raise AccessDenied("face", "face not recognised as a registered user")
        if user.lower() not in self.users:
            raise AccessDenied("face", f"'{user}' has no customer record in config.json")
        # The classifier always picks one of the enrolled members, even for a stranger, so also
        # require the photo to resemble that member's enrolled photos.
        sim = self.face.similarity(vector, user)
        print(f"  Similarity : {sim:.2f} to {user}'s enrolled photos (required {self.face_min_similarity:.2f})")
        if sim < self.face_min_similarity:
            raise AccessDenied("face", f"photo does not resemble {user}'s enrolled photos")
        return user, conf

    # ---- step 2 ----
    def verify_voice(self, audio_path, expected_user):
        if not os.path.isfile(audio_path):
            raise AccessDenied("voice", f"audio file not found: {audio_path}")
        try:
            feats = fx.extract_audio_features(audio_path)
            ranked = self.voice.predict_proba(feats)
        except Exception as e:
            raise AccessDenied("voice", f"could not process the recording: {e}")
        speaker, conf = ranked[0]
        print(f"  Best match : {speaker} (confidence {conf:.0%}, required {self.voice_threshold:.0%})")
        if speaker.lower() != expected_user.lower():
            raise AccessDenied("voice", f"voice does not belong to {expected_user}, the user identified by face")
        if conf < self.voice_threshold:
            raise AccessDenied("voice", "voiceprint match is not confident enough")
        dist = self.voice.distance(feats, speaker)
        print(f"  Distance   : {dist:.1f} from {speaker}'s enrolled recordings (allowed up to {self.voice_max_distance:.1f})")
        if dist > self.voice_max_distance:
            raise AccessDenied("voice", f"recording is too different from {speaker}'s enrolled recordings")
        return speaker, conf

    # ---- step 3 ----
    def recommend(self, user, top_n=3):
        customer_id = self.users[user.lower()]
        rows = self.customers[self.customers["customer_id"].astype(str) == str(customer_id)]
        if rows.empty:
            raise AccessDenied("product", f"customer {customer_id} is not in {self.cfg['merged_dataset']}")
        return customer_id, self.product.predict_proba(rows.iloc[0].to_dict())[:top_n]


def log_attempt(image, audio, user, outcome, stage, detail):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "image", "audio", "identified_user", "outcome", "failed_step", "detail"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), image, audio or "", user or "",
                    outcome, stage or "", detail])


def run_transaction(system, image_path, get_audio_path):
    """One full attempt. get_audio_path is only called if the face step passes."""
    user, audio_path = None, None
    print(f"\n{LINE}\n NEW TRANSACTION\n{LINE}")
    try:
        print(f"\n[1/3] FACE RECOGNITION\n  Image      : {image_path}")
        user, _ = system.verify_face(image_path)
        print(f"  PASSED     : welcome, {user}. You may request a product prediction.")

        audio_path = get_audio_path()
        print(f"\n[2/3] VOICE VERIFICATION\n  Recording  : {audio_path}")
        system.verify_voice(audio_path, user)
        print(f"  PASSED     : voice confirms {user}. Prediction approved.")

        print("\n[3/3] PRODUCT RECOMMENDATION")
        customer_id, ranked = system.recommend(user)
        print(f"  Customer   : {customer_id}")
        for i, (product, p) in enumerate(ranked, 1):
            print(f"  {i}. {product:<14}{p:>5.0%}")
        print(f"\n{LINE}\n TRANSACTION APPROVED - recommended product for {user}: {ranked[0][0]}\n{LINE}")
        log_attempt(image_path, audio_path, user, "granted", None, f"recommended {ranked[0][0]}")
        return True
    except AccessDenied as e:
        print(f"  FAILED     : {e.reason}")
        print(f"\n{LINE}\n ACCESS DENIED at the {e.stage} step. No prediction was made.\n{LINE}")
        log_attempt(image_path, audio_path, user, "denied", e.stage, e.reason)
        return False


def ask_path(prompt):
    # dragging a file into the terminal adds quotes or backslash-escaped spaces: tidy both
    return input(prompt).strip().strip('"').strip("'").replace("\\ ", " ")


def show_users(system):
    full, partial = system.enrolment()
    print(f" Registered users (face + voice): {', '.join(full) or 'none'}")
    for user, missing in partial.items():
        print(f" Note: {user} is not in the {' or '.join(missing)} model, so cannot pass that step.")


def interactive(system):
    print(f"{LINE}\n USER IDENTITY AND PRODUCT RECOMMENDATION SYSTEM\n{LINE}")
    show_users(system)
    while True:
        print("\n 1. Start a transaction\n 2. View access log\n 3. Exit")
        choice = input(" Choose [1-3]: ").strip()
        if choice == "1":
            image = ask_path(" Path to face image: ")
            run_transaction(system, image, lambda: ask_path("\n Path to voice recording: "))
        elif choice == "2":
            if os.path.exists(LOG_PATH):
                print(pd.read_csv(LOG_PATH).fillna("").tail(10).to_string(index=False))
            else:
                print(" No attempts logged yet.")
        elif choice == "3":
            print(" Goodbye.")
            return
        else:
            print(" Please enter 1, 2 or 3.")


def main():
    ap = argparse.ArgumentParser(description="Face + voice authenticated product recommendation.")
    ap.add_argument("--face", help="path to a face image (.jpg or .png)")
    ap.add_argument("--voice", help="path to a voice recording (.wav, .mp3, .m4a, .flac, .ogg)")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    if bool(args.face) != bool(args.voice):
        ap.error("--face and --voice must be given together")
    print("Loading models (the first run downloads the MobileNetV2 weights, about 9 MB)...")
    try:
        system = System(args.config)
    except (FileNotFoundError, KeyError) as e:
        sys.exit(f"Could not start: {e}\nRun the app from the repository root and check the paths in {args.config}.")
    if args.face:
        sys.exit(0 if run_transaction(system, args.face, lambda: args.voice) else 1)
    try:
        interactive(system)
    except (KeyboardInterrupt, EOFError):
        print("\n Goodbye.")


if __name__ == "__main__":
    main()
