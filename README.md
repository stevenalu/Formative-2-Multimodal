# Formative 2: Multimodal Data Preprocessing

**Group 11** | Machine Learning Pipeline | BSE (Machine Learning), African Leadership University

A Multimodal Biometric Authentication and Product Recommendation System. A user is checked by a
facial recognition model, the product recommendation model runs, and the prediction is only shown
after the voiceprint verification model confirms the same identity. If either check fails, access is
denied.

## Links

| Item | Link |
| --- | --- |
| Demo video | [https://youtu.be/7eo35uVNTMc](https://youtu.be/7eo35uVNTMc) |
| Report | [Multimodal Data Preprocessing Report](https://docs.google.com/document/d/1FM-0uddmthaTTCjWLR-usuTwSWouglpHrsD04o4rgqk/edit?tab=t.0#heading=h.ta7nb5idig5q) |
| Contribution tracker | [ML Pipeline Task Sheet, Group 11](https://docs.google.com/spreadsheets/d/1nO_AxQ_Mabwoh3_F5_F7LJzwCG1Tg0AM_vAg7T540jk/edit?usp=sharing) |

## Repository structure

```
datasets/
  customer_transactions - customer_transactions.csv
  customer_social_profiles - customer_social_profiles.csv
  merged_customer_data.csv
  image_features (1).csv             image features for all four members (used by the app)
  image_features.csv                 earlier version with three members
audio_features.csv
notebooks/
  Group_11_product_recommendation_model.ipynb
  Formative2_ImagePipeline_David (1).ipynb
  audio_voiceprint.ipynb
models/
  product_recommendation_model.joblib
  face_model_logreg.pkl
  face_scaler.pkl
  voiceprint_model.joblib
audio_pipeline.py                    audio loading, augmentation and feature extraction
voiceprint_model.py                  voiceprint model training and verification
face_auth.py                         standalone face authentication script
app.py                               CLI application (system simulation)
feature_extraction.py                turns a new photo or recording into model features
model_utils.py                       loads the three models
evaluate_models.py                   accuracy, F1 and log loss for all three models
config.json                          thresholds, file paths and member-to-customer mapping
requirements.txt
CLI_APP.md                           full guide to the CLI application
README.md
```

## How to run

To run the system simulation, see section 4 below or [CLI_APP.md](CLI_APP.md).

The notebooks are written for Google Colab.

1. Open the notebook from `notebooks/` in Google Colab.
2. Upload the CSV files it needs into `/content` using the file panel on the left.
3. Run all cells from top to bottom.
4. Download any file the notebook writes before the runtime disconnects, because Colab clears
   `/content` when the session ends.

For `Group_11_product_recommendation_model.ipynb`, upload these two files with exactly these names:

```
customer_transactions - customer_transactions.csv
customer_social_profiles - customer_social_profiles.csv
```

If you rename them, update the two paths in the data loading cell. The notebook writes
`merged_customer_data.csv` and `models/product_recommendation_model.joblib`.

## 1. Data merge and product recommendation model

Predicts the product category a customer is likely to buy from their social media behaviour.

**Cleaning.** 10 missing values in `customer_rating` filled with the median of 3.0. 5 exact duplicate
rows removed from the social profiles. `purchase_date` cast to datetime. The social id format `A178`
converted to `178` so it matches `customer_id_legacy`.

**Merge.** A plain inner join expands 150 transactions into 213 rows, because a customer has one
social row per platform. That would put the same transaction in both the training and the test set.
The social table is therefore collapsed to one row per customer first, using the mean engagement
score, the mean purchase interest score, the platform count, the most used platform and the most
common review sentiment. The inner join then gives 117 rows with no duplicated transaction ids and
no nulls.

**Features.** `engagement_score`, `purchase_interest_score`, `platform_count`,
`engagement_per_platform`, plus one hot encoded `main_platform` and `main_sentiment`.
`purchase_amount` and `customer_rating` are excluded because they describe a transaction that has
already happened, and the system has to recommend before the customer buys.

**Results.** Split 80/20 stratified, 93 training rows and 24 test rows. Because the test set is
small, 5 fold cross validation was used to select the model.

| Experiment | CV Accuracy | Test Accuracy | F1 | Log Loss |
| --- | --- | --- | --- | --- |
| Random Forest (baseline) | 0.236 | 0.250 | 0.237 | 3.550 |
| Logistic Regression | 0.257 | 0.125 | 0.111 | 1.990 |
| XGBoost (baseline) | 0.247 | 0.292 | 0.281 | 3.114 |
| Random Forest (tuned) | 0.289 | 0.125 | 0.143 | 1.711 |
| **XGBoost (tuned)** | **0.290** | **0.250** | **0.226** | **2.144** |

Tuned XGBoost was selected. A model that guesses scores 0.20 accuracy and about 1.61 log loss on five
classes, so the baselines were confidently wrong and tuning fixed the calibration. Accuracy stays low
because 117 rows across 5 categories is a small dataset and the social features barely separate the
classes, which the correlation matrix and the feature importance chart both show.


## 2. Facial recognition model

Notebook: `notebooks/Formative2_ImagePipeline_David (1).ipynb`.

Identifies which team member is in a face photo. Each of the four members gave three photos
(neutral, smiling, surprised), 12 originals in total.

**Preprocessing and augmentation.** Each photo is cropped to the face with a cascade face detector
and resized to 224x224. Five augmentations are applied to every photo (rotation, horizontal flip,
grayscale, brightness, Gaussian noise), giving 6 versions per photo and 72 samples.

**Features.** 1,328 values per sample: a 16-bin colour histogram per RGB channel (48 values) and a
MobileNetV2 embedding (1,280 values), saved in `datasets/image_features (1).csv`.

**Model.** Logistic Regression on standardised features, trained on the neutral and smiling photos
and tested on the surprised photos, an expression it never saw in training. It reaches accuracy and
macro F1 of 1.0 on that test set. A face is accepted only when a face is detected and the top class
probability is at least 0.60.

## 3. Voiceprint verification model

Notebook: `notebooks/audio_voiceprint.ipynb`; code: `audio_pipeline.py`
and `voiceprint_model.py`.

Identifies which team member is speaking. Each of the four members recorded two phrases, "Yes,
approve" and "Confirm transaction", 8 recordings in total.

**Preprocessing and augmentation.** Recordings are converted to mono at 16,000 Hz and trimmed of
silence. Each one is augmented with a time stretch and added background noise, giving 24 samples.

**Features.** 35 values per sample: the mean and standard deviation of 13 MFCCs, spectral roll-off,
RMS energy and total energy, spectral centroid, bandwidth, zero-crossing rate and duration, saved in
`audio_features.csv`.

**Results.** Leave-one-recording-out, scored on the 8 original recordings.

| Model | Accuracy | Macro F1 | Log Loss |
| --- | --- | --- | --- |
| **Logistic Regression** | **0.875** | **0.867** | **0.549** |
| Random Forest | 0.625 | 0.617 | 0.850 |

Logistic Regression was selected and saved as `models/voiceprint_model.joblib`. A voice is accepted
only when the predicted speaker matches the claimed identity with a probability of at least 0.60.

## 4. System simulation (CLI application)

Full guide: [CLI_APP.md](CLI_APP.md). Demo:
[https://youtu.be/7eo35uVNTMc](https://youtu.be/7eo35uVNTMc).

`app.py` runs the whole flow from the command line. A face photo is checked by the facial
recognition model; a recognised face unlocks the product recommendation model, whose prediction is
held; the prediction is displayed only after a voice recording is recognised as the same member by
the voiceprint model. A failed face check stops before any prediction is made, and a failed voice
check discards the prediction without showing it.

```
pip install -r requirements.txt
python app.py                                      # interactive menu
python app.py --face photo.jpg --voice clip.wav    # one transaction
python evaluate_models.py                          # metrics for all three models
```

Because both classifiers always answer with one of the enrolled members, the app also requires the
new photo and recording to be close to that member's enrolled samples, which is what rejects a
person from outside the group.

## Team contributions

| Member | Part |
| --- | --- |
| Steven Kayitare | Data merge and product recommendation model |
| David Ngarambe | Image pipeline and facial recognition model |
| Faith Irakoze | Audio pipeline and voiceprint verification model |
| Marie Honette Ihozo | CLI application and system simulation |

