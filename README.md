# Formative 2: Multimodal Data Preprocessing

**Group 11** | Machine Learning Pipeline | BSE (Machine Learning), African Leadership University

A Multimodal Biometric Authentication and Product Recommendation System. A user is checked by a
facial recognition model, the product recommendation model runs, and the prediction is only shown
after the voiceprint verification model confirms the same identity. If either check fails, access is
denied.

## Links

| Item | Link |
| --- | --- |
| Demo video | To be added |
| Report | To be added |
| Contribution tracker | To be added |

## Repository structure

```
datasets/
  customer_transactions - customer_transactions.csv
  customer_social_profiles - customer_social_profiles.csv
  merged_customer_data.csv
  image_features.csv
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

Done by Steven Kayitare.

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

To be added.

## 3. Voiceprint verification model

To be added.

## 4. System simulation (CLI application)

Done by Marie Honette Ihozo. Full guide: [CLI_APP.md](CLI_APP.md).

`app.py` runs the whole flow from the command line. A face photo is checked by the facial
recognition model; if it is accepted, a voice recording must be recognised as the same member by
the voiceprint model; only then does the product recommendation model run and show its prediction.
Any failed check prints `ACCESS DENIED` and no prediction is made.

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
| To be added | Facial recognition model |
| To be added | Voiceprint verification model |
| Marie Honette Ihozo | CLI application and system simulation |

