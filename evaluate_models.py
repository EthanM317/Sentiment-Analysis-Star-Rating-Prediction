
# Evaluate all three models over five random seeds

import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import pearsonr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# Root directory and cache files
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

BASELINE_MODEL_FILE = ROOT_DIR / "model-baseline.cache"
NB_MODEL_FILE = ROOT_DIR / "model-Naive-Bayes.cache"
VADER_MODEL_FILE = ROOT_DIR / "model-VADER.cache"

DATASET_FILE = ROOT_DIR / "imdb_ratings.csv"


# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

SEEDS = [12, 17, 23, 42, 100]

TEST_SIZE = 4000

# Original sampling configuration used to train the models
ORIGINAL_SAMPLE_SIZE = 40000
ORIGINAL_SAMPLE_SEED = 12


# --------------------------------------------------
# Load cached models
# --------------------------------------------------

def load_model(path):

    print(f"\nLoading model: {path.name}", flush=True)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find model cache: {path}"
        )

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


# Load baseline model
baseline_model = load_model(BASELINE_MODEL_FILE)


# Load Naive Bayes + regression models
nb_models = load_model(NB_MODEL_FILE)

nb_sentiment_model = nb_models["sentiment_model"]
nb_rating_model = nb_models["rating_model"]


# Load VADER + regression models
vader_models = load_model(VADER_MODEL_FILE)

vader_sentiment_model = vader_models["sentiment_model"]
vader_rating_model = vader_models["rating_model"]


# --------------------------------------------------
# Load numerical rating dataset
# --------------------------------------------------

print("\nLoading numerical rating dataset...", flush=True)

ratings_df = pd.read_csv(DATASET_FILE)


# --------------------------------------------------
# Exclude the original 40,000 reviews
# --------------------------------------------------

print("\nReproducing original training sample...", flush=True)

original_sample = ratings_df.sample(
    n=ORIGINAL_SAMPLE_SIZE,
    random_state=ORIGINAL_SAMPLE_SEED
)

# Exclude every review used in the original experiment
# This prevents overlap with the original training,
# tuning, and testing datasets.

evaluation_df = ratings_df.loc[
    ~ratings_df.index.isin(original_sample.index)
].copy()

print(
    f"Reviews available for evaluation: {len(evaluation_df)}",
    flush=True
)

if len(evaluation_df) < TEST_SIZE:
    raise ValueError(
        "Not enough reviews available for evaluation."
    )


# --------------------------------------------------
# Model prediction functions
# --------------------------------------------------

def predict_baseline(reviews):

    # TF-IDF + SVR directly predicts numerical ratings
    predictions = baseline_model.predict(reviews)

    return (
        pd.Series(predictions)
        .round()
        .clip(1, 10)
        .astype(int)
        .to_numpy()
    )


def predict_naive_bayes(reviews):

    # Obtain positive sentiment probabilities
    probabilities = nb_sentiment_model.predict_proba(reviews)

    # Identify the positive class column
    classes = nb_sentiment_model.classes_

    positive_index = list(classes).index("positive")

    positive_probabilities = probabilities[:, positive_index]

    # Prepare regression features
    X_features = pd.DataFrame({
        "positive_probability": positive_probabilities
    })

    # Predict numerical ratings
    predictions = nb_rating_model.predict(X_features)

    return (
        pd.Series(predictions)
        .round()
        .clip(1, 10)
        .astype(int)
        .to_numpy()
    )


def predict_vader(reviews):

    # Generate VADER compound scores
    compound_scores = reviews.fillna("").apply(
        lambda review:
            vader_sentiment_model.polarity_scores(review)["compound"]
    )

    # Prepare regression features
    X_features = pd.DataFrame({
        "compound_score": compound_scores.to_numpy()
    })

    # Predict numerical ratings
    predictions = vader_rating_model.predict(X_features)

    return (
        pd.Series(predictions)
        .round()
        .clip(1, 10)
        .astype(int)
        .to_numpy()
    )


# --------------------------------------------------
# Evaluate a model
# --------------------------------------------------

def evaluate_model(y_actual, y_pred):

    mae = mean_absolute_error(y_actual, y_pred)

    mse = mean_squared_error(y_actual, y_pred)

    pearson, p_value = pearsonr(y_actual, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "Pearson": pearson
    }


# --------------------------------------------------
# Store results from all five seeds
# --------------------------------------------------

results = {
    "TF-IDF + SVR": {
        "MAE": [],
        "MSE": [],
        "Pearson": []
    },
    "Naive Bayes + Regression": {
        "MAE": [],
        "MSE": [],
        "Pearson": []
    },
    "VADER + Regression": {
        "MAE": [],
        "MSE": [],
        "Pearson": []
    }
}


# --------------------------------------------------
# Run evaluation over five seeds
# --------------------------------------------------

for seed in SEEDS:

    print("\n" + "=" * 60)
    print(f"EVALUATION SEED: {seed}")
    print("=" * 60)

    # Select a new 4,000-review test sample
    test_df = evaluation_df.sample(
        n=TEST_SIZE,
        random_state=seed
    )

    X_test = test_df["review"].fillna("")
    y_test = test_df["rating"].to_numpy()


    # ----------------------------------------------
    # Evaluate TF-IDF + SVR
    # ----------------------------------------------

    print("\nEvaluating TF-IDF + SVR...", flush=True)

    baseline_predictions = predict_baseline(X_test)

    baseline_results = evaluate_model(
        y_test,
        baseline_predictions
    )

    for metric, value in baseline_results.items():
        results["TF-IDF + SVR"][metric].append(value)

        print(f"{metric}: {value:.4f}")


    # ----------------------------------------------
    # Evaluate Naive Bayes + Regression
    # ----------------------------------------------

    print("\nEvaluating Naive Bayes + Regression...", flush=True)

    nb_predictions = predict_naive_bayes(X_test)

    nb_results = evaluate_model(
        y_test,
        nb_predictions
    )

    for metric, value in nb_results.items():
        results["Naive Bayes + Regression"][metric].append(value)

        print(f"{metric}: {value:.4f}")


    # ----------------------------------------------
    # Evaluate VADER + Regression
    # ----------------------------------------------

    print("\nEvaluating VADER + Regression...", flush=True)

    vader_predictions = predict_vader(X_test)

    vader_results = evaluate_model(
        y_test,
        vader_predictions
    )

    for metric, value in vader_results.items():
        results["VADER + Regression"][metric].append(value)

        print(f"{metric}: {value:.4f}")


# --------------------------------------------------
# Calculate mean and standard deviation
# --------------------------------------------------

print("\n\n" + "=" * 75)
print("FINAL RESULTS: MEAN ± STANDARD DEVIATION")
print("=" * 75)

for model_name, metrics in results.items():

    print(f"\n{model_name}")
    print("-" * 50)

    for metric, values in metrics.items():

        mean = np.mean(values)

        # Sample standard deviation
        uncertainty = np.std(values, ddof=1)

        print(
            f"{metric}: {mean:.4f} ± {uncertainty:.4f}"
        )


print("\nEvaluation complete!", flush=True)
