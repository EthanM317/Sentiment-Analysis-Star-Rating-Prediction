
import pandas as pd
import pickle
from pathlib import Path

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------
# File paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

RATING_DATASET = ROOT_DIR / "imdb_ratings.csv"

BASELINE_MODEL_FILE = ROOT_DIR / "model-baseline.cache"
NAIVE_BAYES_MODEL_FILE = ROOT_DIR / "model-NaiveBayes.cache"
VADER_MODEL_FILE = ROOT_DIR / "model-VADER.cache"

OUTPUT_FILE = ROOT_DIR / "three_model_comparison.csv"


# --------------------------------------------------
# Load cached models
# --------------------------------------------------

print("Loading cached models...", flush=True)

# TF-IDF + SVR
with open(BASELINE_MODEL_FILE, "rb") as file:
    baseline_model = pickle.load(file)

# Naive Bayes + Linear Regression
with open(NAIVE_BAYES_MODEL_FILE, "rb") as file:
    nb_data = pickle.load(file)

nb_sentiment_model = nb_data["sentiment_model"]
nb_rating_model = nb_data["rating_model"]

# VADER + Linear Regression
with open(VADER_MODEL_FILE, "rb") as file:
    vader_data = pickle.load(file)

vader_sentiment_model = vader_data["sentiment_model"]
vader_rating_model = vader_data["rating_model"]


# --------------------------------------------------
# Load and split the rating dataset
# --------------------------------------------------

print("Loading rating dataset...", flush=True)

rating_df = pd.read_csv(RATING_DATASET)

rating_df = rating_df.sample(
    n=40000,
    random_state=12
)

X_rating = rating_df["review"].fillna("")
y_rating = rating_df["rating"]

# 90% remaining, 10% test
X_rating_train, X_rating_test, y_rating_train, y_rating_test = train_test_split(
    X_rating,
    y_rating,
    test_size=0.10,
    random_state=12
)

print(f"Test reviews: {len(X_rating_test)}")


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(model_name, predictions):

    # Round predictions and restrict ratings to 1-10
    predictions = predictions.round().clip(1, 10)

    mae = mean_absolute_error(
        y_rating_test,
        predictions
    )

    mse = mean_squared_error(
        y_rating_test,
        predictions
    )

    pearson, p_value = pearsonr(
        y_rating_test,
        predictions
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "Pearson": pearson
    }


results = []


# --------------------------------------------------
# 1. TF-IDF + SVR
# --------------------------------------------------

print("\nEvaluating TF-IDF + SVR...", flush=True)

baseline_predictions = baseline_model.predict(X_rating_test)

results.append(
    evaluate_model(
        "TF-IDF + SVR",
        baseline_predictions
    )
)


# --------------------------------------------------
# 2. Naive Bayes + Linear Regression
# --------------------------------------------------

print("Evaluating Naive Bayes + Linear Regression...", flush=True)

positive_index = list(nb_sentiment_model.classes_).index("positive")

nb_features = nb_sentiment_model.predict_proba(
    X_rating_test
)[:, positive_index]

nb_predictions = nb_rating_model.predict(
    pd.DataFrame({
        "positive_probability": nb_features
    })
)

results.append(
    evaluate_model(
        "Naive Bayes + Linear Regression",
        nb_predictions
    )
)


# --------------------------------------------------
# 3. VADER + Linear Regression
# --------------------------------------------------

print("Evaluating VADER + Linear Regression...", flush=True)

vader_features = [
    vader_sentiment_model.polarity_scores(review)["compound"]
    for review in X_rating_test
]

vader_predictions = vader_rating_model.predict(
    pd.DataFrame({
        "compound_score": vader_features
    })
)

results.append(
    evaluate_model(
        "VADER + Linear Regression",
        vader_predictions
    )
)


# --------------------------------------------------
# Display and save results
# --------------------------------------------------

results_df = pd.DataFrame(results)

# Round displayed metrics
results_df[["MAE", "MSE", "Pearson"]] = results_df[
    ["MAE", "MSE", "Pearson"]
].round(4)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== THREE MODEL COMPARISON ==========")
print(f"Random seed: 12")
print(f"Test reviews: {len(X_rating_test)}")
print()
print(results_df.to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
