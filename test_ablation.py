
import pandas as pd
import pickle
from pathlib import Path

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Input files
DATASET_FILE = ROOT_DIR / "imdb_ratings.csv"
MODEL_FILE = ROOT_DIR / "model-NaiveBayes.cache"

# Output file
OUTPUT_FILE = ROOT_DIR / "naive_bayes_ablation.csv"


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(rating_model, X_test_features, y_test):

    # Predict ratings
    predictions = rating_model.predict(X_test_features)

    # Round and restrict predictions to the 1-10 rating range
    predictions = predictions.round().clip(1, 10)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    pearson, p_value = pearsonr(y_test, predictions)

    return {
        "mae": mae,
        "mse": mse,
        "pearson": pearson,
        "p_value": p_value
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # Load the cached model
    with open(MODEL_FILE, "rb") as file:
        cached_data = pickle.load(file)

    sentiment_model = cached_data["sentiment_model"]
    original_rating_model = cached_data["rating_model"]

    # Load the rating dataset
    rating_df = pd.read_csv(DATASET_FILE)

    # Reproduce the original 40,000-review sample
    rating_df = rating_df.sample(
        n=40000,
        random_state=12
    )

    X_rating = rating_df["review"]
    y_rating = rating_df["rating"]

    # Reproduce the original 70/20/10 split
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X_rating,
        y_rating,
        test_size=0.10,
        random_state=12
    )

    X_train, X_tune, y_train, y_tune = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.222222,
        random_state=12
    )

    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Identify the positive sentiment class
    positive_index = list(sentiment_model.classes_).index("positive")

    # --------------------------------------------------
    # Original model: positive sentiment probability
    # --------------------------------------------------

    print("\nGenerating original sentiment features...")

    X_train_probability = pd.DataFrame({
        "positive_probability": sentiment_model.predict_proba(
            X_train
        )[:, positive_index]
    })

    X_test_probability = pd.DataFrame({
        "positive_probability": sentiment_model.predict_proba(
            X_test
        )[:, positive_index]
    })

    # Use the original cached regression model
    original_results = evaluate_model(
        original_rating_model,
        X_test_probability,
        y_test
    )

    # --------------------------------------------------
    # Ablation model: binary sentiment classification
    # --------------------------------------------------

    print("\nGenerating binary sentiment features...")

    # Predict sentiment classes (negative or positive)
    y_train_sentiment = sentiment_model.predict(X_train)

    y_test_sentiment = sentiment_model.predict(X_test)

    # Convert sentiment classes into binary values
    X_train_binary = pd.DataFrame({
        "sentiment": [
            1 if sentiment == "positive" else 0
            for sentiment in y_train_sentiment
        ]
    })

    X_test_binary = pd.DataFrame({
        "sentiment": [
            1 if sentiment == "positive" else 0
            for sentiment in y_test_sentiment
        ]
    })

    # Train a new regression model using binary features
    binary_rating_model = LinearRegression(
        fit_intercept=original_rating_model.fit_intercept,
        positive=original_rating_model.positive
    )

    binary_rating_model.fit(
        X_train_binary,
        y_train
    )

    # Evaluate the binary-feature model
    binary_results = evaluate_model(
        binary_rating_model,
        X_test_binary,
        y_test
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n========== ABLATION RESULTS ==========")

    print("\nOriginal model (positive probability):")

    for metric, value in original_results.items():
        print(f"{metric}: {value:.6f}")

    print("\nAblation model (binary sentiment):")

    for metric, value in binary_results.items():
        print(f"{metric}: {value:.6f}")

    # Calculate differences
    print("\n========== METRIC DIFFERENCES ==========")

    for metric in ["mae", "mse", "pearson"]:

        difference = binary_results[metric] - original_results[metric]

        print(f"{metric}: {difference:+.6f}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_df = pd.DataFrame([
        {
            "model": "positive_probability",
            **original_results
        },
        {
            "model": "binary_sentiment",
            **binary_results
        }
    ])

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
