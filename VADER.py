
# VADER sentiment analysis + regression

import pandas as pd
import os
import pickle
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scipy.stats import pearsonr

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.linear_model import LinearRegression


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Cache file
# Separate cache so it does not load the Naive Bayes model
FULL_MODEL_FILE = ROOT_DIR / "model-VADER.cache"


# Cache both models together
def cache_model(path, builder):

    if os.path.exists(path):

        print("Loading complete model from cache...", flush=True)

        with open(path, "rb") as file:
            models = pickle.load(file)

    else:

        print("Training complete model (this might take a while)...", flush=True)

        models = builder()

        print("Training complete! Caching both models...", flush=True)

        with open(path, "wb") as file:
            pickle.dump(models, file)

    return models


# Generate VADER sentiment scores for a collection of reviews
def generate_sentiment_features(reviews, sentiment_model):

    reviews = pd.Series(reviews).fillna("")

    # Generate compound sentiment score for every review
    compound_scores = reviews.apply(
        lambda review: sentiment_model.polarity_scores(review)["compound"]
    )

    # Prepare regression features
    X_features = pd.DataFrame({
        "compound_score": compound_scores
    }, index=reviews.index)

    return X_features


# Build the complete sentiment + rating model
def build_full_model(
    X_train_rating,
    y_train_rating,
    X_tune_rating,
    y_tune_rating
):

    # --------------------------------------------------
    # Initialize VADER sentiment analyzer
    # --------------------------------------------------

    print("\nInitializing VADER sentiment analyzer...", flush=True)

    sentiment_model = SentimentIntensityAnalyzer()


    # --------------------------------------------------
    # Generate sentiment features for rating datasets
    # --------------------------------------------------

    print(
        "\nGenerating VADER sentiment scores for rating datasets...",
        flush=True
    )

    # Generate features for rating training reviews
    X_train_features = generate_sentiment_features(
        X_train_rating,
        sentiment_model
    )

    # Generate features for rating tuning reviews
    X_tune_features = generate_sentiment_features(
        X_tune_rating,
        sentiment_model
    )


    # --------------------------------------------------
    # Tune Linear Regression hyperparameters
    # --------------------------------------------------

    print("\nTuning rating regression model...", flush=True)

    # Initial regression model
    rating_pipeline = LinearRegression()

    # Test all four combinations of the two Boolean parameters
    rating_param_grid = {
        "fit_intercept": [True, False],
        "positive": [True, False]
    }

    # Grid search evaluates every possible combination
    rating_search = GridSearchCV(
        estimator=rating_pipeline,
        param_grid=rating_param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
        refit=False
    )

    # Tune using ONLY the rating tuning dataset
    rating_search.fit(
        X_tune_features,
        y_tune_rating
    )

    best_rating_params = rating_search.best_params_

    print("\nBest regression hyperparameters:", flush=True)

    for parameter, value in best_rating_params.items():
        print(f"{parameter}: {value}", flush=True)

    print(
        f"\nBest regression cross-validation MAE: "
        f"{-rating_search.best_score_:.4f}",
        flush=True
    )


    # --------------------------------------------------
    # Train final regression model
    # --------------------------------------------------

    print("\nTraining final rating regression model...", flush=True)

    rating_model = LinearRegression(
        fit_intercept=best_rating_params["fit_intercept"],
        positive=best_rating_params["positive"]
    )

    # Train ONLY using the rating training dataset
    rating_model.fit(
        X_train_features,
        y_train_rating
    )

    return {
        "sentiment_model": sentiment_model,
        "rating_model": rating_model,
        "best_rating_params": best_rating_params
    }


# Analyze reviews using the trained models
def analyze_reviews(reviews, sentiment_model, rating_model):

    reviews = pd.Series(reviews).fillna("")

    # Generate VADER sentiment features
    X = generate_sentiment_features(
        reviews,
        sentiment_model
    )

    # Extract compound sentiment scores
    compound_scores = X["compound_score"]

    # Determine sentiment labels using VADER's standard thresholds
    sentiment_predictions = compound_scores.apply(
        lambda score:
            "positive" if score >= 0.05
            else "negative" if score <= -0.05
            else "neutral"
    )

    # Predict numerical ratings
    predicted_ratings = rating_model.predict(X)

    # Round predictions and restrict them to the 1-10 range
    rounded_ratings = (
        pd.Series(predicted_ratings, index=reviews.index)
        .round()
        .clip(1, 10)
        .astype(int)
    )

    return pd.DataFrame({
        "review": reviews,
        "predicted_sentiment": sentiment_predictions,
        "compound_score": compound_scores,
        "predicted_rating": predicted_ratings,
        "rounded_rating": rounded_ratings
    })


if __name__ == "__main__":

    # --------------------------------------------------
    # Load and prepare numerical rating dataset
    # --------------------------------------------------

    print("Loading numerical rating dataset...", flush=True)

    ratings_df = pd.read_csv(ROOT_DIR / "imdb_ratings.csv")

    # Sample 20,000 reviews using the same seed as baseline
    ratings_df = ratings_df.sample(
        n=40000,
        random_state=12
    )

    X_rating = ratings_df["review"].fillna("")
    y_rating = ratings_df["rating"]

    # First reserve 10% for final rating testing
    X_train_tune_rating, X_test_rating, y_train_tune_rating, y_test_rating = train_test_split(
        X_rating,
        y_rating,
        test_size=0.10,
        random_state=12
    )

    # Divide remaining 90% into training and tuning datasets
    # 22.2222% of 18,000 = 4,000
    X_train_rating, X_tune_rating, y_train_rating, y_tune_rating = train_test_split(
        X_train_tune_rating,
        y_train_tune_rating,
        test_size=0.222222,
        random_state=12
    )

    print("\nRating dataset split:", flush=True)
    print("Training reviews:", len(X_train_rating))
    print("Tuning reviews:", len(X_tune_rating))
    print("Testing reviews:", len(X_test_rating))


    # --------------------------------------------------
    # Train or load complete model
    # --------------------------------------------------

    models = cache_model(
        FULL_MODEL_FILE,
        lambda: build_full_model(
            X_train_rating,
            y_train_rating,
            X_tune_rating,
            y_tune_rating
        )
    )

    sentiment_model = models["sentiment_model"]
    rating_model = models["rating_model"]


    # --------------------------------------------------
    # Display selected hyperparameters
    # --------------------------------------------------

    print("\nSelected regression hyperparameters:")

    for parameter, value in models["best_rating_params"].items():
        print(f"{parameter}: {value}")


    # --------------------------------------------------
    # Evaluate numerical rating regression model
    # --------------------------------------------------

    print("\nEvaluating rating regression model...", flush=True)

    # Generate VADER sentiment features for rating test reviews
    X_test_features = generate_sentiment_features(
        X_test_rating,
        sentiment_model
    )

    # Predict numerical ratings
    y_pred = rating_model.predict(X_test_features)

    # Round predictions and restrict them to the 1-10 range
    y_pred = (
        pd.Series(y_pred, index=y_test_rating.index)
        .round()
        .clip(1, 10)
        .astype(int)
    )

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_rating, y_pred)

    mse = mean_squared_error(y_test_rating, y_pred)

    pearson, p_value = pearsonr(y_test_rating, y_pred)

    print("\nRating Prediction Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Pearson Correlation: {pearson:.4f}")
    print(f"Pearson p-value: {p_value:.6f}")


    # --------------------------------------------------
    # Example: Analyze individual reviews
    # --------------------------------------------------

    example_reviews = [
        "This movie was absolutely fantastic. The acting and story were amazing!",
        "I hated this movie. It was boring and poorly written.",
        "The movie was okay, but nothing particularly special."
    ]

    results = analyze_reviews(
        example_reviews,
        sentiment_model,
        rating_model
    )

    print("\nExample Review Predictions:")
    print(results.to_string(index=False))
