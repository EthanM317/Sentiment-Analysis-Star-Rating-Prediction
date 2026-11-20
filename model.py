
# Naive-Bayes sentiment classification + regression

import pandas as pd
import os
import pickle

from scipy.stats import pearsonr, loguniform, randint

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LinearRegression


# Cache file
FULL_MODEL_FILE = "model-NaiveBayes.cache"


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


# Build the complete sentiment + rating model
def build_full_model(
    X_train_sentiment,
    y_train_sentiment,
    X_tune,
    y_tune,
    X_train_rating,
    y_train_rating
):

    # Initial sentiment classification pipeline
    sentiment_pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Hyperparameter search space
    param_distributions = {
        "classifier__alpha": loguniform(0.001, 10),
        "vectorizer__max_features": randint(1000, 20001),
        "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]
    }

    # Tune hyperparameters using the separate tuning dataset
    random_search = RandomizedSearchCV(
        estimator=sentiment_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring="f1_macro",
        random_state=17,
        n_jobs=-1,
        verbose=1,
        refit=False
    )

    print("\nTuning sentiment classifier...", flush=True)

    random_search.fit(X_tune, y_tune)

    best_params = random_search.best_params_

    print("\nBest sentiment parameters:", best_params, flush=True)

    # Build the final sentiment classifier using the best parameters
    sentiment_model = Pipeline([
        ("vectorizer", CountVectorizer(
            max_features=best_params["vectorizer__max_features"],
            ngram_range=best_params["vectorizer__ngram_range"]
        )),
        ("classifier", MultinomialNB(
            alpha=best_params["classifier__alpha"]
        ))
    ])

    # Train the final sentiment classifier using ONLY the training dataset
    print("\nTraining final sentiment classifier...", flush=True)

    sentiment_model.fit(X_train_sentiment, y_train_sentiment)

    # Extract positive sentiment probabilities from the rating training reviews
    print("\nGenerating sentiment probabilities for rating training data...", flush=True)

    probabilities = sentiment_model.predict_proba(
        X_train_rating.fillna("")
    )

    # Find the column corresponding to positive sentiment
    positive_index = list(sentiment_model.classes_).index("positive")

    positive_probability = probabilities[:, positive_index]

    # Prepare features for the regression model
    X_train_features = pd.DataFrame({
        "positive_probability": positive_probability
    }, index=X_train_rating.index)

    # Train the numerical rating regression model
    print("\nTraining rating regression model...", flush=True)

    rating_model = LinearRegression()

    rating_model.fit(X_train_features, y_train_rating)

    return {
        "sentiment_model": sentiment_model,
        "rating_model": rating_model,
        "best_params": best_params
    }


# Analyze reviews using the trained models
def analyze_reviews(reviews, sentiment_model, rating_model):

    reviews = pd.Series(reviews).fillna("")

    # Predict sentiment
    sentiment_predictions = sentiment_model.predict(reviews)

    # Predict sentiment probabilities
    probabilities = sentiment_model.predict_proba(reviews)

    positive_index = list(sentiment_model.classes_).index("positive")

    positive_probability = probabilities[:, positive_index]

    # Prepare regression features
    X = pd.DataFrame({
        "positive_probability": positive_probability
    })

    # Predict numerical ratings
    predicted_ratings = rating_model.predict(X)

    # Round predictions and restrict them to the 1-10 range
    rounded_ratings = (
        pd.Series(predicted_ratings)
        .round()
        .clip(1, 10)
        .astype(int)
    )

    return pd.DataFrame({
        "review": reviews,
        "predicted_sentiment": sentiment_predictions,
        "positive_probability": positive_probability,
        "predicted_rating": predicted_ratings,
        "rounded_rating": rounded_ratings
    })


if __name__ == "__main__":

    # --------------------------------------------------
    # Load and prepare sentiment classification dataset
    # --------------------------------------------------

    print("Loading sentiment dataset...", flush=True)

    sentiment_df = pd.read_csv("IMDB-Dataset.csv")

    # Sample 20,000 reviews
    sentiment_df = sentiment_df.sample(
        n=20000,
        random_state=17
    )

    X = sentiment_df["review"].fillna("")
    y = sentiment_df["sentiment"]

    # First reserve 10% for final sentiment testing
    X_train_tune, X_test_sentiment, y_train_tune, y_test_sentiment = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=17,
        stratify=y
    )

    # Divide the remaining 90% into training and tuning datasets
    # 22.2222% of 18,000 = 4,000
    X_train_sentiment, X_tune, y_train_sentiment, y_tune = train_test_split(
        X_train_tune,
        y_train_tune,
        test_size=0.222222,
        random_state=17,
        stratify=y_train_tune
    )

    print("\nSentiment dataset split:", flush=True)
    print("Training reviews:", len(X_train_sentiment))
    print("Tuning reviews:", len(X_tune))
    print("Testing reviews:", len(X_test_sentiment))


    # --------------------------------------------------
    # Load and prepare numerical rating dataset
    # --------------------------------------------------

    print("\nLoading numerical rating dataset...", flush=True)

    ratings_df = pd.read_csv("imdb_ratings.csv")

    # Sample 20,000 reviews using the same seed as the baseline
    ratings_df = ratings_df.sample(
        n=20000,
        random_state=12
    )

    X_rating = ratings_df["review"].fillna("")
    y_rating = ratings_df["rating"]

    # Split into training and testing datasets
    X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(
        X_rating,
        y_rating,
        test_size=0.10,
        random_state=12
    )

    print("\nRating dataset split:", flush=True)
    print("Training reviews:", len(X_train_rating))
    print("Testing reviews:", len(X_test_rating))


    # --------------------------------------------------
    # Train or load complete model
    # --------------------------------------------------

    models = cache_model(
        FULL_MODEL_FILE,
        lambda: build_full_model(
            X_train_sentiment,
            y_train_sentiment,
            X_tune,
            y_tune,
            X_train_rating,
            y_train_rating
        )
    )

    sentiment_model = models["sentiment_model"]
    rating_model = models["rating_model"]


    # --------------------------------------------------
    # Evaluate sentiment classification model
    # --------------------------------------------------

    print("\nEvaluating sentiment classifier...", flush=True)

    sentiment_predictions = sentiment_model.predict(
        X_test_sentiment
    )

    sentiment_accuracy = accuracy_score(
        y_test_sentiment,
        sentiment_predictions
    )

    print("\nSentiment Classification Accuracy:")
    print(f"{sentiment_accuracy:.4f}")

    print("\nSentiment Classification Report:")
    print(
        classification_report(
            y_test_sentiment,
            sentiment_predictions
        )
    )


    # --------------------------------------------------
    # Evaluate numerical rating regression model
    # --------------------------------------------------

    print("\nEvaluating rating regression model...", flush=True)

    # Generate positive sentiment probabilities for rating test reviews
    test_probabilities = sentiment_model.predict_proba(
        X_test_rating.fillna("")
    )

    positive_index = list(sentiment_model.classes_).index("positive")

    positive_probability = test_probabilities[:, positive_index]

    # Prepare regression features
    X_test_features = pd.DataFrame({
        "positive_probability": positive_probability
    }, index=X_test_rating.index)

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
