
# Naive-Bayes sentiment classification + regression

import pandas as pd
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, loguniform, randint

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV
)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LinearRegression


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Cache file
FULL_MODEL_FILE = ROOT_DIR / "model-NaiveBayes.cache"


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
    X_tune_sentiment,
    y_tune_sentiment,
    X_train_rating,
    y_train_rating,
    X_tune_rating,
    y_tune_rating
):

    # --------------------------------------------------
    # Tune Naive Bayes sentiment classifier
    # --------------------------------------------------

    print("\nTuning sentiment classifier...", flush=True)

    # Initial sentiment classification pipeline
    sentiment_pipeline = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Sentiment hyperparameter search space
    sentiment_param_distributions = {

        "classifier__alpha": loguniform(0.001, 10),

        "vectorizer__max_features": randint(1000, 20001),

        "vectorizer__ngram_range": [
            (1, 1),
            (1, 2),
            (1, 3)
        ],

        "vectorizer__min_df": randint(1, 11),

        "classifier__fit_prior": [True, False]
    }

    # Randomized hyperparameter search
    sentiment_search = RandomizedSearchCV(
        estimator=sentiment_pipeline,
        param_distributions=sentiment_param_distributions,
        n_iter=20,
        cv=3,
        scoring="f1_macro",
        random_state=17,
        n_jobs=-1,
        verbose=1,
        refit=False
    )

    # Tune using ONLY the sentiment tuning dataset
    sentiment_search.fit(
        X_tune_sentiment,
        y_tune_sentiment
    )

    best_sentiment_params = sentiment_search.best_params_

    print("\nBest sentiment hyperparameters:", flush=True)

    for parameter, value in best_sentiment_params.items():
        print(f"{parameter}: {value}", flush=True)

    print(
        f"\nBest sentiment cross-validation F1: "
        f"{sentiment_search.best_score_:.4f}",
        flush=True
    )

    # Build final sentiment classifier using selected parameters
    sentiment_model = Pipeline([
        ("vectorizer", CountVectorizer(
            max_features=best_sentiment_params["vectorizer__max_features"],
            ngram_range=best_sentiment_params["vectorizer__ngram_range"],
            min_df=best_sentiment_params["vectorizer__min_df"]
        )),

        ("classifier", MultinomialNB(
            alpha=best_sentiment_params["classifier__alpha"],
            fit_prior=best_sentiment_params["classifier__fit_prior"]
        ))
    ])

    # Train final sentiment classifier using ONLY training data
    print("\nTraining final sentiment classifier...", flush=True)

    sentiment_model.fit(
        X_train_sentiment,
        y_train_sentiment
    )


    # --------------------------------------------------
    # Generate sentiment features for rating datasets
    # --------------------------------------------------

    print(
        "\nGenerating sentiment probabilities for rating datasets...",
        flush=True
    )

    # Generate features for rating training reviews
    train_probabilities = sentiment_model.predict_proba(
        X_train_rating.fillna("")
    )

    # Generate features for rating tuning reviews
    tune_probabilities = sentiment_model.predict_proba(
        X_tune_rating.fillna("")
    )

    # Find the column corresponding to positive sentiment
    positive_index = list(sentiment_model.classes_).index("positive")

    train_positive_probability = train_probabilities[:, positive_index]

    tune_positive_probability = tune_probabilities[:, positive_index]

    # Prepare training features
    X_train_features = pd.DataFrame({
        "positive_probability": train_positive_probability
    }, index=X_train_rating.index)

    # Prepare tuning features
    X_tune_features = pd.DataFrame({
        "positive_probability": tune_positive_probability
    }, index=X_tune_rating.index)


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
        "best_sentiment_params": best_sentiment_params,
        "best_rating_params": best_rating_params
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

    sentiment_df = pd.read_csv(ROOT_DIR / "IMDB-Dataset.csv")

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

    # Divide remaining 90% into training and tuning datasets
    # 22.2222% of 18,000 = 4,000
    X_train_sentiment, X_tune_sentiment, y_train_sentiment, y_tune_sentiment = train_test_split(
        X_train_tune,
        y_train_tune,
        test_size=0.222222,
        random_state=17,
        stratify=y_train_tune
    )

    print("\nSentiment dataset split:", flush=True)
    print("Training reviews:", len(X_train_sentiment))
    print("Tuning reviews:", len(X_tune_sentiment))
    print("Testing reviews:", len(X_test_sentiment))


    # --------------------------------------------------
    # Load and prepare numerical rating dataset
    # --------------------------------------------------

    print("\nLoading numerical rating dataset...", flush=True)

    ratings_df = pd.read_csv(ROOT_DIR / "imdb_ratings.csv")

    # Sample 20,000 reviews using the same seed as baseline
    ratings_df = ratings_df.sample(
        n=20000,
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
            X_train_sentiment,
            y_train_sentiment,
            X_tune_sentiment,
            y_tune_sentiment,
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

    print("\nSelected sentiment hyperparameters:")

    for parameter, value in models["best_sentiment_params"].items():
        print(f"{parameter}: {value}")

    print("\nSelected regression hyperparameters:")

    for parameter, value in models["best_rating_params"].items():
        print(f"{parameter}: {value}")


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

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test_sentiment,
        sentiment_predictions,
        labels=["negative", "positive"]
    )

    print("\nConfusion Matrix:")
    print(cm)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"]
    ).plot()

    plt.title("Naive Bayes Sentiment Classification")
    plt.show()


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
