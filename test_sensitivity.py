
import pandas as pd
import pickle
from pathlib import Path

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error
)


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Input files
SENTIMENT_DATASET = ROOT_DIR / "IMDB-Dataset.csv"
RATING_DATASET = ROOT_DIR / "imdb_ratings.csv"
MODEL_FILE = ROOT_DIR / "model-NaiveBayes.cache"

# Output files
SENTIMENT_OUTPUT = ROOT_DIR / "naive_bayes_sensitivity.csv"
REGRESSION_OUTPUT = ROOT_DIR / "regression_sensitivity.csv"


# Load the previously selected hyperparameters
with open(MODEL_FILE, "rb") as file:
    cached_data = pickle.load(file)

best_sentiment_params = cached_data["best_sentiment_params"]
best_rating_params = cached_data["best_rating_params"]

sentiment_model = cached_data["sentiment_model"]


# Retrieve a parameter regardless of its pipeline prefix
def get_param(params, name, default):
    for key, value in params.items():
        if key.split("__")[-1] == name:
            return value

    return default


# Get the best sentiment hyperparameters
best_alpha = get_param(best_sentiment_params, "alpha", 1.0)
best_max_features = get_param(best_sentiment_params, "max_features", None)
best_ngram_range = get_param(best_sentiment_params, "ngram_range", (1, 1))
best_min_df = get_param(best_sentiment_params, "min_df", 1)
best_fit_prior = get_param(best_sentiment_params, "fit_prior", True)


# Get the best regression hyperparameters
best_fit_intercept = get_param(best_rating_params, "fit_intercept", True)
best_positive = get_param(best_rating_params, "positive", False)


# Load and split the sentiment dataset
sentiment_df = pd.read_csv(SENTIMENT_DATASET)

sentiment_df = sentiment_df.sample(
    n=20000,
    random_state=17
)

X_sentiment = sentiment_df["review"]
y_sentiment = sentiment_df["sentiment"]

# 10% test, 90% remaining
X_sentiment_remaining, X_sentiment_test, y_sentiment_remaining, y_sentiment_test = train_test_split(
    X_sentiment,
    y_sentiment,
    test_size=0.10,
    random_state=17,
    stratify=y_sentiment
)

# 70% training, 20% tuning, 10% test
X_sentiment_train, X_sentiment_tune, y_sentiment_train, y_sentiment_tune = train_test_split(
    X_sentiment_remaining,
    y_sentiment_remaining,
    test_size=0.222222,
    random_state=17,
    stratify=y_sentiment_remaining
)


# Create a function to build and evaluate a sentiment model
def evaluate_sentiment_model(params):

    model = Pipeline([
        ("vectorizer", CountVectorizer(
            max_features=params["max_features"],
            ngram_range=params["ngram_range"],
            min_df=params["min_df"]
        )),
        ("classifier", MultinomialNB(
            alpha=params["alpha"],
            fit_prior=params["fit_prior"]
        ))
    ])

    # Train using the original training set
    model.fit(X_sentiment_train, y_sentiment_train)

    # Evaluate using the same test set
    predictions = model.predict(X_sentiment_test)

    accuracy = accuracy_score(y_sentiment_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_sentiment_test,
        predictions,
        labels=["negative", "positive"],
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "negative_precision": precision[0],
        "negative_recall": recall[0],
        "negative_f1": f1[0],
        "positive_precision": precision[1],
        "positive_recall": recall[1],
        "positive_f1": f1[1],
        "macro_f1": (f1[0] + f1[1]) / 2
    }


# Store the best sentiment parameters
best_params = {
    "alpha": best_alpha,
    "max_features": best_max_features,
    "ngram_range": best_ngram_range,
    "min_df": best_min_df,
    "fit_prior": best_fit_prior
}


# Define the values to test for each hyperparameter
alpha_values = sorted(set([
    max(0.001, best_alpha / 10),
    max(0.001, best_alpha / 2),
    best_alpha,
    best_alpha * 2,
    best_alpha * 10
]))

if best_max_features is None:
    max_features_values = [5000, 10000, 15000, 20000]
else:
    max_features_values = sorted(set([
        max(1000, int(best_max_features * 0.5)),
        best_max_features,
        min(20000, int(best_max_features * 1.5))
    ]))

ngram_range_values = [
    (1, 1),
    (1, 2),
    (1, 3)
]

min_df_values = sorted(set([
    max(1, best_min_df - 2),
    best_min_df,
    best_min_df + 2
]))

fit_prior_values = [False, True]


# Store sensitivity results
sentiment_results = []


# Test one hyperparameter at a time
sensitivity_tests = {
    "alpha": alpha_values,
    "max_features": max_features_values,
    "ngram_range": ngram_range_values,
    "min_df": min_df_values,
    "fit_prior": fit_prior_values
}


for parameter, values in sensitivity_tests.items():

    print(f"\nTesting sentiment hyperparameter: {parameter}")

    for value in values:

        # Start with the best parameters
        current_params = best_params.copy()

        # Modify only the parameter currently being tested
        current_params[parameter] = value

        metrics = evaluate_sentiment_model(current_params)

        result = {
            "parameter": parameter,
            "value": str(value),
            "is_best_value": value == best_params[parameter],
            **metrics
        }

        sentiment_results.append(result)

        print(
            f"{parameter} = {value} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )


# Save sentiment sensitivity results
sentiment_results_df = pd.DataFrame(sentiment_results)

sentiment_results_df.to_csv(
    SENTIMENT_OUTPUT,
    index=False
)

print("\nSentiment sensitivity results saved.")
print(sentiment_results_df.to_string(index=False))


# --------------------------------------------------
# Regression hyperparameter sensitivity
# --------------------------------------------------

# Load and split the rating dataset
rating_df = pd.read_csv(RATING_DATASET)

rating_df = rating_df.sample(
    n=40000,
    random_state=12
)

X_rating = rating_df["review"]
y_rating = rating_df["rating"]

# 10% test, 90% remaining
X_rating_remaining, X_rating_test, y_rating_remaining, y_rating_test = train_test_split(
    X_rating,
    y_rating,
    test_size=0.10,
    random_state=12
)

# 70% training, 20% tuning, 10% test
X_rating_train, X_rating_tune, y_rating_train, y_rating_tune = train_test_split(
    X_rating_remaining,
    y_rating_remaining,
    test_size=0.222222,
    random_state=12
)


# Use the cached sentiment model to generate rating features
X_rating_train_features = pd.DataFrame({
    "positive_probability": sentiment_model.predict_proba(X_rating_train)[
        :, list(sentiment_model.classes_).index("positive")
    ]
})

X_rating_test_features = pd.DataFrame({
    "positive_probability": sentiment_model.predict_proba(X_rating_test)[
        :, list(sentiment_model.classes_).index("positive")
    ]
})


# Test all combinations of the regression hyperparameters
regression_results = []

for fit_intercept in [False, True]:

    for positive in [False, True]:

        model = LinearRegression(
            fit_intercept=fit_intercept,
            positive=positive
        )

        # Train using the same training set
        model.fit(
            X_rating_train_features,
            y_rating_train
        )

        # Evaluate using the same test set
        predictions = model.predict(X_rating_test_features)

        # Round and restrict predictions to the 1-10 rating range
        predictions = predictions.round().clip(1, 10)

        mae = mean_absolute_error(y_rating_test, predictions)
        mse = mean_squared_error(y_rating_test, predictions)

        pearson, p_value = pearsonr(y_rating_test, predictions)

        regression_results.append({
            "fit_intercept": fit_intercept,
            "positive": positive,
            "is_best_combination": (
                fit_intercept == best_fit_intercept
                and positive == best_positive
            ),
            "mae": mae,
            "mse": mse,
            "pearson": pearson,
            "p_value": p_value
        })

        print(
            f"\nRegression: fit_intercept={fit_intercept}, "
            f"positive={positive}"
        )

        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"Pearson: {pearson:.4f}")
        print(f"P-value: {p_value:.6f}")


# Save regression sensitivity results
regression_results_df = pd.DataFrame(regression_results)

regression_results_df.to_csv(
    REGRESSION_OUTPUT,
    index=False
)

print("\nRegression sensitivity results saved.")
print(regression_results_df.to_string(index=False))
