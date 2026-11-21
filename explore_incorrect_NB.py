
# Find 50 incorrectly predicted reviews using Naive Bayes + Regression

import pickle
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Root directory and files
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent

MODEL_FILE = ROOT_DIR / "model-NaiveBayes.cache"
DATASET_FILE = ROOT_DIR / "imdb_ratings.csv"
OUTPUT_FILE = ROOT_DIR / "incorrect_predictions.txt"


# --------------------------------------------------
# Load cached model
# --------------------------------------------------

print("Loading Naive Bayes model from cache...", flush=True)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Could not find model cache: {MODEL_FILE}"
    )

with open(MODEL_FILE, "rb") as file:
    models = pickle.load(file)

sentiment_model = models["sentiment_model"]
rating_model = models["rating_model"]


# --------------------------------------------------
# Load and prepare numerical rating dataset
# --------------------------------------------------

print("Loading numerical rating dataset...", flush=True)

ratings_df = pd.read_csv(DATASET_FILE)

# Reproduce the original 40,000-review sample
ratings_df = ratings_df.sample(
    n=40000,
    random_state=12
)

X_rating = ratings_df["review"].fillna("")
y_rating = ratings_df["rating"]


# --------------------------------------------------
# Reproduce original 70/20/10 split
# --------------------------------------------------

# First reserve 10% for final testing
X_train_tune_rating, X_test_rating, y_train_tune_rating, y_test_rating = train_test_split(
    X_rating,
    y_rating,
    test_size=0.10,
    random_state=12
)

# Divide remaining 90% into training and tuning
X_train_rating, X_tune_rating, y_train_rating, y_tune_rating = train_test_split(
    X_train_tune_rating,
    y_train_tune_rating,
    test_size=0.222222,
    random_state=12
)


# --------------------------------------------------
# Generate sentiment features
# --------------------------------------------------

print("Generating sentiment predictions...", flush=True)

# Obtain positive sentiment probabilities
probabilities = sentiment_model.predict_proba(X_test_rating)

# Identify the positive class column
classes = sentiment_model.classes_

positive_index = list(classes).index("positive")

positive_probabilities = probabilities[:, positive_index]

# Prepare regression features
X_test_features = pd.DataFrame({
    "positive_probability": positive_probabilities
})


# --------------------------------------------------
# Predict numerical ratings
# --------------------------------------------------

print("Predicting numerical ratings...", flush=True)

predictions = rating_model.predict(X_test_features)

# Round predictions and restrict them to the 1-10 range
predictions = (
    pd.Series(predictions, index=X_test_rating.index)
    .round()
    .clip(1, 10)
    .astype(int)
)


# --------------------------------------------------
# Identify incorrectly predicted reviews
# --------------------------------------------------

results = pd.DataFrame({
    "predicted_score": predictions,
    "actual_score": y_test_rating,
    "review": X_test_rating
})

# Keep only reviews where the predicted score is incorrect
incorrect_reviews = results[
    results["predicted_score"] != results["actual_score"]
].copy()

print(
    f"\nIncorrect predictions found: {len(incorrect_reviews)}",
    flush=True
)

# Select up to 50 incorrectly predicted reviews
incorrect_reviews = incorrect_reviews.head(50)


# --------------------------------------------------
# Write results to text file
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    for _, row in incorrect_reviews.iterrows():

        file.write(
            f"{row['predicted_score']} "
            f"{row['actual_score']} "
            f"{row['review']}\n"
        )


print(
    f"\nSaved {len(incorrect_reviews)} incorrect predictions to:",
    flush=True
)

print(OUTPUT_FILE, flush=True)
