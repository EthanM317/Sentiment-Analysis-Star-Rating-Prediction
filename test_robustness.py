
import pandas as pd
import pickle
import random
import math
import string
from pathlib import Path

from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Input files
DATASET_FILE = ROOT_DIR / "imdb_ratings.csv"
MODEL_FILE = ROOT_DIR / "full-nb-rating-model-v4.cache"

# Output file
OUTPUT_FILE = ROOT_DIR / "naive_bayes_robustness.csv"

# Experiment settings
TYPO_RATE = 0.10
RANDOM_SEED = 17


# --------------------------------------------------
# Introduce spelling errors into reviews
# --------------------------------------------------

def introduce_typo(word, rng):

    # Identify positions containing letters
    letter_positions = [
        i for i, character in enumerate(word)
        if character.isalpha()
    ]

    # Cannot introduce a typo without at least one letter
    if not letter_positions:
        return word

    typo_type = rng.choice([
        "insertion",
        "deletion",
        "substitution",
        "transposition"
    ])

    position = rng.choice(letter_positions)

    if typo_type == "insertion":

        # Insert a random letter
        new_letter = rng.choice(string.ascii_lowercase)

        return word[:position] + new_letter + word[position:]

    elif typo_type == "deletion":

        # Delete a character
        return word[:position] + word[position + 1:]

    elif typo_type == "substitution":

        # Replace a character with a random letter
        new_letter = rng.choice(string.ascii_lowercase)

        while new_letter == word[position].lower():
            new_letter = rng.choice(string.ascii_lowercase)

        return word[:position] + new_letter + word[position + 1:]

    elif typo_type == "transposition":

        # Swap two adjacent characters
        if len(word) < 2:
            return word

        possible_positions = [
            i for i in range(len(word) - 1)
            if word[i].isalpha() and word[i + 1].isalpha()
        ]

        if not possible_positions:
            return word

        position = rng.choice(possible_positions)

        characters = list(word)

        characters[position], characters[position + 1] = (
            characters[position + 1],
            characters[position]
        )

        return "".join(characters)


def introduce_typos(review, typo_rate, rng):

    words = review.split()

    if not words:
        return review

    # Calculate how many words should receive typos
    number_of_typos = math.ceil(len(words) * typo_rate)

    # Select words without replacement
    selected_positions = rng.sample(
        range(len(words)),
        min(number_of_typos, len(words))
    )

    # Introduce one typo into each selected word
    for position in selected_positions:

        words[position] = introduce_typo(
            words[position],
            rng
        )

    return " ".join(words)


# --------------------------------------------------
# Evaluate the model
# --------------------------------------------------

def evaluate_model(reviews, actual_ratings, sentiment_model, rating_model):

    # Generate positive sentiment probabilities
    positive_index = list(sentiment_model.classes_).index("positive")

    positive_probabilities = sentiment_model.predict_proba(
        reviews
    )[:, positive_index]

    # Prepare features for the regression model
    rating_features = pd.DataFrame({
        "positive_probability": positive_probabilities
    })

    # Predict ratings
    predictions = rating_model.predict(rating_features)

    # Round and restrict predictions to the 1-10 rating range
    predictions = predictions.round().clip(1, 10)

    # Calculate evaluation metrics
    mae = mean_absolute_error(actual_ratings, predictions)

    mse = mean_squared_error(actual_ratings, predictions)

    pearson, p_value = pearsonr(
        actual_ratings,
        predictions
    )

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

    # Load the cached models
    with open(MODEL_FILE, "rb") as file:
        cached_data = pickle.load(file)

    sentiment_model = cached_data["sentiment_model"]
    rating_model = cached_data["rating_model"]

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

    # Reset indices for consistent processing
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"Test reviews: {len(X_test)}")
    print(f"Typo rate: {TYPO_RATE * 100:.0f}%")

    # Evaluate original reviews
    print("\nEvaluating original reviews...")

    original_results = evaluate_model(
        X_test,
        y_test,
        sentiment_model,
        rating_model
    )

    # Create a separate random generator for typo generation
    rng = random.Random(RANDOM_SEED)

    # Modify 10% of the words in each review
    print("\nIntroducing typos into test reviews...")

    X_test_modified = X_test.apply(
        lambda review: introduce_typos(
            str(review),
            TYPO_RATE,
            rng
        )
    )

    # Evaluate modified reviews
    print("\nEvaluating modified reviews...")

    modified_results = evaluate_model(
        X_test_modified,
        y_test,
        sentiment_model,
        rating_model
    )

    # Display results
    print("\n========== ROBUSTNESS RESULTS ==========")

    print("\nOriginal test set:")

    for metric, value in original_results.items():
        print(f"{metric}: {value:.6f}")

    print("\nModified test set (10% typos):")

    for metric, value in modified_results.items():
        print(f"{metric}: {value:.6f}")

    # Calculate changes in evaluation metrics
    print("\n========== METRIC DIFFERENCES ==========")

    for metric in ["mae", "mse", "pearson"]:

        difference = (
            modified_results[metric] - original_results[metric]
        )

        print(
            f"{metric}: {difference:+.6f}"
        )

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            "condition": "original",
            **original_results
        },
        {
            "condition": "10_percent_typos",
            **modified_results
        }
    ])

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
