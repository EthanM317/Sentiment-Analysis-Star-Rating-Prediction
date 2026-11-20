
import pandas as pd
import os
import pickle

from pathlib import Path
from scipy.stats import pearsonr, loguniform, randint

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Root directory
ROOT_DIR = Path(__file__).resolve().parent

# Cache file
BASELINE_MODEL_FILE = ROOT_DIR / "model-baseline.cache"


if __name__ == "__main__":

    # --------------------------------------------------
    # Step 1: Load dataset
    # --------------------------------------------------

    if not (ROOT_DIR / "imdb_ratings.csv").exists():
        print(
            "imdb_ratings.csv does not exist. "
            "Did you forget to run initial-eda.py first?"
        )
        exit(0)

    print("Step 1/5: Loading dataset...", flush=True)

    df = pd.read_csv(ROOT_DIR / "imdb_ratings.csv")

    # Reduce to random sample of 20,000 reviews
    df = df.sample(n=40000, random_state=12)

    X = df["review"].fillna("")
    y = df["rating"]

    print(f"Loaded {len(df):,} reviews.", flush=True)


    # --------------------------------------------------
    # Step 2: Split dataset
    # --------------------------------------------------

    print("\nStep 2/5: Splitting dataset...", flush=True)

    # First reserve 10% for final testing
    X_train_tune, X_test, y_train_tune, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=12
    )

    # Divide remaining 90% into training and tuning datasets
    # 22.2222% of 18,000 = 4,000
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_train_tune,
        y_train_tune,
        test_size=0.222222,
        random_state=12
    )

    print(f"Training reviews: {len(X_train):,}", flush=True)
    print(f"Tuning reviews: {len(X_tune):,}", flush=True)
    print(f"Testing reviews: {len(X_test):,}", flush=True)


    # --------------------------------------------------
    # Step 3: Tune hyperparameters or load cached model
    # --------------------------------------------------

    if BASELINE_MODEL_FILE.exists():

        print(
            "\nStep 3/5: Loading baseline model from cache...",
            flush=True
        )

        with open(BASELINE_MODEL_FILE, "rb") as file:
            baseline = pickle.load(file)

        print("Cached model loaded successfully.", flush=True)

        print("\nSelected hyperparameters:")

        for parameter, value in baseline.best_params_.items():
            print(f"{parameter}: {value}")

    else:

        print("\nStep 3/5: Tuning baseline hyperparameters...", flush=True)

        # Initial TF-IDF + SVR pipeline
        baseline_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("svr", SVR())
        ])

        # Hyperparameter search space
        param_distributions = {

            # SVR hyperparameters
            "svr__C": loguniform(0.1, 100),
            "svr__epsilon": loguniform(0.01, 1.0),
            "svr__kernel": ["linear", "rbf"],

            # TF-IDF hyperparameters
            "tfidf__max_features": randint(2000, 20001),
            "tfidf__ngram_range": [
                (1, 1),
                (1, 2),
                (1, 3)
            ]
        }

        # Randomized hyperparameter search
        random_search = RandomizedSearchCV(
            estimator=baseline_pipeline,
            param_distributions=param_distributions,
            n_iter=25,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=12,
            n_jobs=-1,
            verbose=1,
            refit=False
        )

        # Tune using ONLY the tuning dataset
        random_search.fit(X_tune, y_tune)

        best_params = random_search.best_params_

        print("\nBest hyperparameters selected:", flush=True)

        for parameter, value in best_params.items():
            print(f"{parameter}: {value}", flush=True)

        print(
            f"\nBest cross-validation MAE: "
            f"{-random_search.best_score_:.4f}",
            flush=True
        )

        # --------------------------------------------------
        # Train final baseline using selected parameters
        # --------------------------------------------------

        print("\nTraining final baseline model...", flush=True)

        baseline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=best_params["tfidf__max_features"],
                ngram_range=best_params["tfidf__ngram_range"]
            )),
            ("svr", SVR(
                C=best_params["svr__C"],
                epsilon=best_params["svr__epsilon"],
                kernel=best_params["svr__kernel"]
            ))
        ])

        # Train ONLY using the training dataset
        baseline.fit(X_train, y_train)

        # Store selected hyperparameters in the pipeline
        baseline.best_params_ = best_params

        # Save trained model to cache
        print("\nTraining complete! Caching model...", flush=True)

        with open(BASELINE_MODEL_FILE, "wb") as file:
            pickle.dump(baseline, file)

        print("Model successfully cached.", flush=True)


    # --------------------------------------------------
    # Step 4: Evaluate baseline model
    # --------------------------------------------------

    print("\nStep 4/5: Evaluating baseline model...", flush=True)

    # Predict ratings using the untouched testing dataset
    y_pred = baseline.predict(X_test)

    # Round predictions and restrict them to the 1-10 range
    y_pred = (
        pd.Series(y_pred, index=y_test.index)
        .round()
        .clip(1, 10)
        .astype(int)
    )

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    pearson, p_value = pearsonr(y_test, y_pred)

    print("\nBaseline Rating Prediction Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Pearson correlation: {pearson:.4f}")
    print(f"Pearson p-value: {p_value:.6f}")


    # --------------------------------------------------
    # Step 5: Finish
    # --------------------------------------------------

    print("\nStep 5/5: All done!", flush=True)
