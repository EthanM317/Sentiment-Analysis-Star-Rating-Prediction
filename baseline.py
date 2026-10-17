
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Step 1/5: Loading dataset...", flush=True)
df = pd.read_csv("imdb_ratings.csv")
#reducing to random sample of 10,000 instead of 137,000 due to long train times & match research paper
df = df.sample(n=10000, random_state=12)
X = df["review"]
y = df["rating"]
print(f"Loaded {len(df):,} reviews.", flush=True)

print("Step 2/5: Splitting dataset...", flush=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=12
)
print(f"Training reviews: {len(X_train):,}", flush=True)
print(f"Testing reviews: {len(X_test):,}", flush=True)

print("Step 3/5: Creating baseline model...", flush=True)
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000)),
    ("svr", SVR(C=4, epsilon=0.01))
])

print("Step 4/5: Training model (this may take a while)...", flush=True)
baseline.fit(X_train, y_train)
print("Training complete!", flush=True)

print("Step 5/5: Predicting and evaluating...", flush=True)
y_pred = baseline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
pearson, p_value = pearsonr(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"Pearson correlation: {pearson:.3f}")
print("All done!", flush=True)
