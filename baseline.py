import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_csv("data/star-rating/imdb_ratings.csv")
x = df["review"]
y = df["rating"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=12
)

# Create the baseline model
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=2000)),
    ("svr", SVR(C=4, epsilon=0.01))
])

# Train the model
baseline.fit(X_train, y_train)

# Predict ratings for unseen reviews
y_pred = baseline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
pearson, p_value = pearsonr(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"Pearson correlation: {pearson:.3f}")