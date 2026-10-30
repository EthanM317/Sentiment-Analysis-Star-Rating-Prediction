# Naive-Bayes sentiment classification + regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


#SENTIMENT ANALYSIS COMPONENT
#loading data
df = pd.read_csv("data/sentiment/IMDB-Dataset.csv")
df = df.sample(n=20000, random_state=17) #csv is too large, only going to use a random sample of 20,000 reviews
X = df["review"]
y = df["sentiment"]

#split data into training and test sets
#90% train, 10% test, stratify ensures test & train splits have same positive/negative split as original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y) 

#classification
model = Pipeline([("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())])

#training
model.fit(X_train, y_train)

#testing
y_pred = model.predict(X_test)

#output metrics:
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("------------------")
print(model.predict_proba(["This movie was fantastic. The acting and story were excellent."]))


#load data
ratings_df = pd.read_csv("data/star-rating/imdb_ratings.csv")
ratings_df = ratings_df.sample(n=200, random_state=17) #csv is too large, only going to use a random sample of 20,000 reviews
reviews = ratings_df["review"].fillna("")
ratings_df["sentiment"] = model.predict(reviews)
#generate sentiment predictions and probabilities
probabilities = model.predict_proba(reviews)
class_probabilities = dict(zip(model.classes_, range(len(model.classes_))))

ratings_df["positive_probability"] = probabilities[:, class_probabilities["positive"]]

ratings_df["sentiment_confidence"] = probabilities.max(axis=1)

print(ratings_df[["review", "rating", "sentiment","positive_probability", "sentiment_confidence"]].head())

#--------------------------------------------------------------------------------------------------


#REGRESSION COMPONENT

#load dataset
ratings_df = pd.read_csv("data/star-rating/imdb_ratings.csv")

#sample 20,000 reviews
ratings_df = ratings_df.sample(n=20000, random_state=17)

#perform sentiment analysis using our model
reviews = ratings_df["review"].fillna("")
ratings_df["sentiment"] = model.predict(reviews)
probabilities = model.predict_proba(reviews)
class_probabilities = dict(
    zip(model.classes_, range(len(model.classes_)))
)
ratings_df["positive_probability"] = probabilities[
    :, class_probabilities["positive"]
]
X = ratings_df[["positive_probability"]]
y = ratings_df["rating"]

#splitting into test and train sets (90%train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=12
)

#training
rating_model = LinearRegression()
rating_model.fit(X_train, y_train)

#predicitons
y_pred = rating_model.predict(X_test)
y_pred = y_pred.round().astype(int)

#regression model metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
pearson, p_value = pearsonr(y_test, y_pred)

print("\nREGRESSION RESULTS")
print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"Pearson correlation: {pearson:.3f}")

#display actual vs predicted ratings
results = pd.DataFrame({
    "Actual Rating": y_test,
    "Predicted Rating": y_pred
})

print("\nPREDICTED RATINGS")
print(results.head(10).round(2))