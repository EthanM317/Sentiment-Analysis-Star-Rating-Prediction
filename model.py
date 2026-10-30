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

import os
import pickle

#SENTIMENT ANALYSIS COMPONENT

SENTIMENT_MODEL_FILE = "sentiment-model.cache"

RATING_MODEL_FILE = "rating-model.cache"


def cache_model(path, builder):
    model = None
    if os.path.exists(path):
        print(f"Loading model from cache", flush=True)
        with open(path, "rb") as file:
            model = pickle.load(file)
    else:
        print("Training model (This might take a while)...", flush=True)
        model = builder()
        
        print("Training complete! Cacheing model", flush=True)
        with open(path, "wb") as file:
            pickle.dump(model, file=file)
    return model
            

if __name__ == "__main__":
    if not os.path.exists("IMDB-Dataset.csv"):
        print("IMDB-Dataset.csv does not exist. Did you forget to run \"initial-eda.py\" first?")
        exit(0)
            
    if not os.path.exists("imdb_ratings.csv"):
        print("imdb_ratings.csv does not exist. Did you forget to run \"initial-eda.py\" first?")
        exit(0)
        
    print("Building Sentiment Model", flush=True)
    print("Step 1/4: Loading dataset...", flush=True)
    df = pd.read_csv("IMDB-Dataset.csv")
    df = df.sample(n=20000, random_state=17) #csv is too large, only going to use a random sample of 20,000 reviews
    X = df["review"]
    y = df["sentiment"]
    print(f"Loaded {len(df):,} reviews.", flush=True)
    

    #split data into training and test sets
    #90% train, 10% test, stratify ensures test & train splits have same positive/negative split as original dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17, stratify=y) 

    
    print("Step 2/3: Getting dataset...", flush=True)

    def build_sentimental_model():
        sentiment_model = Pipeline([("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())])
        sentiment_model.fit(X_train, y_train)
        return sentiment_model
    
    sentiment_model:Pipeline = cache_model(SENTIMENT_MODEL_FILE, build_sentimental_model)
    
    print("Step 3/3: Evaluating Model", flush=True)
    y_pred = sentiment_model.predict(X_test)

    #output metrics:
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("------------------")
    
    
    
    print("Building Full Model", flush=True)
    
    print("Step 1/3: Loading dataset...", flush=True)    
    #load dataset
    ratings_df = pd.read_csv("imdb_ratings.csv")

    #sample 20,000 reviews
    ratings_df = ratings_df.sample(n=20000, random_state=17)

    #perform sentiment analysis using our model
    reviews = ratings_df["review"].fillna("")
    ratings_df["sentiment"] = sentiment_model.predict(reviews)
    probabilities = sentiment_model.predict_proba(reviews)
    class_probabilities = dict(
        zip(sentiment_model.classes_, range(len(sentiment_model.classes_)))
    )
    
    ratings_df["positive_probability"] = probabilities[
        :, class_probabilities["positive"]
    ]
    
    ratings_df["sentiment_confidence"] = probabilities.max(axis=1)
    
    X = ratings_df[["positive_probability"]]
    y = ratings_df["rating"]
    
    print(ratings_df[["review", "rating", "sentiment","positive_probability", "sentiment_confidence"]].head())

    #--------------------------------------------------------------------------------------------------

    
    #splitting into test and train sets (90%train, 10% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=12
    )

    print("Step 2/3: Getting dataset...", flush=True)
    #training
    def build_rating_model():
        rating_model = LinearRegression()
        rating_model.fit(X_train, y_train)
        return rating_model
    rating_model:LinearRegression = cache_model(RATING_MODEL_FILE, build_rating_model)
    
    print("Step 3/3: Evaluating Model", flush=True)

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