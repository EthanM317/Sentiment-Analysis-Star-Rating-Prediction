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
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the rating of a review")

    # 2. Add arguments
    # Positional argument (Required by default)
    parser.add_argument("review", type=str, help="The review text")
    
    args = parser.parse_args()
    
    if not os.path.exists("sentiment-model.cache"):
        print("sentiment-model.cache does not exist. Did you forget to run \"model.py\" first?")
        exit(0)
    
    if not os.path.exists("rating-model.cache"):
        print("rating-model.cache does not exist. Did you forget to run \"model.py\" first?")
        exit(0)


            
    rating_model:LinearRegression = None
    with open("rating-model.cache", "rb") as file:
        rating_model = pickle.load(file)
           
    sentiment_model:Pipeline = None
    with open("sentiment-model.cache", "rb") as file:
        sentiment_model = pickle.load(file)

    prob = sentiment_model.predict_proba([args.review])
    rating = rating_model.predict([[prob[0, 1]]])
    print(f"Probable rating: {round(rating[0])} out of 10 stars")