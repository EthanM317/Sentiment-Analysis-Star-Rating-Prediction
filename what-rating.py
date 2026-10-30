import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


import os
import pickle
import argparse




if __name__ == "__main__":

    if not os.path.exists("model.cache"):
        print("model.cache does not exist. Did you forget to run \"baseline.py\" first?")
        exit(0)
        
    baseline:Pipeline = None
    with open("model.cache", "rb") as file:
        baseline = pickle.load(file)
        
    parser = argparse.ArgumentParser(description="Predict the rating of a review")

    # 2. Add arguments
    # Positional argument (Required by default)
    parser.add_argument("review", type=str, help="The review text")
    
    args = parser.parse_args()
    y_pred = baseline.predict([args.review])
    print(f"Probable rating: {round(y_pred[0])} out of 10 stars")