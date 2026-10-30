# Sentiment-Analysis-Star-Rating-Prediction
Final Project for CSCI 3052U


## How to Use

- Clone repository:

`git clone https://github.com/EthanM317/Sentiment-Analysis-Star-Rating-Prediction.git`

- Download [IMDB movie reviews grouped by ratings](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place .csv files into project root


- Create a local python environment and install the dependencies

`python3 -m venv .venv`

`source .venv/ac` Or `.venv\Scripts\activate.bat`

`pip install -r requirements`



- Combine .csv data files into one file:

`python initial-eda.py`


### Baseline
Train and evaluate model:
`python baseline.py`

Once the model is trained it will be saved in baseline-model.cache. You can use `what-rating-baseline.py` to make a prediction on any piece of text. 

`python what-rating-baseline.py "This movie is amazing!"`


### Our Model


- Download [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place the csv in the repositories root directory.

Train and evaluate model:
`python model.py`

Once the model is trained it will be saved in baseline-model.cache. You can use `what-rating.py` to make a prediction on any piece of text.

`python what-rating.py "This movie is garbage!"`