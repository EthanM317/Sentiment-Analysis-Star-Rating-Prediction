# Sentiment-Analysis-Star-Rating-Prediction
Final Project for CSCI 3052U


## How to Use

- Clone repository:
`git clone https://github.com/EthanM317/Sentiment-Analysis-Star-Rating-Prediction.git`
- Download IMDB movie reviews grouped by ratings and place .csv files into project root


- Create a local python environment and install the dependencies
`python3 -m venv .venv`
`source .venv/ac` Or `.venv\Scripts\activate.bat`
`pip install -r requirements`

- Combine .csv data files into one file:
`python baseline-processing.py`
Train and test model:
`python baseline.py`

Once the model is trained it will be saved in model.cache. You can use `what-rating.py` to make a prediction on any piece of text. 
`python baseline.py "This movie is amazing!"`
