import os
import pandas as pd
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns

cwd = os.getcwd()


def load_positive_and_negative_data(path:Path):
    def get_id_and_rating(text_file:Path):
        id, rating = map(int, text_file.stem.split("_"))
        content = ""
        with open(text_file, "r", encoding="utf-8") as file:
            content = file.read()
        return (str(cwd), content, rating)


    def load_reviews_from_dir(dir:Path):
        return pd.DataFrame([get_id_and_rating(x) for x in dir.rglob(f"*.*")], columns=['file', 'content', 'rating'])


    df_pos = load_reviews_from_dir(path / Path("pos"))
    df_pos['sentiment'] = 1


    df_neg = load_reviews_from_dir(path / Path(".neg"))
    df_neg['sentiment'] = 0

    return pd.merge(df_pos, df_neg, how='outer').sample(frac=1).reset_index(drop=True)


df_train = load_positive_and_negative_data("./aclImdb/train")

print(df_train)

# sum and show all nan values
print (df_train.isna().sum().tolist())
print(f"{df_train.columns[df_train.isna().any()].tolist()}")

# # show duplicated valuess
print (df_train.duplicated().tolist())
