# Import caching tools
from functools import lru_cache

# Import data processing and visualisation tools
import pandas as pd
# import matplotlib.pyplot as plt

# Import machine learning libraries
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection  import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics

# Load the initial database
def load_dataset0():
    df = pd.read_csv("diabetes.csv")
    return df

@lru_cache(maxsize=None)
@time
def load_dataset1():
    df = pd.read_csv("diabetes.csv")
    return df

print(load_dataset0().head())
print(load_dataset1().head())


# Intialize machine learning parameters
K = 17

# Load transformed data set
# @cache
# def transfrom_cache_db(dataset):
    # def affineTransform(x):
        #return (2*x + 1)
    
    # transformed_db = dataset
    # transformed_db.appy(affineTransform)
    # transformed_db.head(5)
    # return transformed_db

# original_db = load_dataset()
# test_db = transfrom_cache_db(original_db)