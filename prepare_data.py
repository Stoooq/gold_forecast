import pandas as pd
import numpy as np

df = pd.read_csv("/Users/miloszglowacki/Desktop/code/python/gold_forecast/data/gold.csv", parse_dates=["Date"])
df = df.iloc[-2000:]

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 4])
    return np.array(X), np.array(y)

print(df.head)