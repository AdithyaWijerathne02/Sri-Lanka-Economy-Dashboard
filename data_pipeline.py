import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("raw_data.csv")
    return df

def clean_data(df):
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.dropna()
    return df

def save_clean(df):
    df.to_csv("data_clean/final_economic_data.csv", index=False)

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save_clean(df)
