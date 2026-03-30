import pandas as pd

def encode_categorical(df):
    df_encoded = df.copy()
    if "class" in df_encoded.columns:
        df_encoded["class"] = df_encoded["class"].map({
            "Normal": 0,
            "Abnormal": 1
        })

    return df_encoded