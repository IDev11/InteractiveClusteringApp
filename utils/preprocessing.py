# utils/preprocessing.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def handle_missing_data(df, strategy_dict):
    df = df.copy()
    for col, strategy in strategy_dict.items():
        if strategy != 'none':
            imputer = SimpleImputer(strategy=strategy)
            df[[col]] = imputer.fit_transform(df[[col]])
    return df


def normalize_data(df, normalization_dict):
    df = df.copy()
    for col, method in normalization_dict.items():
        if method == 'standard':
            scaler = StandardScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[[col]] = scaler.fit_transform(df[[col]])
    return df
