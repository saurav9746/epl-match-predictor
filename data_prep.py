# data_prep.py
import pandas as pd
import numpy as np
from feature_engineering import add_elo_and_form_features

TARGET = 'FullTimeResult'


def load_and_clean(path='epl_final.csv'):
    df = pd.read_csv(path)

    # basic clean: drop rows with missing target
    df = df.dropna(subset=[TARGET])

    # convert MatchDate if exists
    if 'MatchDate' in df.columns:
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')

    # remove leakage columns
    drop_cols = [
        'FullTimeHomeGoals', 'FullTimeAwayGoals',
        'FullTimeResult', 'HalfTimeResult'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # construct features
    df = add_elo_and_form_features(df)

    return df
