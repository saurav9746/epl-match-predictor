import optuna
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from data_prep import load_and_clean


def objective(trial):
    df = load_and_clean('epl_final.csv')

    # Target
    y = df['FullTimeResult']
    X = df.drop(columns=['FullTimeResult'])

    # Hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 40)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    # Columns
    categorical_cols = ['HomeTeam', 'AwayTeam']
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocessing
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    # Model
    model = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    return scores.mean()


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    print('Best params:', study.best_params)
    print('Best value:', study.best_value)

    # Optional: train final model with best params and save it
    df = load_and_clean('epl_final.csv')
    y = df['FullTimeResult']
    X = df.drop(columns=['FullTimeResult'])

    categorical_cols = ['HomeTeam', 'AwayTeam']
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    best_model = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(
            **study.best_params,
            random_state=42,
            n_jobs=-1
        ))
    ])

    best_model.fit(X, y)
    joblib.dump(best_model, 'epl_model_optimized.pkl')

    print("Saved optimized model as epl_model_optimized.pkl")
