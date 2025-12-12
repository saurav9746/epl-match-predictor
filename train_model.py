# train_model.py
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from data_prep import load_and_clean

if __name__ == '__main__':
    # Load cleaned data
    df = load_and_clean('epl_final.csv')

    # Target column
    TARGET = 'FullTimeResult'

    # Split X and y
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Categorical and numeric features
    categorical_cols = ['HomeTeam', 'AwayTeam']
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Preprocessing
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    # Model pipeline
    model = Pipeline([
        ('pre', pre),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, 'epl_model.pkl')
    print('Saved epl_model.pkl')
