# EPL Match Predictor

Predict the outcome of English Premier League (EPL) matches using team statistics and machine learning.

---

## **Overview**

This project predicts the probability of a **Home Win, Draw, or Away Win** for EPL matches based on historical team performance.  
It uses a **Random Forest Classifier** trained on match-level features like:

- Team average goals scored/conceded
- Recent form (last N matches)
- Home/Away performance
- Engineered difference features

The model also comes with **a Streamlit UI** for interactive prediction by entering team names.

---

## **Features**

- Team-based statistics and rolling form
- Engineered difference features for better prediction
- Scaled inputs using `StandardScaler`
- Random Forest classifier for multi-class prediction
- Interactive **Streamlit app** to predict match results

---

## **Files in the repository**

- `train_team_model.py` – Prepares features, trains the model, saves scaler and latest team stats.
- `streamlit_app.py` – Streamlit UI for predicting EPL match outcomes.
- `app.py` – Optional FastAPI endpoint to get teams or predictions.
- `team_match_model.pkl` – Trained Random Forest model.
- `scaler_team.pkl` – Scaler used for input features.
- `team_stats_latest.pkl` – Latest snapshot of all team stats.
- `requirements.txt` – Python dependencies.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/saurav9746/epl-match-predictor.git
cd epl-match-predictor


Create a virtual environment:

python -m venv venv


Activate the environment:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

Running the Streamlit App
streamlit run streamlit_app.py
