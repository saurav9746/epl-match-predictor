import streamlit as st
import pandas as pd
import joblib

# ---------------- Load Model, Scaler & Team Stats ----------------
MODEL_PATH = "team_match_model.pkl"
SCALER_PATH = "scaler_team.pkl"
TEAM_STATS_PATH = "team_stats_latest.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
team_stats = joblib.load(TEAM_STATS_PATH)

teams = sorted(list(team_stats.keys()))

# ---------------- Page Setup ----------------
st.set_page_config(page_title="EPL Match Predictor", layout="wide")
st.title("EPL Match Winning Probability Predictor (Team-Based)")
st.markdown("Select **Home Team** and **Away Team** to generate probabilities.")

# ---------------- Input Form ----------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox("Home Team", teams)
    with col2:
        away = st.selectbox("Away Team", teams)

    submitted = st.form_submit_button("Predict Result")

# ---------------- Build Features ----------------
def build_features(home, away):
    H = team_stats[home]
    A = team_stats[away]

    feat = {
        "H_matches": H["matches"],
        "H_avg_gf": H["avg_gf"],
        "H_avg_ga": H["avg_ga"],
        "H_home_avg_gf": H["home_avg_gf"],
        "H_recent_form": H["recent_form"],

        "A_matches": A["matches"],
        "A_avg_gf": A["avg_gf"],
        "A_avg_ga": A["avg_ga"],
        "A_away_avg_gf": A["away_avg_gf"],
        "A_recent_form": A["recent_form"],

        "diff_avg_gf": H["avg_gf"] - A["avg_gf"],
        "diff_avg_ga": H["avg_ga"] - A["avg_ga"],
        "diff_recent_form": H["recent_form"] - A["recent_form"],

        "home_adv_home_avg_gf": H["home_avg_gf"]
    }

    df = pd.DataFrame([feat])
    scaled = scaler.transform(df)
    return scaled

# ---------------- Predict ----------------
if submitted:
    if home == away:
        st.error("Home team and Away team cannot be the same.")
    else:
        X_scaled = build_features(home, away)
        probs = model.predict_proba(X_scaled)[0]

        st.subheader("Match Win Probabilities:")
        colA, colB, colC = st.columns(3)

        colA.metric("Home Win Probability", f"{probs[2]:.3f}")
        colB.metric("Draw Probability", f"{probs[1]:.3f}")
        colC.metric("Away Win Probability", f"{probs[0]:.3f}")
