# app_team_predictor.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

MODEL = "team_match_model.pkl"
SCALER = "scaler_team.pkl"
TEAM_STATS = "team_stats_latest.pkl"

st.set_page_config(page_title="EPL Team-Only Predictor", layout="centered")
st.title("EPL Match Outcome Predictor — Enter Teams Only")
st.markdown("Model predicts probability of Home Win / Draw / Away Win using team-strength features computed from historical matches.")

# load
clf = joblib.load(MODEL)
scaler = joblib.load(SCALER)
team_stats = joblib.load(TEAM_STATS)

teams = sorted(list(team_stats.keys()))

home = st.selectbox("Home Team", teams, index=teams.index(teams[0]) if teams else 0)
away = st.selectbox("Away Team", teams, index=teams.index(teams[1]) if len(teams)>1 else 0)

if home == away:
    st.warning("Choose two different teams.")
else:
    if st.button("Predict"):
        # prepare feature vector using team_stats snapshot
        h = team_stats[home]
        a = team_stats[away]

        feat = [
            h["matches"],
            h["avg_gf"],
            h["avg_ga"],
            h["home_avg_gf"],
            h["recent_form"],
            a["matches"],
            a["avg_gf"],
            a["avg_ga"],
            a["away_avg_gf"],
            a["recent_form"],
            h["avg_gf"] - a["avg_gf"],
            h["avg_ga"] - a["avg_ga"],
            h["recent_form"] - a["recent_form"],
            h["home_avg_gf"]
        ]
        feat_arr = np.array(feat).reshape(1, -1)
        feat_scaled = scaler.transform(feat_arr)
        probs = clf.predict_proba(feat_scaled)[0]

        st.subheader("Prediction (probabilities)")
        st.write(f"🏠 Home Win : **{probs[2]:.3f}**")
        st.write(f"🤝 Draw     : **{probs[1]:.3f}**")
        st.write(f"🏳️ Away Win : **{probs[0]:.3f}**")

        idx = np.argmax(probs)
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        st.success(f"Predicted outcome: **{label_map[idx]}**")

        # optional: show both teams' statistics
        st.markdown("---")
        st.subheader("Team snapshots (latest)")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home}**")
            st.write(pd.Series(h))
        with col2:
            st.write(f"**{away}**")
            st.write(pd.Series(a))
