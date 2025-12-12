# train_team_model.py
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ---------- PARAMETERS ----------
CSV_PATH = "epl_final.csv"
MODEL_OUT = "team_match_model.pkl"
SCALER_OUT = "scaler_team.pkl"
TEAM_STATS_OUT = "team_stats_latest.pkl"
RECENT_N = 5   # last-N matches used for 'form' features

# ---------- HELPER ----------
def points_from_result(res, side):
    # res is "H","D","A"
    # side: "home" or "away"
    if res == "D":
        return 1
    if res == "H":
        return 3 if side == "home" else 0
    if res == "A":
        return 3 if side == "away" else 0
    return 0

# ---------- LOAD & SORT ----------
df = pd.read_csv(CSV_PATH)
# ensure MatchDate is datetime
if "MatchDate" in df.columns:
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
else:
    # if no date, create simple index order
    df["MatchDate"] = pd.RangeIndex(len(df))

df = df.sort_values("MatchDate").reset_index(drop=True)
print(f"Loaded {len(df)} matches")

# ---------- SETUP PER-TEAM HISTORIES ----------
teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
# For each team we keep rolling info BEFORE each match (no leakage)
history = {}
for t in teams:
    history[t] = {
        "matches": 0,
        "goals_for": 0,
        "goals_against": 0,
        "home_matches": 0,
        "away_matches": 0,
        "home_goals_for": 0,
        "home_goals_against": 0,
        "away_goals_for": 0,
        "away_goals_against": 0,
        "points": 0,
        "recent_points": deque(maxlen=RECENT_N),  # last N match points
        "recent_home_points": deque(maxlen=RECENT_N),
        "recent_away_points": deque(maxlen=RECENT_N),
        # optional: head-to-head counts could be added
    }

# ---------- BUILD FEATURES PER MATCH (time-safe) ----------
rows = []
for idx, row in df.iterrows():
    h = row["HomeTeam"]
    a = row["AwayTeam"]
    # use FullTimeHomeGoals and FullTimeAwayGoals presence
    if ("FullTimeHomeGoals" not in row.index) or ("FullTimeAwayGoals" not in row.index):
        continue

    # fetch each team's current stats (before this match)
    def team_snapshot(team):
        s = history[team]
        matches = s["matches"]
        # avoid division by zero
        avg_gf = s["goals_for"] / matches if matches > 0 else 0.0
        avg_ga = s["goals_against"] / matches if matches > 0 else 0.0
        home_avg_gf = s["home_goals_for"] / s["home_matches"] if s["home_matches"] > 0 else 0.0
        away_avg_gf = s["away_goals_for"] / s["away_matches"] if s["away_matches"] > 0 else 0.0
        win_rate = (s["points"] / (3 * matches)) if matches > 0 else 0.0  # normalized to [0,1]
        recent_avg = (sum(s["recent_points"]) / (3 * len(s["recent_points"]))) if len(s["recent_points"])>0 else 0.0
        recent_home = (sum(s["recent_home_points"]) / (3 * len(s["recent_home_points"]))) if len(s["recent_home_points"])>0 else 0.0
        recent_away = (sum(s["recent_away_points"]) / (3 * len(s["recent_away_points"]))) if len(s["recent_away_points"])>0 else 0.0
        return {
            "matches": matches,
            "avg_gf": avg_gf,
            "avg_ga": avg_ga,
            "home_avg_gf": home_avg_gf,
            "away_avg_gf": away_avg_gf,
            "win_rate": win_rate,
            "recent_form": recent_avg,
            "recent_home_form": recent_home,
            "recent_away_form": recent_away
        }

    snap_h = team_snapshot(h)
    snap_a = team_snapshot(a)

    # features to represent state before the match
    feat = {
        "HomeTeam": h,
        "AwayTeam": a,
        # Home team features
        "H_matches": snap_h["matches"],
        "H_avg_gf": snap_h["avg_gf"],
        "H_avg_ga": snap_h["avg_ga"],
        "H_home_avg_gf": snap_h["home_avg_gf"],
        "H_recent_form": snap_h["recent_form"],
        # Away team features
        "A_matches": snap_a["matches"],
        "A_avg_gf": snap_a["avg_gf"],
        "A_avg_ga": snap_a["avg_ga"],
        "A_away_avg_gf": snap_a["away_avg_gf"],
        "A_recent_form": snap_a["recent_form"],
    }

    # engineered features (differences)
    feat["diff_avg_gf"] = feat["H_avg_gf"] - feat["A_avg_gf"]
    feat["diff_avg_ga"] = feat["H_avg_ga"] - feat["A_avg_ga"]
    feat["diff_recent_form"] = feat["H_recent_form"] - feat["A_recent_form"]

    # home advantage: we include whether home team historically scores more at home
    feat["home_adv_home_avg_gf"] = feat["H_home_avg_gf"]

    # target: encode from FullTimeResult ("H","D","A") -> 2,1,0
    res = row["FullTimeResult"]
    if pd.isna(res):
        continue
    target = {"H": 2, "D": 1, "A": 0}.get(res, None)
    if target is None:
        continue
    feat["target"] = target
    rows.append(feat)

    # ---- AFTER collecting features for this match: update team histories WITH this match outcome ----
    hg = int(row["FullTimeHomeGoals"])
    ag = int(row["FullTimeAwayGoals"])
    # update home
    s = history[h]
    s["matches"] += 1
    s["goals_for"] += hg
    s["goals_against"] += ag
    s["home_matches"] += 1
    s["home_goals_for"] += hg
    s["home_goals_against"] += ag
    pts_h = points_from_result(res, "home")
    s["points"] += pts_h
    s["recent_points"].append(pts_h)
    s["recent_home_points"].append(pts_h)

    # update away
    s2 = history[a]
    s2["matches"] += 1
    s2["goals_for"] += ag
    s2["goals_against"] += hg
    s2["away_matches"] += 1
    s2["away_goals_for"] += ag
    s2["away_goals_against"] += hg
    pts_a = points_from_result(res, "away")
    s2["points"] += pts_a
    s2["recent_points"].append(pts_a)
    s2["recent_away_points"].append(pts_a)

# ---------- CREATE DATAFRAME ----------
fe_df = pd.DataFrame(rows)
print("Constructed feature dataframe with", fe_df.shape[0], "rows")

# Remove rows where both teams had zero prior matches? (optional)
# fe_df = fe_df[(fe_df["H_matches"]>0) & (fe_df["A_matches"]>0)]

# Features to train on
feature_cols = [
    "H_matches", "H_avg_gf", "H_avg_ga", "H_home_avg_gf", "H_recent_form",
    "A_matches", "A_avg_gf", "A_avg_ga", "A_away_avg_gf", "A_recent_form",
    "diff_avg_gf", "diff_avg_ga", "diff_recent_form", "home_adv_home_avg_gf"
]

# fill NaN
fe_df[feature_cols] = fe_df[feature_cols].fillna(0.0)

X = fe_df[feature_cols].values
y = fe_df["target"].values.astype(int)

# ---------- SCALE, TRAIN, EVALUATE ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- SAVE MODEL + SCALER ---------- 
joblib.dump(clf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print(f"\nSaved model to {MODEL_OUT} and scaler to {SCALER_OUT}")

# ---------- SAVE LATEST TEAM STATS for prediction time ----------
# Build a final per-team stat snapshot (latest history)
team_stats = {}
for t, s in history.items():
    matches = s["matches"]
    team_stats[t] = {
        "matches": matches,
        "avg_gf": s["goals_for"]/matches if matches>0 else 0.0,
        "avg_ga": s["goals_against"]/matches if matches>0 else 0.0,
        "home_avg_gf": s["home_goals_for"]/s["home_matches"] if s["home_matches"]>0 else 0.0,
        "away_avg_gf": s["away_goals_for"]/s["away_matches"] if s["away_matches"]>0 else 0.0,
        "win_rate": (s["points"]/(3*matches)) if matches>0 else 0.0,
        "recent_form": (sum(s["recent_points"])/(3*len(s["recent_points"]))) if len(s["recent_points"])>0 else 0.0,
        "recent_home_form": (sum(s["recent_home_points"])/(3*len(s["recent_home_points"]))) if len(s["recent_home_points"])>0 else 0.0,
        "recent_away_form": (sum(s["recent_away_points"])/(3*len(s["recent_away_points"]))) if len(s["recent_away_points"])>0 else 0.0
    }

joblib.dump(team_stats, TEAM_STATS_OUT)
print(f"Saved latest team stats to {TEAM_STATS_OUT}")

print("\nTraining complete. You can now run the app_team_predictor.py Streamlit app to predict by team names.")
