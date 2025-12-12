# feature_engineering.py
import pandas as pd

# Simple ELO implementation and rolling form feature generator

def initial_elo(df, k=20, base=1500):
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    elo = {t: base for t in teams}
    return elo


def add_elo_and_form_features(df):
    df = df.sort_values('MatchDate') if 'MatchDate' in df.columns else df.copy()
    elo = initial_elo(df)

    ELO_HOME = []
    ELO_AWAY = []
    HOME_FORM = []
    AWAY_FORM = []

    # keep small history for form: last 5 results encoded as points (win=3, draw=1, loss=0)
    history = {t: [] for t in elo.keys()}

    def avg_points(team):
        hs = history.get(team, [])
        if len(hs) == 0:
            return 0.0
        return sum(hs[-5:]) / min(5, len(hs))

    for _, row in df.iterrows():
        h = row['HomeTeam']
        a = row['AwayTeam']

        # record current elo
        ELO_HOME.append(elo.get(h, 1500))
        ELO_AWAY.append(elo.get(a, 1500))

        # form: average points last 5
        HOME_FORM.append(avg_points(h))
        AWAY_FORM.append(avg_points(a))

        # update ELO & form history only if goals exist
        if 'FullTimeHomeGoals' in row.index and 'FullTimeAwayGoals' in row.index:
            hg = row['FullTimeHomeGoals']
            ag = row['FullTimeAwayGoals']

            # compute points (simple system)
            if hg > ag:
                rh, ra = 3, 0
            elif hg == ag:
                rh, ra = 1, 1
            else:
                rh, ra = 0, 3

            # update match history
            history[h].append(rh)
            history[a].append(ra)

            # ELO update
            R_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
            R_a = 1 / (1 + 10 ** ((elo[h] - elo[a]) / 400))
            S_h = 1 if hg > ag else (0.5 if hg == ag else 0)
            S_a = 1 - S_h
            K = 20
            elo[h] = elo[h] + K * (S_h - R_h)
            elo[a] = elo[a] + K * (S_a - R_a)

    df['ELO_Home'] = ELO_HOME
    df['ELO_Away'] = ELO_AWAY
    df['HomeForm'] = HOME_FORM
    df['AwayForm'] = AWAY_FORM

    return df
