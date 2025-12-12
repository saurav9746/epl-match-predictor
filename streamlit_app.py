# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout='wide', page_title='EPL Match Win Probabilities')

# Load model
model = joblib.load('epl_model.pkl')

st.title('EPL Match Winning Probability (Home / Draw / Away)')

# Load teams from encoder
teams = sorted(list(model.named_steps['pre'].named_transformers_['cat'].categories_[0]))
teams2 = sorted(list(model.named_steps['pre'].named_transformers_['cat'].categories_[1]))

# Input form
with st.form('match'):
    home = st.selectbox('Home Team', teams)
    away = st.selectbox('Away Team', teams2)

    st.write('Match stats — optional (defaults kept neutral)')
    half_h = st.number_input('HalfTimeHomeGoals', min_value=0, value=0)
    half_a = st.number_input('HalfTimeAwayGoals', min_value=0, value=0)
    shots_h = st.number_input('HomeShots', min_value=0, value=8)
    shots_a = st.number_input('AwayShots', min_value=0, value=7)
    sot_h = st.number_input('HomeShotsOnTarget', min_value=0, value=4)
    sot_a = st.number_input('AwayShotsOnTarget', min_value=0, value=3)

    submitted = st.form_submit_button('Predict')

if submitted:
    # Build row for prediction
    new_match = pd.DataFrame([{
        'HomeTeam': home,
        'AwayTeam': away,
        'HalfTimeHomeGoals': half_h,
        'HalfTimeAwayGoals': half_a,
        'HomeShots': shots_h,
        'AwayShots': shots_a,
        'HomeShotsOnTarget': sot_h,
        'AwayShotsOnTarget': sot_a,
        'ELO_Home': 1500,   # Placeholder
        'ELO_Away': 1500,   # Placeholder
        'HomeForm': 1.0,
        'AwayForm': 1.0
    }])

    # Predict probabilities
    probs = model.predict_proba(new_match)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric('Home Win', f'{probs[0]:.3f}')
    col2.metric('Draw', f'{probs[1]:.3f}')
    col3.metric('Away Win', f'{probs[2]:.3f}')
