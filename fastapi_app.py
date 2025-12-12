# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Optional: allow all origins (good for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load('epl_model.pkl')


class MatchIn(BaseModel):
    HomeTeam: str
    AwayTeam: str
    HalfTimeHomeGoals: int = 0
    HalfTimeAwayGoals: int = 0
    HomeShots: int = 0
    AwayShots: int = 0
    HomeShotsOnTarget: int = 0
    AwayShotsOnTarget: int = 0
    ELO_Home: float = 1500.0
    ELO_Away: float = 1500.0
    HomeForm: float = 1.0
    AwayForm: float = 1.0


@app.post('/predict')
def predict(m: MatchIn):
    row = pd.DataFrame([m.dict()])
    probs = model.predict_proba(row)[0]
    return {
        'Home': float(probs[0]),
        'Draw': float(probs[1]),
        'Away': float(probs[2])
    }


# Run using:
# uvicorn fastapi_app:app --reload --port 8000
