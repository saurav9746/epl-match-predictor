# 1) Train baseline model
python train_model.py


# 2) (optional) Tune with Optuna
python tune_optuna.py


# 3) Start Streamlit dashboard
streamlit run streamlit_app.py


# 4) Start API
uvicorn fastapi_app:app --reload --port 8000


# 5) Use curl to test API
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"HomeTeam":"Arsenal","AwayTeam":"Chelsea"}'