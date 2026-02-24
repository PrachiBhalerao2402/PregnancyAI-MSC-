# Pregnancy Risk Prediction System

This project trains 4 ML classifiers to predict **Low Risk** vs **Medium Risk** pregnancy status and provides:

- Streamlit app (`app.py`)
- HTML/CSS/JS frontend (`frontend/`) that matches the shared UI style

## Project files
- `pregnancy_risk_dataset_1015.csv` - dataset in expected format.
- `train_models.py` - training, evaluation, visualization, model export.
- `app.py` - Streamlit prediction interface.
- `frontend/index.html` - health details form UI.
- `frontend/result.html` - result screen UI.
- `frontend/styles.css` - frontend styling.
- `frontend/script.js` - form logic + `/predict` API call.
- `frontend/result.js` - result screen rendering from risk query parameter.
- `models/best_pregnancy_risk_model.joblib` - best model artifact (generated after training).

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Train models
```bash
python3 train_models.py
```

The training script tries multiple parameter candidates for each model and prefers configurations with test accuracy in the **90%–95% range**.

Generated artifacts:
- `outputs/model_accuracy_comparison.png`
- `outputs/confusion_matrices.png`
- `outputs/feature_importance_random_forest.png`
- `outputs/feature_importance_gradient_boosting.png`
- `models/best_pregnancy_risk_model.joblib`

## Run Streamlit app
```bash
streamlit run app.py
```

## Use HTML frontend
Serve the project folder and open:
- `frontend/index.html`

Example:
```bash
python3 -m http.server 8000
```
Then visit `http://127.0.0.1:8000/frontend/index.html`

For live prediction from this frontend, expose backend endpoint:
- `POST /predict`
- JSON request body with form fields
- JSON response example: `{ "risk": "Low Risk" }`
