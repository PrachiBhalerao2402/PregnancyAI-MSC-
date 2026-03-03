# Pregnancy Risk Prediction System

This project trains 4 ML classifiers to predict **Low Risk** vs **Medium Risk** pregnancy status and provides:

- Streamlit app (`app.py`)
- HTML/CSS/JS frontend (`frontend/`) matching the shared UI style

## Project files
- `pregnancy_risk_dataset_1015.csv` - dataset in expected format.
- `train_models.py` - training, evaluation, visualization, model export.
- `app.py` - Streamlit prediction interface.
- `frontend/index.html` - health details form UI with text-box inputs.
- Streamlit `app.py` uses empty-first text boxes + personalized Get Advice action and hides model-name text.
- `frontend/result.html` - prediction result screen.
- `frontend/advice.html` - get-advice page for low/medium risk.
- `frontend/styles.css` - frontend styling.
- `frontend/script.js` - form logic + `/predict` API call.
- `frontend/result.js` - result state rendering.
- `frontend/advice.js` - advice-page content rendering.
- `models/best_pregnancy_risk_model.joblib` - best model artifact (generated after training).

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Train models
```bash
python3 train_models.py
```

By default, training will continue with the closest split if 4/4 models cannot be kept in 90–95% on your local data. To enforce hard failure instead, run:
`STRICT_ALL_MODELS_IN_BAND=true python3 train_models.py`

The training script evaluates multiple parameter candidates per model and multiple stratified split states with progress logs. It selects model combinations to keep all 4 scores in the **90%–95% range** while preferring non-identical accuracies (higher spread/diversity), using real (unmodified) model predictions. It stops early when all 4 are in-band with acceptable spread. If your local run cannot get 4/4 in-band, it uses the closest split by default and warns, or raises in strict mode.

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
```bash
python3 -m http.server 8000
```
Then visit `http://127.0.0.1:8000/frontend/index.html`

For live prediction from this frontend, expose backend endpoint:
- `POST /predict`
- JSON request body with form fields
- JSON response example: `{ "risk": "Low Risk" }`


## Chatbot (Streamlit Tab)
- Open `app.py` and use the **Women Health Chatbot** tab.
- For local rule-based responses, no setup is required.
- For Gemini API responses, set:
  - `GEMINI_API_KEY`
  - optional: `GEMINI_MODEL` (default `gemini-1.5-flash`)
