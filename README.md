# Pregnancy Risk Prediction System

This project trains ML classifiers to predict **Low Risk** vs **Medium Risk** pregnancy status and provides:

- Streamlit app (`app.py`)
- HTML/CSS/JS frontend (`frontend/`)

## Project files
- `pregnancy_risk_dataset_1015.csv` - dataset in expected format.
- `train_models.py` - training, evaluation, visualization, model export.
- `app.py` - Streamlit prediction interface.
- `frontend/index.html` - standalone frontend matching the requested health form style.
- `frontend/styles.css` - frontend styling.
- `frontend/script.js` - frontend form logic + `/predict` API call.
- `models/best_pregnancy_risk_model.joblib` - best model artifact (generated after training).

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Train models
```bash
python3 train_models.py
```

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
Open `frontend/index.html` in a browser.

If you want live predictions from this frontend, expose a backend endpoint:
- `POST /predict`
- JSON body with the form fields
- JSON response example: `{ "risk": "Low Risk" }`
