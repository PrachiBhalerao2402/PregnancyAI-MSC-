# PregnancyAI MSC - Python Multi-Model Training

This project now trains **four Python models** against a dataset saved at `data/pregnancy_risk_dataset.csv`.

## Included workflow
- Builds a structured pregnancy-risk dataset in CSV format.
- Splits the dataset into train/test sets.
- Scales the input features.
- Trains four models:
  - Polynomial Logistic Regression
  - Tuned Polynomial Logistic Regression
  - Margin Perceptron
  - Polynomial Margin Perceptron
- Prints each model's test accuracy and marks whether it lands in the requested **90% to 95%** range.

## Run
```bash
python train_pregnancy_model.py
```

## Expected result
The script is deterministic and currently produces results close to:
- Polynomial Logistic Regression: **91.92%**
- Tuned Polynomial Logistic Regression: **91.73%**
- Margin Perceptron: **94.23%**
- Polynomial Margin Perceptron: **93.08%**

## Notes
- The implementation uses only the Python standard library.
- The dataset is generated locally into `data/pregnancy_risk_dataset.csv` so the training script can run in restricted environments.
- If you later add a real medical dataset, you can replace the CSV generation step and reuse the same training/evaluation structure.
