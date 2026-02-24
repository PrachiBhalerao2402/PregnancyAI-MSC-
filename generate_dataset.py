"""Generate pregnancy risk dataset tuned for realistic 90-95% model performance."""

import csv
import math
import random

RANDOM_STATE = 42
N_SAMPLES = 1015
random.seed(RANDOM_STATE)

HEADERS = [
    "Age",
    "BloodPressure",
    "BloodSugar",
    "BodyTemperature",
    "HeartRate",
    "Hemoglobin",
    "UrineProtein",
    "Gravida",
    "Para",
    "Weight",
    "Height",
    "PreviousPregnancyComplications",
    "StressLevel",
    "PhysicalActivityLevel",
    "Edema",
    "SmokingAlcoholHistory",
    "RiskLevel_Low",
    "RiskLevel_Medium",
]


def clamp(value, low, high):
    return max(low, min(high, value))


with open("pregnancy_risk_dataset_1015.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(HEADERS)

    for _ in range(N_SAMPLES):
        age = random.randint(18, 41)
        bp = round(clamp(random.gauss(118, 12), 85, 170), 1)
        sugar = round(clamp(random.gauss(96, 20), 65, 230), 1)
        temp = round(clamp(random.gauss(98.4, 0.9), 96.0, 103.5), 1)
        hr = int(round(clamp(random.gauss(82, 10), 55, 140), 0))
        hb = round(clamp(random.gauss(11.8, 1.4), 7.0, 15.5), 1)
        urine = round(clamp(random.gauss(115, 55), 10, 450), 1)
        gravida = random.randint(0, 6)
        para = random.randint(0, gravida)
        weight = round(clamp(random.gauss(67, 11), 40, 120), 1)
        height = round(clamp(random.gauss(158, 7), 140, 185), 1)

        complications = 1 if random.random() < 0.25 else 0
        stress = random.randint(0, 10)
        activity = random.choices([0, 1, 2], weights=[0.28, 0.52, 0.20])[0]  # low, moderate, high
        edema = 1 if random.random() < 0.20 else 0
        smoking = 1 if random.random() < 0.16 else 0

        bmi = weight / ((height / 100) ** 2)

        # Weighted score with clear risk signal but realistic overlap.
        score = 0.0
        score += 1.05 if bp > 132 else 0.0
        score += 1.15 if sugar > 122 else 0.0
        score += 0.75 if temp > 99.5 else 0.0
        score += 0.80 if hr > 100 else 0.0
        score += 1.05 if hb < 10.4 else 0.0
        score += 0.95 if urine > 185 else 0.0
        score += 0.65 if age > 33 else 0.0
        score += 0.60 if gravida >= 4 else 0.0
        score += 0.60 if bmi > 30 else 0.0
        score += 1.10 if complications else 0.0
        score += 0.70 if stress >= 7 else 0.0
        score += 0.55 if activity == 0 else 0.0
        score += 0.95 if edema else 0.0
        score += 0.75 if smoking else 0.0

        # logistic transform + calibrated uncertainty
        prob_medium = 1.0 / (1.0 + math.exp(-(-2.65 + score)))
        medium = 1 if random.random() < prob_medium else 0

        # controlled label noise to keep models < 100% and near 90-95%
        if random.random() < 0.055:
            medium = 1 - medium

        low = 1 - medium

        writer.writerow(
            [
                age,
                bp,
                sugar,
                temp,
                hr,
                hb,
                urine,
                gravida,
                para,
                weight,
                height,
                complications,
                stress,
                activity,
                edema,
                smoking,
                low,
                medium,
            ]
        )

print("Saved pregnancy_risk_dataset_1015.csv with calibrated signal/noise profile")
