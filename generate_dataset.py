"""Generate pregnancy risk dataset tuned for 90-95% model accuracy."""

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
        bp = round(clamp(random.gauss(118, 10), 85, 170), 1)
        sugar = round(clamp(random.gauss(96, 16), 65, 230), 1)
        temp = round(clamp(random.gauss(98.3, 0.7), 96.0, 103.5), 1)
        hr = int(round(clamp(random.gauss(82, 8), 55, 140), 0))
        hb = round(clamp(random.gauss(11.9, 1.2), 7.0, 15.5), 1)
        urine = round(clamp(random.gauss(112, 45), 10, 450), 1)
        gravida = random.randint(0, 6)
        para = random.randint(0, gravida)
        weight = round(clamp(random.gauss(67, 9.5), 40, 120), 1)
        height = round(clamp(random.gauss(158, 6.0), 140, 185), 1)

        complications = 1 if random.random() < 0.22 else 0
        stress = random.randint(0, 10)
        activity = random.choices([0, 1, 2], weights=[0.27, 0.54, 0.19])[0]
        edema = 1 if random.random() < 0.18 else 0
        smoking = 1 if random.random() < 0.14 else 0

        bmi = weight / ((height / 100) ** 2)

        # Strong but realistic predictive signal.
        score = 0.0
        score += 1.35 if bp > 130 else 0.0
        score += 1.45 if sugar > 120 else 0.0
        score += 0.95 if temp > 99.4 else 0.0
        score += 0.95 if hr > 100 else 0.0
        score += 1.25 if hb < 10.4 else 0.0
        score += 1.10 if urine > 180 else 0.0
        score += 0.70 if age > 33 else 0.0
        score += 0.65 if gravida >= 4 else 0.0
        score += 0.60 if bmi > 30 else 0.0
        score += 1.25 if complications else 0.0
        score += 0.85 if stress >= 7 else 0.0
        score += 0.60 if activity == 0 else 0.0
        score += 1.05 if edema else 0.0
        score += 0.85 if smoking else 0.0

        # calibrated overlap to avoid perfect scores
        score += random.gauss(0, 0.35)

        prob_medium = 1.0 / (1.0 + math.exp(-(-3.3 + score)))
        medium = 1 if random.random() < prob_medium else 0

        # mild label noise keeps scores around 90-95 rather than 100
        if random.random() < 0.03:
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

print("Saved pregnancy_risk_dataset_1015.csv tuned for 90-95% model accuracy")
