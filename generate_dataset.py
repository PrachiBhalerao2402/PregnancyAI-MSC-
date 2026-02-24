import csv
import math
import random

RANDOM_STATE = 42
N_SAMPLES = 1015
random.seed(RANDOM_STATE)

headers = [
    "Age","BloodPressure","BloodSugar","BodyTemperature","HeartRate","Hemoglobin","UrineProtein",
    "Gravida","Para","Weight","Height","PreviousPregnancyComplications","StressLevel",
    "PhysicalActivityLevel","Edema","SmokingAlcoholHistory","RiskLevel_Low","RiskLevel_Medium"
]

def clamp(v, low, high):
    return max(low, min(high, v))

with open("pregnancy_risk_dataset_1015.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(headers)
    for _ in range(N_SAMPLES):
        age = random.randint(18, 41)
        bp = round(clamp(random.gauss(118, 14), 85, 170), 1)
        sugar = round(clamp(random.gauss(98, 24), 65, 230), 1)
        temp = round(clamp(random.gauss(98.4, 1.2), 96.0, 103.5), 1)
        hr = int(round(clamp(random.gauss(82, 12), 55, 140), 0))
        hb = round(clamp(random.gauss(11.7, 1.6), 7.0, 15.5), 1)
        urine = round(clamp(random.gauss(120, 70), 10, 450), 1)
        gravida = random.randint(0, 6)
        para = random.randint(0, gravida)
        weight = round(clamp(random.gauss(68, 12), 40, 120), 1)
        height = round(clamp(random.gauss(158, 8), 140, 185), 1)
        comp = 1 if random.random() < 0.28 else 0
        stress = random.randint(0, 10)
        activity = random.choices([0,1,2],[0.25,0.5,0.25])[0]
        edema = 1 if random.random() < 0.22 else 0
        smoke = 1 if random.random() < 0.17 else 0

        bmi = weight / ((height/100)**2)
        risk = 0
        risk += 0.8 if age > 33 else 0
        risk += 1.0 if bp > 132 else 0
        risk += 1.1 if sugar > 122 else 0
        risk += 0.8 if temp > 99.6 else 0
        risk += 0.9 if hr > 102 else 0
        risk += 1.1 if hb < 10.4 else 0
        risk += 1.0 if urine > 185 else 0
        risk += 0.7 if gravida >= 4 else 0
        risk += 0.7 if bmi > 30 else 0
        risk += 1.2 if comp else 0
        risk += 0.8 if stress >= 7 else 0
        risk += 0.5 if activity == 0 else 0
        risk += 1.0 if edema else 0
        risk += 0.8 if smoke else 0

        prob = 1/(1+math.exp(-(-2.0+risk)))
        medium = 1 if random.random() < prob else 0
        if random.random() < 0.08:
            medium = 1 - medium
        low = 1 - medium

        w.writerow([age,bp,sugar,temp,hr,hb,urine,gravida,para,weight,height,comp,stress,activity,edema,smoke,low,medium])

print("Saved pregnancy_risk_dataset_1015.csv")
