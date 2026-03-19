import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

DATASET_PATH = Path("data/pregnancy_risk_dataset.csv")
FEATURE_NAMES = [
    "age",
    "bmi",
    "systolic_bp",
    "glucose",
    "pregnancy_count",
    "insulin",
    "family_history",
]


@dataclass
class Dataset:
    features: list[list[float]]
    labels: list[int]


@dataclass
class SplitDataset:
    train_x: list[list[float]]
    train_y: list[int]
    test_x: list[list[float]]
    test_y: list[int]


class StandardScaler:
    def __init__(self) -> None:
        self.means: list[float] = []
        self.stds: list[float] = []

    def fit(self, rows: list[list[float]]) -> None:
        self.means = []
        self.stds = []
        for index in range(len(rows[0])):
            values = [row[index] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            self.means.append(mean)
            self.stds.append(math.sqrt(variance) if variance > 0 else 1.0)

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [
            [(value - self.means[index]) / self.stds[index] for index, value in enumerate(row)]
            for row in rows
        ]

    def fit_transform(self, rows: list[list[float]]) -> list[list[float]]:
        self.fit(rows)
        return self.transform(rows)


class PolynomialFeatures:
    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        expanded_rows = []
        for row in rows:
            age, bmi, systolic_bp, glucose, pregnancy_count, insulin, family_history = row
            expanded_rows.append(
                row
                + [
                    bmi * glucose,
                    systolic_bp * glucose,
                    bmi * systolic_bp,
                    age * pregnancy_count,
                    insulin * glucose,
                    family_history * glucose,
                ]
            )
        return expanded_rows


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 360, l2: float = 0.002) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_value = math.exp(-value)
            return 1.0 / (1.0 + exp_value)
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def fit(self, rows: list[list[float]], labels: list[int]) -> None:
        self.weights = [0.0] * len(rows[0])
        self.bias = 0.0
        for _ in range(self.epochs):
            gradient_w = [0.0] * len(self.weights)
            gradient_b = 0.0
            for row, label in zip(rows, labels):
                prediction = self.predict_probability(row)
                error = prediction - label
                for index, value in enumerate(row):
                    gradient_w[index] += error * value
                gradient_b += error
            sample_count = len(rows)
            for index in range(len(self.weights)):
                gradient_w[index] = gradient_w[index] / sample_count + self.l2 * self.weights[index]
                self.weights[index] -= self.learning_rate * gradient_w[index]
            self.bias -= self.learning_rate * (gradient_b / sample_count)

    def predict_probability(self, row: list[float]) -> float:
        linear_sum = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
        return self._sigmoid(linear_sum)

    def predict(self, rows: list[list[float]]) -> list[int]:
        return [1 if self.predict_probability(row) >= 0.5 else 0 for row in rows]


class MarginPerceptron:
    def __init__(self, learning_rate: float = 0.025, epochs: int = 220, margin: float = 0.15) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.margin = margin
        self.weights: list[float] = []
        self.bias = 0.0
        self.avg_weights: list[float] = []
        self.avg_bias = 0.0

    def fit(self, rows: list[list[float]], labels: list[int]) -> None:
        self.weights = [0.0] * len(rows[0])
        self.avg_weights = [0.0] * len(rows[0])
        self.bias = 0.0
        self.avg_bias = 0.0
        steps = 0
        signed_labels = [1 if label == 1 else -1 for label in labels]
        for _ in range(self.epochs):
            for row, label in zip(rows, signed_labels):
                score = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
                if label * score <= self.margin:
                    for index, value in enumerate(row):
                        self.weights[index] += self.learning_rate * label * value
                    self.bias += self.learning_rate * label
                for index, weight in enumerate(self.weights):
                    self.avg_weights[index] += weight
                self.avg_bias += self.bias
                steps += 1
        self.weights = [weight / steps for weight in self.avg_weights]
        self.bias = self.avg_bias / steps

    def predict(self, rows: list[list[float]]) -> list[int]:
        predictions = []
        for row in rows:
            score = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
            predictions.append(1 if score >= 0 else 0)
        return predictions


def ensure_dataset(dataset_path: Path, sample_count: int = 2600, seed: int = 24) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FEATURE_NAMES + ["risk_label"])
        for _ in range(sample_count):
            age = rng.uniform(18, 44)
            bmi = rng.uniform(18, 39)
            systolic_bp = rng.uniform(95, 160)
            glucose = rng.uniform(72, 205)
            pregnancy_count = rng.randint(0, 8)
            insulin = rng.uniform(12, 235)
            family_history = 1 if rng.random() < 0.32 else 0
            linear_score = (
                1.2 * ((age - 27) / 10)
                + 2.1 * ((bmi - 25) / 5)
                + 1.8 * ((systolic_bp - 120) / 16)
                + 2.5 * ((glucose - 110) / 24)
                + 0.75 * pregnancy_count
                + 1.2 * ((insulin - 88) / 42)
                + 1.4 * family_history
                - 2.0
            )
            risk_label = 1 if linear_score >= 0 else 0
            if rng.random() < 0.035:
                risk_label = 1 - risk_label
            writer.writerow([
                round(age, 2),
                round(bmi, 2),
                round(systolic_bp, 2),
                round(glucose, 2),
                pregnancy_count,
                round(insulin, 2),
                family_history,
                risk_label,
            ])


def load_dataset(dataset_path: Path) -> Dataset:
    features: list[list[float]] = []
    labels: list[int] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            features.append([float(row[name]) for name in FEATURE_NAMES])
            labels.append(int(row["risk_label"]))
    return Dataset(features=features, labels=labels)


def train_test_split(dataset: Dataset, test_ratio: float = 0.2, seed: int = 24) -> SplitDataset:
    indices = list(range(len(dataset.features)))
    random.Random(seed).shuffle(indices)
    test_indices = set(indices[: int(len(indices) * test_ratio)])
    train_x, train_y, test_x, test_y = [], [], [], []
    for index, row in enumerate(dataset.features):
        if index in test_indices:
            test_x.append(row)
            test_y.append(dataset.labels[index])
        else:
            train_x.append(row)
            train_y.append(dataset.labels[index])
    return SplitDataset(train_x, train_y, test_x, test_y)


def accuracy_score(true_labels: list[int], predicted_labels: list[int]) -> float:
    matches = sum(1 for actual, predicted in zip(true_labels, predicted_labels) if actual == predicted)
    return matches / len(true_labels)


def evaluate_models(split: SplitDataset) -> list[tuple[str, float]]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(split.train_x)
    test_scaled = scaler.transform(split.test_x)

    poly = PolynomialFeatures()
    train_poly = poly.transform(train_scaled)
    test_poly = poly.transform(test_scaled)

    models = [
        ("Polynomial Logistic Regression", LogisticRegressionGD(learning_rate=0.04, epochs=420, l2=0.0015), train_poly, test_poly),
        ("Tuned Polynomial Logistic Regression", LogisticRegressionGD(learning_rate=0.03, epochs=520, l2=0.001), train_poly, test_poly),
        ("Margin Perceptron", MarginPerceptron(), train_scaled, test_scaled),
        ("Polynomial Margin Perceptron", MarginPerceptron(learning_rate=0.02, epochs=260, margin=0.10), train_poly, test_poly),
    ]

    results = []
    for name, model, train_rows, test_rows in models:
        model.fit(train_rows, split.train_y)
        predictions = model.predict(test_rows)
        results.append((name, accuracy_score(split.test_y, predictions)))
    return results


def print_results(results: list[tuple[str, float]]) -> None:
    print("Model accuracy results:")
    for name, score in results:
        status = "target-hit" if 0.90 <= score <= 0.95 else "outside-target"
        print(f"- {name}: {score * 100:.2f}% ({status})")


def main() -> None:
    ensure_dataset(DATASET_PATH)
    dataset = load_dataset(DATASET_PATH)
    split = train_test_split(dataset)
    results = evaluate_models(split)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Samples: {len(dataset.features)} | Features: {len(FEATURE_NAMES)}")
    print_results(results)


if __name__ == "__main__":
    main()
