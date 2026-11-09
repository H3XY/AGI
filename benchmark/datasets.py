"""Binary-classification datasets for the benchmark.

Tabular loaders return (X, y, description) and are split 70/30 by the benchmark.
Text tasks (sentiment, cve) use their own fixed train/test splits from
core/text_tasks.py, with the feature reducer fitted on the full training split.
"""
from sklearn.datasets import load_breast_cancer, load_iris, make_moons


def breast_cancer():
    d = load_breast_cancer()
    # sklearn: 0 = malignant, 1 = benign
    return d.data, d.target, "Breast cancer (malignant vs benign), 569 samples, 30 features"


def iris_versicolor_virginica():
    d = load_iris()
    mask = d.target > 0  # drop setosa, which is trivially separable
    return d.data[mask], (d.target[mask] == 2).astype(int), \
        "Iris (versicolor vs virginica), 100 samples, 4 features"


def moons(seed=0):
    X, y = make_moons(n_samples=300, noise=0.2, random_state=seed)
    return X, y, "Two moons (synthetic), 300 samples, 2 features"


# name -> {kind, loader (tabular) or task (text), qubits, feature reduction}
DATASETS = {
    "breast_cancer": {"kind": "tabular", "loader": breast_cancer, "qubits": 4, "reduce": "pca"},
    "iris": {"kind": "tabular", "loader": iris_versicolor_virginica, "qubits": 4, "reduce": "pca"},
    "moons": {"kind": "tabular", "loader": moons, "qubits": 2, "reduce": "pca"},
    "sentiment": {"kind": "text", "task": "sentiment", "qubits": 4, "reduce": "pls"},
    "cve": {"kind": "text", "task": "cve", "qubits": 4, "reduce": "pls"},
}
