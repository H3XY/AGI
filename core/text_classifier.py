"""Load a trained quantum text classifier (models/<task>.joblib) and score text."""
import os

import joblib
import numpy as np

from core.qnn_core import build_qnn, predict_proba
from core.text_encoder import embed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_path(task):
    return os.path.join(ROOT, "models", f"{task}.joblib")


class TextClassifier:
    def __init__(self, task="sentiment"):
        path = model_path(task)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained model at {path}. Train one first:  python train.py --task {task}")
        saved = joblib.load(path)
        self.task = task
        self.labels = tuple(saved["labels"])
        self.num_qubits = saved["num_qubits"]
        self.reps = saved["reps"]
        self.weights = np.asarray(saved["weights"])
        self.pipeline = saved["pipeline"]
        self.metrics = saved.get("metrics", {})
        self.qnn = build_qnn(num_qubits=self.num_qubits, reps=self.reps)

    def encode(self, texts):
        """Texts -> rotation angles, shape (len(texts), num_qubits)."""
        return self.pipeline.transform(embed(texts))

    def predict(self, texts, qnn=None):
        """P(class 1) for each text. Pass `qnn` to run on another backend."""
        return predict_proba(qnn or self.qnn, self.encode(texts), self.weights)

    def build_qnn_for(self, sampler, pass_manager):
        """Same circuit, different backend (noisy simulator or IBM hardware)."""
        return build_qnn(num_qubits=self.num_qubits, reps=self.reps,
                         sampler=sampler, pass_manager=pass_manager)
