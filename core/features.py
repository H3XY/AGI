"""Turn raw feature vectors into rotation angles the QNN can encode.

standardise -> reduce to one feature per qubit -> rescale to [0, pi].

Reduction is either
  - "pca": unsupervised, keeps the directions with the most variance, or
  - "pls": supervised (partial least squares), keeps the directions that best
    predict the label. Much better when the label signal is spread thinly over
    many dimensions, e.g. sentence embeddings.
Fit on training data only, then reuse for test data and live inputs.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler


def clip_to_angles(a):
    # live inputs can fall outside the training range; keep them in [0, pi]
    return np.clip(a, 0.0, np.pi)


class PLSReducer(BaseEstimator, TransformerMixin):
    """PLS as a plain transformer (PLSRegression.fit_transform returns a tuple)."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


def make_angle_pipeline(num_qubits, seed=0, method="pca"):
    if method == "pca":
        reducer = PCA(n_components=num_qubits, random_state=seed)
    elif method == "pls":
        reducer = PLSReducer(n_components=num_qubits)
    else:
        raise ValueError(f"unknown reduction method: {method}")
    return Pipeline([
        ("standardise", StandardScaler()),
        ("reduce", reducer),
        ("to_angles", MinMaxScaler(feature_range=(0.0, np.pi))),
        ("clip", FunctionTransformer(clip_to_angles)),
    ])
