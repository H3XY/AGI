"""Quantum neural network (QNN) core.

Circuit (data re-uploading classifier), repeated `reps` times:
    RY(x_i) on each qubit i        (angle-encode the input features)
    RealAmplitudes(reps=1) block   (trainable RY rotations + CNOT entanglers)
then measure every qubit. The class is qubit 0's measured bit, so the network
outputs [P(class 0), P(class 1)].

Inputs must already be scaled to [0, pi], one feature per qubit
(see core/features.py).

Why this design: an earlier version used ZZFeatureMap + parity readout and
trained on sampled (shot-noisy) probabilities. On the benchmark datasets it
barely beat a coin flip. Plain angle encoding, re-uploading the input before
every trainable layer, and training on exact probabilities fixed that.
"""
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import L_BFGS_B

NUM_QUBITS = 2  # default for the original 2-feature demo


def _qubit0(bitstring_index):
    """Class = measured value of qubit 0 (the least significant bit)."""
    return bitstring_index & 1


def build_circuit(num_qubits=NUM_QUBITS, reps=2):
    x = ParameterVector("x", num_qubits)
    qc = QuantumCircuit(num_qubits)
    weights = []
    for layer in range(reps):
        for i in range(num_qubits):
            qc.ry(x[i], i)
        block = RealAmplitudes(num_qubits, reps=1, parameter_prefix=f"w{layer}")
        qc.compose(block, inplace=True)
        weights += list(block.parameters)
    qc.measure_all()
    return qc, list(x), weights


def exact_sampler():
    """Noise-free sampler that returns exact probabilities (no shot noise).

    Uses Qiskit 1.x's reference Sampler, which is deprecated in favour of
    StatevectorSampler; StatevectorSampler always samples shots, which makes
    training stall, so we keep the exact one while Qiskit is pinned < 2.
    """
    from qiskit.primitives import Sampler
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Sampler()


def build_qnn(num_qubits=NUM_QUBITS, reps=2, sampler=None, pass_manager=None,
              input_gradients=False, **_ignored):
    """Build the QNN.

    By default it runs on an exact local simulator. To run on IBM hardware or
    a noisy fake backend, pass an IBM Runtime SamplerV2 as `sampler` and a
    pass manager for that backend as `pass_manager` (see ibmq/run_on_ibmq.py).
    """
    circuit, input_params, weight_params = build_circuit(num_qubits, reps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "No gradient function provided" notice
        return SamplerQNN(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            sampler=sampler if sampler is not None else exact_sampler(),
            interpret=_qubit0,
            output_shape=2,
            input_gradients=input_gradients,
            pass_manager=pass_manager,
        )


def predict_proba(qnn, X, weights):
    """P(class 1) for each row of X (shape: samples x num_qubits)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.asarray(qnn.forward(X, weights))[:, 1]


def predict_with_qnn(qnn, x, weights=None):
    """P(class 1) for a single input; falls back to a classical heuristic on error."""
    x = np.asarray(x, dtype=float).reshape(-1)
    try:
        if weights is None:
            weights = np.zeros(qnn.num_weights)  # untrained
        return float(predict_proba(qnn, x, weights)[0])
    except Exception as e:
        print("[qnn_core] quantum prediction failed, falling back to classical heuristic:", e)
        return float(np.tanh(np.sum(x)))


def train_classifier(qnn, X, y, maxiter=60, seed=0, verbose=True):
    """Train the QNN's weights with L-BFGS-B on cross-entropy loss.

    X: features scaled to [0, pi], shape (samples, num_qubits)
    y: labels 0/1
    Returns (trained weight vector, list of losses per iteration).
    """
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-np.pi, np.pi, qnn.num_weights)
    losses = []

    def callback(_weights, loss):
        losses.append(loss)
        if verbose and (len(losses) % 25 == 0 or len(losses) == 1):
            print(f"  eval {len(losses):4d}  loss {loss:.4f}", flush=True)

    clf = NeuralNetworkClassifier(
        qnn,
        loss="cross_entropy",
        optimizer=L_BFGS_B(maxiter=maxiter),
        one_hot=True,
        initial_point=initial,
        callback=callback,
    )
    clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return np.asarray(clf.weights), losses
