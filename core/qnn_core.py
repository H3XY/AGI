from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit import Aer
import numpy as np

def build_qnn(seed=123):
    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    backend = Aer.get_backend('aer_simulator')
    quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)

    qnn = CircuitQNN(
        circuit=feature_map.compose(ansatz),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        quantum_instance=quantum_instance,
        input_gradients=True,
        output_shape=1,
        interpret=lambda x: x[0]  # simple interpret function to extract output
    )
    return qnn

def predict_with_qnn(qnn, x, weights=None):
    x = np.asarray(x).reshape(-1)
    try:
        if weights is None:
            weights = np.zeros(qnn.num_weights)  # default zero weights
        out = qnn.forward(x, weights)
        return float(out[0]) if hasattr(out, "__len__") else float(out)
    except Exception as e:
        print("[qnn_core] quantum prediction failed, falling back to classical heuristic:", e)
        return float(np.tanh(np.sum(x)))

def train_qnn(qnn, data_x, data_y, epochs=100, lr=0.1):
    """
    Simple training loop using gradient descent on qnn weights.
    data_x: input features, shape (num_samples, feature_dim)
    data_y: target outputs, shape (num_samples,)
    """
    weights = np.zeros(qnn.num_weights)
    
    for epoch in range(epochs):
        total_loss = 0
        grad = np.zeros_like(weights)
        
        for x, y_true in zip(data_x, data_y):
            x = np.asarray(x).reshape(-1)
            y_pred = qnn.forward(x, weights)
            y_pred_val = float(y_pred[0]) if hasattr(y_pred, "__len__") else float(y_pred)
            error = y_pred_val - y_true
            total_loss += error**2
            
            # Compute gradient wrt weights using QNN's backward method
            # NOTE: CircuitQNN supports backward() returning gradients wrt inputs and weights
            input_grad, weight_grad = qnn.backward(x, weights)
            grad += 2 * error * weight_grad  # chain rule
            
        # Update weights with average gradient
        weights -= lr * grad / len(data_x)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data_x):.4f}")
    
    return weights
