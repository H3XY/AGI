# Example script to run a forward pass on an IBM backend (requires IBMQ account)
from qiskit import IBMQ
from qiskit import transpile
from qiskit import QuantumCircuit, Aer
from qiskit.providers.ibmq import least_busy
import os
from core.qnn_core import build_qnn, predict_with_qnn
from core.input_interface import text_to_angles

def get_ibmq_backend():
    token = os.environ.get('IBMQ_TOKEN', None)
    if token is None:
        print('Set IBMQ_TOKEN environment variable or save account via IBMQ.save_account')
        return None
    # Save account locally (will create ~/.qiskit)
    try:
        IBMQ.save_account(token, overwrite=True)
    except Exception:
        pass
    provider = IBMQ.load_account()
    backends = provider.backends(filters=lambda b: b.configuration().n_qubits >= 5 and not b.configuration().simulator)
    backend = least_busy(backends)
    print('Selected backend:', backend.name())
    return backend

def run_on_ibmq(input_text):
    backend = get_ibmq_backend()
    if backend is None:
        return None
    qnn = build_qnn()
    x = text_to_angles(input_text)
    try:
        out = predict_with_qnn(qnn, x)
        print('Prediction:', out)
        return out
    except Exception as e:
        print('Run failed:', e)
        return None

if __name__ == '__main__':
    run_on_ibmq('hello')
