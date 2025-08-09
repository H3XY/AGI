from core.qnn_core import build_qnn, predict_with_qnn
from core.agent import QuantumAGIAgent
from modules.perception import simple_text_preprocess
import numpy as np

qnn = build_qnn()
default_weights = np.zeros(qnn.num_weights)  # zeros since no training yet

agent = QuantumAGIAgent(qnn)

print("Quantum AGI Prototype Ready. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    processed = simple_text_preprocess(user_input)
    encoded = agent.perceive(processed)
    decision = agent.reason(encoded)

    response_val = predict_with_qnn(qnn, encoded, weights=default_weights)
    response = agent.act(decision, qnn_output=response_val)
    print("AGI:", response)
    agent.reflect(user_input, decision)
