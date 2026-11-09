# Hardware study: trained text classifiers on ibm_marrakesh and two noise models

## cve on fake_manila (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.695 [0.628, 0.755]
- Flips vs ideal per run: [12]; expected from shot noise alone: 4.43 per run; P(total >= 12) = 0.000295
- Mean signed shift of P(class 1): per run [-0.0253]; pooled -0.0253 (95% CI [-0.0299, -0.0209])
- Qubits [0, 1, 2, 3]; readout error 0.0144-0.0964; two-qubit error 0.0088-0.0139; compiled depth 24, 6 two-qubit gates

## cve on fake_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.680 [0.612, 0.741]
- Flips vs ideal per run: [3]; expected from shot noise alone: 4.43 per run; P(total >= 3) = 0.857
- Mean signed shift of P(class 1): per run [-0.0043]; pooled -0.0043 (95% CI [-0.0066, -0.0019])
- Qubits [0, 1, 2, 3]; readout error 0.0042-0.0112; two-qubit error 0.0015-0.0032; compiled depth 37, 6 two-qubit gates

## cve on ibm_marrakesh (200 inputs, 1024 shots, 2 run(s), sessions pilot, session1)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.685 [0.618, 0.745], 0.655 [0.587, 0.717]
- Flips vs ideal per run: [4, 8]; expected from shot noise alone: 4.43 per run; P(total >= 12) = 0.154
- Mean signed shift of P(class 1): per run [-0.0063, -0.0053]; pooled -0.0058 (95% CI [-0.0082, -0.0035])
- Run-to-run prediction agreement: 0.960
- Qubits [0, 1, 2, 3]; readout error 0.0059-0.0239; two-qubit error 0.0022-0.0027; compiled depth 37, 6 two-qubit gates
- Job IDs: dbov06tt7fqc7388gab0, dbov3ttt7fqc7388gfrg

## sentiment on fake_manila (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.730 [0.665, 0.787]
- Flips vs ideal per run: [8]; expected from shot noise alone: 3.07 per run; P(total >= 8) = 0.00323
- Mean signed shift of P(class 1): per run [-0.0209]; pooled -0.0209 (95% CI [-0.0246, -0.0167])
- Qubits [0, 1, 2, 3]; readout error 0.0144-0.0964; two-qubit error 0.0088-0.0139; compiled depth 24, 6 two-qubit gates

## sentiment on fake_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.745 [0.680, 0.800]
- Flips vs ideal per run: [3]; expected from shot noise alone: 3.07 per run; P(total >= 3) = 0.629
- Mean signed shift of P(class 1): per run [-0.0027]; pooled -0.0027 (95% CI [-0.0046, -0.0007])
- Qubits [0, 1, 2, 3]; readout error 0.0042-0.0112; two-qubit error 0.0015-0.0032; compiled depth 37, 6 two-qubit gates

## sentiment on ibm_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.730 [0.665, 0.787]
- Flips vs ideal per run: [4]; expected from shot noise alone: 3.07 per run; P(total >= 4) = 0.365
- Mean signed shift of P(class 1): per run [-0.0035]; pooled -0.0035 (95% CI [-0.0059, -0.0013])
- Qubits [0, 1, 2, 3]; readout error 0.0059-0.0239; two-qubit error 0.0022-0.0027; compiled depth 37, 6 two-qubit gates
- Job IDs: dbov4itt7fqc7388ggog

