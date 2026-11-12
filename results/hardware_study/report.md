# Hardware study: trained text classifiers on ibm_marrakesh and two noise models

## cve on fake_fez (100 inputs, 1024 shots, 1 run(s), sessions xdev)

- Ideal accuracy on these inputs: 0.650
- Accuracy per run (Wilson 95% CI): 0.640 [0.542, 0.727]
- Flips vs ideal per run: [3]; expected from shot noise alone: 3.09 per run; P(total >= 3) = 0.63
- Mean signed shift of P(class 1): per run [-0.0092]; pooled -0.0092 (95% CI [-0.0135, -0.0050])
  - run 1: accuracy 0.640; McNemar vs noise-free: device-only correct 1, noise-free-only correct 2, p = 1; flips 3 (p = 0.63); bias -0.0092 [-0.0134, -0.0049]
- Qubits [0, 1, 2, 3]; readout error 0.0049-0.0159; two-qubit error 0.0030-0.0102; compiled depth 37, 6 two-qubit gates

## cve on fake_manila (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.695 [0.628, 0.755]
- Flips vs ideal per run: [12]; expected from shot noise alone: 4.43 per run; P(total >= 12) = 0.000295
- Mean signed shift of P(class 1): per run [-0.0253]; pooled -0.0253 (95% CI [-0.0299, -0.0209])
  - run 1: accuracy 0.695; McNemar vs noise-free: device-only correct 7, noise-free-only correct 5, p = 0.774; flips 12 (p = 0.000295); bias -0.0253 [-0.0297, -0.0206]
- Qubits [0, 1, 2, 3]; readout error 0.0144-0.0964; two-qubit error 0.0088-0.0139; compiled depth 24, 6 two-qubit gates

## cve on fake_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.680 [0.612, 0.741]
- Flips vs ideal per run: [3]; expected from shot noise alone: 4.43 per run; P(total >= 3) = 0.857
- Mean signed shift of P(class 1): per run [-0.0043]; pooled -0.0043 (95% CI [-0.0066, -0.0019])
  - run 1: accuracy 0.680; McNemar vs noise-free: device-only correct 1, noise-free-only correct 2, p = 1; flips 3 (p = 0.857); bias -0.0043 [-0.0066, -0.0019]
- Qubits [0, 1, 2, 3]; readout error 0.0042-0.0112; two-qubit error 0.0015-0.0032; compiled depth 37, 6 two-qubit gates

## cve on ibm_fez (100 inputs, 1024 shots, 1 run(s), sessions xdev)

- Ideal accuracy on these inputs: 0.650
- Accuracy per run (Wilson 95% CI): 0.640 [0.542, 0.727]
- Flips vs ideal per run: [5]; expected from shot noise alone: 3.09 per run; P(total >= 5) = 0.174
- Mean signed shift of P(class 1): per run [-0.0152]; pooled -0.0152 (95% CI [-0.0208, -0.0097])
  - run 1: accuracy 0.640; McNemar vs noise-free: device-only correct 2, noise-free-only correct 3, p = 1; flips 5 (p = 0.174); bias -0.0152 [-0.0208, -0.0094]
- Qubits [0, 1, 2, 3]; readout error 0.0039-0.0293; two-qubit error 0.0016-0.0072; compiled depth 37, 6 two-qubit gates
- Job IDs: dbqhh8gk09pc73at9bc0

## cve on ibm_kingston (100 inputs, 1024 shots, 1 run(s), sessions xdev)

- Ideal accuracy on these inputs: 0.650
- Accuracy per run (Wilson 95% CI): 0.630 [0.532, 0.718]
- Flips vs ideal per run: [2]; expected from shot noise alone: 3.09 per run; P(total >= 2) = 0.854
- Mean signed shift of P(class 1): per run [-0.0087]; pooled -0.0087 (95% CI [-0.0122, -0.0056])
  - run 1: accuracy 0.630; McNemar vs noise-free: device-only correct 0, noise-free-only correct 2, p = 0.5; flips 2 (p = 0.854); bias -0.0087 [-0.0121, -0.0054]
- Qubits [0, 1, 2, 3]; readout error 0.0049-0.0229; two-qubit error 0.0010-0.0060; compiled depth 37, 6 two-qubit gates
- Job IDs: dbqhgr8k09pc73at9aig

## cve on ibm_marrakesh (200 inputs, 1024 shots, 3 run(s), sessions pilot, session1, session2)

- Ideal accuracy on these inputs: 0.685
- Accuracy per run (Wilson 95% CI): 0.685 [0.618, 0.745], 0.655 [0.587, 0.717], 0.700 [0.633, 0.759]
- Flips vs ideal per run: [4, 8, 7]; expected from shot noise alone: 4.43 per run; P(total >= 19) = 0.0548
- Mean signed shift of P(class 1): per run [-0.0063, -0.0053, -0.0022]; pooled -0.0046 (95% CI [-0.0067, -0.0026])
  - run 1: accuracy 0.685; McNemar vs noise-free: device-only correct 2, noise-free-only correct 2, p = 1; flips 4 (p = 0.68); bias -0.0063 [-0.0090, -0.0037]
  - run 2: accuracy 0.655; McNemar vs noise-free: device-only correct 1, noise-free-only correct 7, p = 0.0703; flips 8 (p = 0.0536); bias -0.0053 [-0.0081, -0.0025]
  - run 3: accuracy 0.700; McNemar vs noise-free: device-only correct 5, noise-free-only correct 2, p = 0.453; flips 7 (p = 0.131); bias -0.0022 [-0.0046, +0.0002]
- Pooled bias, two-level bootstrap over runs and inputs: [-0.0076, -0.0016]
- Run-to-run prediction agreement: 0.960
- Qubits [0, 1, 2, 3]; readout error 0.0059-0.0239; two-qubit error 0.0022-0.0027; compiled depth 37, 6 two-qubit gates
- Job IDs: dbov06tt7fqc7388gab0, dbov3ttt7fqc7388gfrg, dbp1pvgou4as73cst2n0

## sentiment on fake_manila (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.730 [0.665, 0.787]
- Flips vs ideal per run: [8]; expected from shot noise alone: 3.07 per run; P(total >= 8) = 0.00323
- Mean signed shift of P(class 1): per run [-0.0209]; pooled -0.0209 (95% CI [-0.0246, -0.0167])
  - run 1: accuracy 0.730; McNemar vs noise-free: device-only correct 3, noise-free-only correct 5, p = 0.727; flips 8 (p = 0.00323); bias -0.0209 [-0.0246, -0.0167]
- Qubits [0, 1, 2, 3]; readout error 0.0144-0.0964; two-qubit error 0.0088-0.0139; compiled depth 24, 6 two-qubit gates

## sentiment on fake_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.745 [0.680, 0.800]
- Flips vs ideal per run: [3]; expected from shot noise alone: 3.07 per run; P(total >= 3) = 0.629
- Mean signed shift of P(class 1): per run [-0.0027]; pooled -0.0027 (95% CI [-0.0046, -0.0007])
  - run 1: accuracy 0.745; McNemar vs noise-free: device-only correct 2, noise-free-only correct 1, p = 1; flips 3 (p = 0.629); bias -0.0027 [-0.0047, -0.0008]
- Qubits [0, 1, 2, 3]; readout error 0.0042-0.0112; two-qubit error 0.0015-0.0032; compiled depth 37, 6 two-qubit gates

## sentiment on ibm_marrakesh (200 inputs, 1024 shots, 1 run(s), sessions session1)

- Ideal accuracy on these inputs: 0.740
- Accuracy per run (Wilson 95% CI): 0.730 [0.665, 0.787]
- Flips vs ideal per run: [4]; expected from shot noise alone: 3.07 per run; P(total >= 4) = 0.365
- Mean signed shift of P(class 1): per run [-0.0035]; pooled -0.0035 (95% CI [-0.0059, -0.0013])
  - run 1: accuracy 0.730; McNemar vs noise-free: device-only correct 1, noise-free-only correct 3, p = 0.625; flips 4 (p = 0.365); bias -0.0035 [-0.0061, -0.0010]
- Qubits [0, 1, 2, 3]; readout error 0.0059-0.0239; two-qubit error 0.0022-0.0027; compiled depth 37, 6 two-qubit gates
- Job IDs: dbov4itt7fqc7388ggog

