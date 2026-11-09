"""Chat with a quantum text classifier.

    python run.py                 # sentiment: is this opinion positive or negative?
    python run.py --task cve      # CVE severity: paste a vulnerability description

The trained 4-qubit QNN runs on a local quantum simulator.
Commands while chatting:  /summary  /exit
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.agent import QuantumAgent  # noqa: E402
from core.text_classifier import TextClassifier  # noqa: E402
from core.text_tasks import TASKS  # noqa: E402

PROMPTS = {
    "sentiment": "Type an opinion (movie-review style works best)",
    "cve": "Paste a vulnerability description",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASKS), default="sentiment")
    args = p.parse_args()

    try:
        model = TextClassifier(args.task)
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    print("Loading text encoder...")
    model.encode(["warm-up"])  # loads the embedding model once, up front
    agent = QuantumAgent(model)

    acc = model.metrics.get("Quantum (ideal sim)")
    print(f"\n{TASKS[args.task]['description']}: {model.num_qubits}-qubit classifier ready"
          + (f" ({acc:.0%} test accuracy)" if acc else "") + ".")
    print(f"{PROMPTS[args.task]}, /summary for session stats, /exit to quit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("/exit", "exit", "quit"):
            break
        if text.lower() == "/summary":
            print("AGI:", agent.summary())
            continue

        p1 = agent.reason(agent.perceive(text))
        print("AGI:", agent.act(p1))
        agent.reflect(text, p1)

    print(agent.summary())


if __name__ == "__main__":
    main()
