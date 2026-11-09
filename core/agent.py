from core.memory import Memory


class QuantumAgent:
    """Perceive -> reason -> act -> reflect loop around a quantum text classifier."""

    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory or Memory()

    def perceive(self, text):
        """Text -> rotation angles for the QNN."""
        return self.model.encode([text])[0]

    def reason(self, angles):
        """QNN forward pass -> P(class 1)."""
        from core.qnn_core import predict_proba
        return float(predict_proba(self.model.qnn, angles, self.model.weights)[0])

    def act(self, p1):
        """Turn the probability into a reply."""
        neg, pos = self.model.labels
        label, confidence = (pos, p1) if p1 >= 0.5 else (neg, 1 - p1)
        strength = "clearly" if confidence >= 0.75 else "leaning" if confidence >= 0.6 else "unsure, slightly"
        return f"{label} ({strength}; P({pos}) = {p1:.2f})"

    def reflect(self, text, p1):
        """Remember the exchange so the session can be summarised."""
        self.memory.store(text, p1)

    def summary(self):
        scores = [p for _, p in self.memory.entries]
        if not scores:
            return "Nothing scored yet."
        neg, pos = self.model.labels
        n_pos = sum(p >= 0.5 for p in scores)
        return (f"Scored {len(scores)} message(s): {n_pos} {pos}, {len(scores) - n_pos} {neg}, "
                f"mean P({pos}) = {sum(scores) / len(scores):.2f}")
