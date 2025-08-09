import collections

class Memory:
    def __init__(self, capacity=256):
        self.entries = collections.deque(maxlen=capacity)

    def store(self, input_text, outcome):
        self.entries.append((input_text, outcome))

    def recall_recent(self, n=5):
        return list(self.entries)[-n:]
