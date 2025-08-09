import numpy as np

def text_to_angles_vector(text, dim=2):
    # convert text to a fixed-size numeric vector normalized to [0, pi]
    ascii_vals = [ord(c) for c in text]
    vec = [float(v % 128) / 127.0 * np.pi for v in ascii_vals[:dim]]
    # pad if too short
    while len(vec) < dim:
        vec.append(0.0)
    return np.array(vec)

# alias used by agent
def text_to_angles(text):
    return text_to_angles_vector(text, dim=2)
