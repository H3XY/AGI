import numpy as np
from core.input_interface import text_to_angles_vector

def simple_text_preprocess(text, dim=2):
    # Tokenize minimally and produce fixed-size vector
    return text_to_angles_vector(text, dim=dim)
