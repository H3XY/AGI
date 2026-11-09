"""Sentence embeddings (sentence-transformers all-MiniLM-L6-v2, 384-dim).

The model (~91 MB) is downloaded to the Hugging Face cache on first use and
loaded from there afterwards without contacting the Hub.
"""
import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        try:  # use the cached copy without contacting the Hub
            _model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
        except Exception:  # first run: download into the Hugging Face cache
            _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts):
    """Embed a list of strings -> array (len(texts), 384)."""
    return np.asarray(get_embedder().encode(list(texts), batch_size=64, show_progress_bar=False))
