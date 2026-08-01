"""
Standalone FastText/GloVe bootstrap utility for SOT cold-start.

Much lighter than SBERT (~130MB model vs 2GB+ torch stack).
Uses gensim KeyedVectors (word2vec format).

Usage:
    from rtmdk.memory.bootstrap_fasttext import run_bootstrap
    run_bootstrap(
        tokenizer=my_tokenizer,
        model_path="glove-wiki-gigaword-100.model",
    )

Or CLI:
    python -m rtmdk bootstrap-fasttext --model glove.model --output sot_fasttext.npz
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def run_bootstrap(
    tokenizer,
    texts,
    model_path: str = "fasttext_bootstrap.model",
    fallback_to_random: bool = True,
):
    """Bootstrap SOT word embeddings from FastText/GloVe vectors.

    For each word in the tokenizer's vocab, looks up the vector in the
    KeyedVectors model. If found, copies it as the token embedding.
    Also learns a projection matrix if latent_dim != vector_dim.

    Args:
        tokenizer: SOTokenizer instance (word mode recommended).
        model_path: Path to gensim KeyedVectors model file.
        fallback_to_random: If True, keep random init for OOV words.
    """
    if not os.path.exists(model_path):
        logger.warning(f"FastText model not found: {model_path}")
        return

    try:
        from gensim.models import KeyedVectors
    except ImportError:
        logger.warning("gensim not installed, cannot load FastText model")
        return

    logger.info(f"Loading FastText model from {model_path}...")
    model = KeyedVectors.load(model_path)
    vector_dim = model.vector_size
    logger.info(f"Model loaded: vocab={len(model)}, dim={vector_dim}")

    # Ensure vocab is populated from texts
    if tokenizer.tokenization_mode == "word":
        for text in texts:
            tokenizer.encode(text)

    matched = 0
    oov = 0
    vectors = []
    token_ids = []

    for word, tid in tokenizer.word_to_id.items():
        if word in model:
            vec = model[word].astype(np.float32)
            tokenizer.token_embeddings[tid] = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)
            token_ids.append(tid)
            matched += 1
        else:
            oov += 1
            if not fallback_to_random and tid in tokenizer.token_embeddings:
                # Keep existing random init
                pass

    logger.info(f"FastText bootstrap: matched={matched}, oov={oov}")

    if not vectors:
        logger.warning("No words matched FastText vocab")
        return

    # Update tokenizer dimensions to match FastText
    old_token_dim = tokenizer.token_dim
    tokenizer.token_dim = vector_dim
    logger.info(f"Token dim updated: {old_token_dim} -> {vector_dim}")

    # Resize ALL existing embeddings to new dim
    for tid, emb in list(tokenizer.token_embeddings.items()):
        if len(emb) < vector_dim:
            padded = np.zeros(vector_dim, dtype=np.float32)
            padded[: len(emb)] = emb
            tokenizer.token_embeddings[tid] = padded / (np.linalg.norm(padded) + 1e-8)
        elif len(emb) > vector_dim:
            tokenizer.token_embeddings[tid] = emb[:vector_dim] / (np.linalg.norm(emb[:vector_dim]) + 1e-8)

    # Learn projection if dimensions differ
    if vector_dim != tokenizer.latent_dim:
        logger.info(f"Learning projection: {vector_dim} -> {tokenizer.latent_dim}")
        X = np.stack(vectors).astype(np.float32)
        # Center
        mean_vec = X.mean(axis=0, keepdims=True)
        X_centered = X - mean_vec
        # PCA via SVD
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        # Take top latent_dim components
        W = vt[: tokenizer.latent_dim, :].T.astype(np.float32)
        tokenizer.projection = W
        logger.info(f"Projection shape: {W.shape}")
    else:
        tokenizer.projection = np.eye(vector_dim, dtype=np.float32)
        logger.info("Projection is identity (dims match)")


def load_model(model_path: str):
    """Load a gensim KeyedVectors model."""
    from gensim.models import KeyedVectors

    return KeyedVectors.load(model_path)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("Usage: python bootstrap_fasttext.py <model_path> <tokenizer_state.json>")
        sys.exit(1)
    # Standalone usage: load tokenizer state, apply bootstrap, save back
    import json
    from rtmdk.memory.self_organizing_field import SOTokenizer

    with open(sys.argv[2], "r", encoding="utf-8") as f:
        state = json.load(f)
    tok = SOTokenizer(latent_dim=state.get("latent_dim", 64))
    tok.load_state(state)
    run_bootstrap(tok, texts=[], model_path=sys.argv[1])
    with open(sys.argv[2] + "_fasttext.json", "w", encoding="utf-8") as f:
        json.dump(tok.get_state(), f)
    print(f"Saved to {sys.argv[2]}_fasttext.json")
