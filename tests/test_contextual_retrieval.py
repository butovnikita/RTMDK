"""Tests for rtmdk/production/contextual_retrieval.py — context headers for chunks."""

import numpy as np

from rtmdk.production.contextual_retrieval import ContextualEmbedderWrapper, ContextualHeaderGenerator


class TestHeuristicBackend:
    def test_empty_text(self):
        gen = ContextualHeaderGenerator()
        assert gen.generate("") == ""

    def test_short_first_sentence_returned_whole(self):
        gen = ContextualHeaderGenerator()
        assert gen.generate("Cats are cute. Dogs are loyal.") == "Cats are cute"

    def test_long_first_sentence_truncated_to_12_words(self):
        gen = ContextualHeaderGenerator()
        text = " ".join(f"word{i}" for i in range(30)) + ". Second sentence."
        header = gen.generate(text)

        assert header.endswith("...")
        assert len(header.split()) == 12  # "..." is glued to the 12th word
        assert header.startswith("word0")
        assert "word12" not in header

    def test_default_backend_is_heuristic(self):
        gen = ContextualHeaderGenerator()
        assert gen.backend == "heuristic"


class _FakeTokenizer:
    def __init__(self, tokens):
        self._tokens = tokens

    def encode(self, text):
        return self._tokens


class _ExplodingTokenizer:
    def encode(self, text):
        raise RuntimeError("tokenizer failure")


class TestSotBackend:
    def test_sot_header_uses_top_tokens(self):
        gen = ContextualHeaderGenerator(backend="sot", sot_tokenizer=_FakeTokenizer([7, 3, 7, 5, 1, 9, 2]))
        header = gen.generate("some text")

        # Unique tokens in order, capped at 5
        assert header == "7 3 5 1 9"

    def test_sot_without_tokenizer_falls_back(self):
        gen = ContextualHeaderGenerator(backend="sot", sot_tokenizer=None)
        assert gen.generate("Fallback works. More.") == "Fallback works"

    def test_sot_empty_tokens_falls_back(self):
        gen = ContextualHeaderGenerator(backend="sot", sot_tokenizer=_FakeTokenizer([]))
        assert gen.generate("Fallback works. More.") == "Fallback works"

    def test_sot_exception_falls_back(self):
        gen = ContextualHeaderGenerator(backend="sot", sot_tokenizer=_ExplodingTokenizer())
        assert gen.generate("Fallback works. More.") == "Fallback works"


class TestEmbedderWrapper:
    def test_header_prepended_before_embedding(self):
        captured = {}

        def embedder(text):
            captured["text"] = text
            return np.zeros(4, dtype=np.float32)

        gen = ContextualHeaderGenerator()
        wrapper = ContextualEmbedderWrapper(embedder, gen)
        result = wrapper("Cats are cute. Dogs are loyal.")

        assert captured["text"] == "Cats are cute\n\nCats are cute. Dogs are loyal."
        assert result.shape == (4,)

    def test_empty_text_embedded_as_is(self):
        captured = {}

        def embedder(text):
            captured["text"] = text
            return np.zeros(4, dtype=np.float32)

        wrapper = ContextualEmbedderWrapper(embedder, ContextualHeaderGenerator())
        wrapper("")

        assert captured["text"] == ""
