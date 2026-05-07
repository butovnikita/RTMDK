"""Tests for encryption at rest."""

import base64
import os

import pytest

from rtmdk.production.encryption import EncryptionManager


class TestEncryptionManager:
    def test_roundtrip_bytes(self):
        key = os.urandom(32)
        mgr = EncryptionManager(key=key)
        plaintext = b"hello world"
        ciphertext = mgr.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = mgr.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_roundtrip_str(self):
        key = os.urandom(32)
        mgr = EncryptionManager(key=key)
        plaintext = "hello world 🌍"
        ciphertext = mgr.encrypt_str(plaintext)
        decrypted = mgr.decrypt_str(ciphertext)
        assert decrypted == plaintext

    def test_roundtrip_json(self):
        key = os.urandom(32)
        mgr = EncryptionManager(key=key)
        obj = {"a": 1, "b": [2, 3], "c": "text"}
        ciphertext = mgr.encrypt_json(obj)
        decrypted = mgr.decrypt_json(ciphertext)
        assert decrypted == obj

    def test_different_nonces(self):
        key = os.urandom(32)
        mgr = EncryptionManager(key=key)
        c1 = mgr.encrypt(b"data")
        c2 = mgr.encrypt(b"data")
        assert c1 != c2

    def test_enabled_from_env(self, monkeypatch):
        key = base64.b64encode(os.urandom(32)).decode("ascii")
        monkeypatch.setenv("RTMDK_ENCRYPTION_KEY", key)
        mgr = EncryptionManager()
        assert mgr.enabled

    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("RTMDK_ENCRYPTION_KEY", raising=False)
        mgr = EncryptionManager()
        assert not mgr.enabled

    def test_invalid_ciphertext(self):
        key = os.urandom(32)
        mgr = EncryptionManager(key=key)
        with pytest.raises(ValueError):
            mgr.decrypt(b"short")
