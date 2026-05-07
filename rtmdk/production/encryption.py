"""rtmdk/production/encryption.py — Encryption at rest.

AES-256-GCM for sensitive data (sessions, WAL, exports).
Master key derived from RTMDK_ENCRYPTION_KEY env var.
"""

import base64
import json
import os
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionManager:
    """Manages AES-256-GCM encryption for data at rest.

    Usage:
        mgr = EncryptionManager()
        ciphertext = mgr.encrypt(b"sensitive data")
        plaintext = mgr.decrypt(ciphertext)
    """

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key_b64 = os.getenv("RTMDK_ENCRYPTION_KEY", "")
            if key_b64:
                key = base64.b64decode(key_b64)
        if key is None or len(key) != 32:
            # Generate a random key for this session if none provided
            # (data will not be decryptable across restarts)
            key = os.urandom(32)
        self._key = key
        self._enabled = os.getenv("RTMDK_ENCRYPTION_KEY", "") != ""

    @property
    def enabled(self) -> bool:
        return self._enabled

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt plaintext; return nonce + ciphertext."""
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data (nonce + ciphertext)."""
        if len(data) < 13:
            raise ValueError("Invalid ciphertext: too short")
        nonce = data[:12]
        ciphertext = data[12:]
        aesgcm = AESGCM(self._key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def encrypt_str(self, plaintext: str) -> str:
        """Encrypt string and return base64-encoded result."""
        return base64.b64encode(self.encrypt(plaintext.encode("utf-8"))).decode("ascii")

    def decrypt_str(self, data: str) -> str:
        """Decrypt base64-encoded string."""
        return self.decrypt(base64.b64decode(data)).decode("utf-8")

    def encrypt_json(self, obj) -> str:
        """Encrypt JSON-serializable object and return base64 string."""
        return self.encrypt_str(json.dumps(obj))

    def decrypt_json(self, data: str):
        """Decrypt base64 string back to JSON object."""
        return json.loads(self.decrypt_str(data))
