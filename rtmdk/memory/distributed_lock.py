"""Cross-process distributed lock via file locking.

Works on both Windows (msvcrt) and Unix (fcntl) without external deps.
"""
from __future__ import annotations
import os
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DistributedLock:
    """File-based inter-process lock with optional timeout.

    Usage:
        with DistributedLock("/tmp/rtmdk.lock", timeout=5.0):
            # critical section
            pass
    """

    def __init__(self, lock_path: str, timeout: Optional[float] = None):
        self.lock_path = lock_path
        self.timeout = timeout
        self._fd: Optional[int] = None
        self._owned = False
        self._thread_lock = threading.Lock()

    def _acquire(self) -> bool:
        """Platform-specific acquire."""
        import platform
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        # Create file if not exists
        self._fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | os.O_TRUNC,
        )
        try:
            if platform.system() == "Windows":
                import msvcrt
                # Windows: lock first 1 byte
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                self._owned = True
                return True
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._owned = True
                return True
        except (IOError, OSError):
            # Lock held by another process
            os.close(self._fd)
            self._fd = None
            return False

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock.

        Args:
            blocking: If True, block until lock is available or timeout.

        Returns:
            True if lock was acquired.
        """
        # First acquire thread-level lock (intra-process)
        if not self._thread_lock.acquire(blocking=blocking, timeout=self.timeout or -1):
            return False
        try:
            if not blocking:
                if not self._acquire():
                    self._thread_lock.release()
                    return False
                return True

            deadline = time.time() + self.timeout if self.timeout else None
            while True:
                if self._acquire():
                    return True
                if deadline and time.time() > deadline:
                    logger.warning("DistributedLock timeout: %s", self.lock_path)
                    self._thread_lock.release()
                    return False
                time.sleep(0.05)
        except Exception:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        """Release the lock."""
        if not self._owned or self._fd is None:
            self._thread_lock.release()
            return
        try:
            import platform
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        self._owned = False
        self._thread_lock.release()

    def __enter__(self) -> "DistributedLock":
        if not self.acquire(blocking=True):
            raise TimeoutError(f"Could not acquire lock: {self.lock_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
