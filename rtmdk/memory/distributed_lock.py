"""Cross-process distributed lock via file locking or Redis.

File backend works on both Windows (msvcrt) and Unix (fcntl).
Redis backend requires `redis-py` and a Redis server.
"""
from __future__ import annotations
import os
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DistributedLock:
    """Distributed lock with file or Redis backend.

    Usage:
        with DistributedLock("/tmp/rtmdk.lock", timeout=5.0):
            # critical section
            pass

        # Redis:
        with DistributedLock("rtmdk:lock", backend="redis",
                             redis_url="redis://localhost:6379", timeout=5.0):
            pass
    """

    def __init__(
        self,
        lock_path: str,
        timeout: Optional[float] = None,
        backend: str = "file",
        redis_url: Optional[str] = None,
    ):
        self.lock_path = lock_path
        self.timeout = timeout
        self.backend = backend
        self.redis_url = redis_url
        self._fd: Optional[int] = None
        self._owned = False
        self._thread_lock = threading.Lock()
        self._redis_client = None

        if backend == "redis":
            try:
                import redis
                self._redis_client = redis.from_url(redis_url or "redis://localhost:6379")
                self._redis_client.ping()
                logger.info("DistributedLock: Redis backend connected")
            except Exception:
                logger.warning("DistributedLock: Redis unavailable, falling back to file", exc_info=True)
                self.backend = "file"
                self._redis_client = None

    def _acquire_file(self) -> bool:
        """Platform-specific file acquire."""
        import platform
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self._fd = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR | os.O_TRUNC,
        )
        try:
            if platform.system() == "Windows":
                import msvcrt
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                self._owned = True
                return True
            else:
                import fcntl
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._owned = True
                return True
        except (IOError, OSError):
            os.close(self._fd)
            self._fd = None
            return False

    def _acquire_redis(self) -> bool:
        """Redis SET NX acquire."""
        if self._redis_client is None:
            return False
        import redis
        timeout_ms = int((self.timeout or 10) * 1000)
        try:
            acquired = self._redis_client.set(
                self.lock_path, "1", nx=True, px=timeout_ms
            )
            if acquired:
                self._owned = True
                return True
        except redis.RedisError:
            pass
        return False

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock.

        Args:
            blocking: If True, block until lock is available or timeout.

        Returns:
            True if lock was acquired.
        """
        if not self._thread_lock.acquire(blocking=blocking, timeout=self.timeout or -1):
            return False
        try:
            if not blocking:
                acquired = (
                    self._acquire_redis()
                    if self.backend == "redis"
                    else self._acquire_file()
                )
                if not acquired:
                    self._thread_lock.release()
                    return False
                return True

            deadline = time.time() + self.timeout if self.timeout else None
            while True:
                acquired = (
                    self._acquire_redis()
                    if self.backend == "redis"
                    else self._acquire_file()
                )
                if acquired:
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
        if not self._owned:
            self._thread_lock.release()
            return
        if self.backend == "redis" and self._redis_client is not None:
            try:
                import redis
                self._redis_client.delete(self.lock_path)
            except redis.RedisError:
                pass
        else:
            if self._fd is not None:
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
