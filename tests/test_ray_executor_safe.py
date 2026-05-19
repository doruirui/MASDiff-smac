from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from src.parallel.ray_executor import RayExecutor


class _FailingRemoteFunction:
    def __init__(self, fn):
        self.fn = fn

    def options(self, **kwargs):
        return self

    def remote(self, value):
        raise OSError("Broken pipe")


class _FakeRayModuleForMapFailure:
    class ObjectRef:
        pass

    def __init__(self) -> None:
        self._initialized = False
        self.init_calls = 0
        self.shutdown_calls = 0

    def is_initialized(self):
        return self._initialized

    def init(self, **kwargs):
        self.init_calls += 1
        self._initialized = True

    def shutdown(self):
        self.shutdown_calls += 1
        self._initialized = False

    def remote(self, fn):
        return _FailingRemoteFunction(fn)

    def put(self, value):
        return value

    def cluster_resources(self):
        return {}


class _FakeRayModuleForInitFailure:
    class ObjectRef:
        pass

    def __init__(self) -> None:
        self.init_calls = 0

    def is_initialized(self):
        return False

    def init(self, **kwargs):
        self.init_calls += 1
        raise RuntimeError("The current node timed out during startup.")

    def shutdown(self):
        return None


class RayExecutorSafeTest(unittest.TestCase):
    def test_raises_when_ray_init_fails(self) -> None:
        fake_ray = _FakeRayModuleForInitFailure()
        with patch.dict(sys.modules, {"ray": fake_ray}):
            with self.assertRaises(RuntimeError):
                RayExecutor(show_progress=False, ray_startup_retries=2, ray_startup_retry_delay_s=0.0)

        self.assertEqual(fake_ray.init_calls, 2)

    def test_retries_transport_error_once_then_raises(self) -> None:
        fake_ray = _FakeRayModuleForMapFailure()
        with patch.dict(sys.modules, {"ray": fake_ray}):
            executor = RayExecutor(show_progress=False)
            with self.assertRaises(OSError):
                executor.map(lambda x: x + 1, [1, 2, 3])

        self.assertEqual(fake_ray.init_calls, 2)
        self.assertGreaterEqual(fake_ray.shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
