from __future__ import annotations

import unittest

try:
    from src.diffusion.sumo_ryl_diffusion import SumoRylDiffusionModel
except ImportError:  # pragma: no cover
    SumoRylDiffusionModel = None  # type: ignore[assignment]


@unittest.skipIf(SumoRylDiffusionModel is None, "torch not available in local validation env")
class DiffusionRuntimeDeviceTest(unittest.TestCase):
    def test_set_runtime_device_keeps_cpu_model_working(self) -> None:
        model = SumoRylDiffusionModel(num_car=2, device="cpu", num_epochs=1, batch_size=1)
        self.assertEqual(str(model.device), "cpu")

        # no-op move should stay stable and not rebuild into an invalid state
        model.set_runtime_device("cpu")

        self.assertEqual(str(model.device), "cpu")
        self.assertEqual(str(model.scheduler.device), "cpu")


if __name__ == "__main__":
    unittest.main()
