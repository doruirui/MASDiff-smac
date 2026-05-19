from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from src.parallel import ray_tasks
from src.pipeline.types import Individual


class _FakeDiffusion:
    def __init__(self) -> None:
        self.loaded_state = None
        self.trained_population = None

    def train_on_population(self, population):
        self.trained_population = population


class _FakeRayModule:
    @staticmethod
    def get(value):
        return value


class RemoteTrainStage51OnlyAstarTest(unittest.TestCase):
    def test_train_diffusion_on_population_consumes_population_refs(self) -> None:
        diffusion = _FakeDiffusion()
        population = [
            Individual(tau="tau0", rewards="r0", rho=1.0),
            Individual(tau="tau1", rewards="r1", rho=2.0),
        ]

        with patch.dict(sys.modules, {"ray": _FakeRayModule()}):
            with patch.object(ray_tasks, "instantiate", return_value=diffusion):
                with patch.object(
                    ray_tasks,
                    "load_diffusion_state",
                    side_effect=lambda model, state: setattr(model, "loaded_state", state),
                ):
                    with patch.object(
                        ray_tasks,
                        "extract_diffusion_state",
                        return_value={"trained": True},
                    ):
                        result = ray_tasks.train_diffusion_on_population(
                            {
                                "population_refs": population,
                                "diffusion_state_ref": {"weights": 1},
                                "diffusion_model_spec": object(),
                                "seed": 42,
                            }
                        )

        self.assertEqual(diffusion.loaded_state, {"weights": 1})
        self.assertEqual(diffusion.trained_population, population)
        self.assertEqual(result, {"diffusion_state": {"trained": True}, "population_count": 2})


if __name__ == "__main__":
    unittest.main()
