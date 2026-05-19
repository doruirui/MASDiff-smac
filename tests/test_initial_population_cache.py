from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline.steps import step_3_4_load_initial_population, step_3_4_save_initial_population
from src.pipeline.types import Individual


class InitialPopulationCacheTest(unittest.TestCase):
    def test_cache_keeps_only_fields_needed_by_only_astar_path(self) -> None:
        population = [
            Individual(
                tau="tau0",
                rewards="r0",
                rho=1.0,
                experience_buffers=[["exp0"]],
                policies=["policy0"],
                metadata={"simulation_data": "sim0", "initial_index": 0},
            ),
            Individual(
                tau="tau1",
                rewards="r1",
                rho=2.0,
                experience_buffers=[["exp1"]],
                policies=["policy1"],
                metadata={"simulation_data": "sim1", "initial_index": 1},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "initial_population.pkl"
            step_3_4_save_initial_population(population, population_path=str(cache_path))
            restored = step_3_4_load_initial_population(str(cache_path))

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].tau, "tau0")
        self.assertEqual(restored[0].rewards, "r0")
        self.assertEqual(restored[0].rho, 1.0)
        self.assertEqual(restored[0].experience_buffers, [])
        self.assertEqual(restored[0].policies, [])
        self.assertEqual(restored[0].metadata, {})

        self.assertEqual(restored[1].tau, "tau1")
        self.assertEqual(restored[1].rewards, "r1")
        self.assertEqual(restored[1].rho, 2.0)
        self.assertEqual(restored[1].experience_buffers, [])
        self.assertEqual(restored[1].policies, [])
        self.assertEqual(restored[1].metadata, {"simulation_data": "sim1"})
