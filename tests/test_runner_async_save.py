from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from src.pipeline.runner import _start_async_initial_population_save
from src.pipeline.steps import step_3_4_load_initial_population
from src.pipeline.types import Individual


@unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "fork start method required")
class RunnerAsyncSaveTest(unittest.TestCase):
    def test_async_initial_population_save_keeps_cache_format(self) -> None:
        population = [
            Individual(
                tau=[[1.0, 2.0]],
                rewards=[[3.0, 4.0]],
                rho=1.0,
                experience_buffers=[[[["edge_a", [1.0, 2.0]], 0, None]]],
                policies=[],
                metadata={"initial_index": 1},
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "initial_population.pkl"
            proc = _start_async_initial_population_save(population, population_path=str(cache_path))
            self.assertIsNotNone(proc)
            assert proc is not None
            proc.join(timeout=10)
            self.assertEqual(proc.exitcode, 0)

            restored = step_3_4_load_initial_population(str(cache_path))

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].tau, population[0].tau)
        self.assertEqual(restored[0].rewards, population[0].rewards)
        self.assertEqual(restored[0].rho, population[0].rho)
        self.assertEqual(restored[0].experience_buffers, [])
        self.assertEqual(restored[0].policies, [])
        self.assertEqual(restored[0].metadata, {})


if __name__ == "__main__":
    unittest.main()
