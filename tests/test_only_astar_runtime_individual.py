from __future__ import annotations

import unittest

from src.pipeline.steps import step_4_build_only_astar_runtime_individual


class OnlyAstarRuntimeIndividualTest(unittest.TestCase):
    def test_runtime_individual_keeps_only_required_fields(self) -> None:
        ind = step_4_build_only_astar_runtime_individual(
            tau="tau",
            rewards="rewards",
            rho=1.23,
            simulation_data="sim",
        )

        self.assertEqual(ind.tau, "tau")
        self.assertEqual(ind.rewards, "rewards")
        self.assertEqual(ind.rho, 1.23)
        self.assertEqual(ind.experience_buffers, [])
        self.assertEqual(ind.policies, [])
        self.assertEqual(ind.metadata, {"simulation_data": "sim"})


if __name__ == "__main__":
    unittest.main()
