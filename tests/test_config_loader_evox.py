from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config.loader import load_config


class EvoXConfigLoaderTest(unittest.TestCase):
    def test_loader_accepts_evox_config_without_legacy_modules(self) -> None:
        yaml_text = textwrap.dedent(
            """
            seed: 1

            algorithm:
              M: 4
              N: 3
              K: 2

            logging:
              best_rho_csv_path: "outputs/test_best.csv"

            q_provider:
              class_path: "src.q.sumo_ryl_nov_q:SumoRylQProvider"
              kwargs:
                num_car: 3

            environment:
              class_path: "src.environments.sumo_ryl_nov:SumoRylNovEnvironment"
              kwargs:
                sumo_config: "maps/testsumo.sumocfg"
                net_file: "maps/testsumo.net.xml"
                route_file: "maps/testsumo.rou.xml"

            evox:
              algorithm:
                class_path: "evox.algorithms:OpenES"
                kwargs:
                  pop_size: 8
                  learning_rate: 0.05
                  noise_stdev: 0.15
              reward_min: 0.0
              reward_max: 5.0
              repo_path: "/mnt/d/evox"
              init_strategy: "best"
              device: "cuda"

            metric:
              class_path: "src.metrics.sumo_ryl_metric:SumoRylMetric"
              kwargs: {}

            parallel_executor:
              class_path: "src.parallel.serial:SerialExecutor"
              kwargs: {}
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")
            cfg = load_config(config_path)

        self.assertIsNone(cfg.dqn_module)
        self.assertIsNone(cfg.diffusion_model)
        self.assertIsNone(cfg.elite_selector)
        self.assertEqual(cfg.evox.algorithm.class_path, "evox.algorithms:OpenES")
        self.assertEqual(cfg.evox.algorithm.kwargs["pop_size"], 8)
        self.assertEqual(cfg.evox.reward_min, 0.0)
        self.assertEqual(cfg.evox.reward_max, 5.0)
        self.assertEqual(cfg.evox.repo_path, "/mnt/d/evox")
        self.assertEqual(cfg.evox.init_strategy, "best")
        self.assertEqual(cfg.evox.device, "cuda")


if __name__ == "__main__":
    unittest.main()
