import sys
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.main import run_pipeline
import yaml

if __name__ == "__main__":
    config_path = project_root / "configs" / "starcraft_smac_es.yaml"
    # 👇 关键修改：指定 UTF-8 编码
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    run_pipeline(config)