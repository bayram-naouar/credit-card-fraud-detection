from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CONFIG_FILE_PATH = ROOT / "config.yaml"
with open(CONFIG_FILE_PATH) as f:
    config = yaml.safe_load(f)
CONFIG = config
PATHS = config["paths"]
PREPROCESSING = config["preprocessing"]
MODELS = config["models"]
TRAINING = config["training"]
PARAM_SPACE = config["param_space"]
