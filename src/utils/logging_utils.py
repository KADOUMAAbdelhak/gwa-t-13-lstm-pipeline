import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(config_path: str = "configs/logging.yaml"):
    """
    Configure logging either from the provided YAML file or fall back to a sane default.
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        # Ensure directories referenced by file handlers exist
        for handler in config.get("handlers", {}).values():
            filename = handler.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

