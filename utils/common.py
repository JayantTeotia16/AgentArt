"""
utils/common.py
Shared utilities used across all phases.
"""
import os
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_output_dir(cfg, phase_name):
    base = Path(cfg["paths"]["output_dir"])
    out = base / phase_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logger(name, log_file=None, level=logging.INFO):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


ARTEMIS_EMOTIONS = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness"
]

EMOTION_IDX = {e: i for i, e in enumerate(ARTEMIS_EMOTIONS)}
