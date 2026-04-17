"""Config loading and SHA256 hashing utility."""

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(config: dict) -> str:
    """Compute SHA256 hash of a config dict for reproducibility tracking."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
