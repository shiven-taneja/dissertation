from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def load_cfg(cfg_path: Path) -> Dict[str, Any]:
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def merge_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override YAML entries with non-None CLI flags.
    Only keys actually present in cfg are overridden.
    """
    mapping = {
        "ticker": ("data", "ticker"),
        "epochs": ("agent", "epochs"),
        "seed": ("agent", "seed"),
    }
    for cli_key, path in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            section, key = path
            cfg[section][key] = val
    return cfg
