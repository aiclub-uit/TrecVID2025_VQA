"""Simplified utilities for multiple choice VQA."""

import yaml
import json
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_prepared_data(data_dir):
    """Load prepared data from directory."""
    data = []
    data_path = Path(data_dir)
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                sample = json.load(f)
                data.append(sample)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return data
