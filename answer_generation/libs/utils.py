"""Utility functions for Video Question Answering."""

import yaml
from pathlib import Path
from models import TransformersModel, VLLMModel
from .custom_logger import create_logger


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(
    name: str = "VQA",
    log_dir: str = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG"
):
    """
    Set up enhanced logging configuration using loguru.
    
    Args:
        name: Logger name/identifier
        log_dir: Directory to save log files (defaults to ./logs)
        console_level: Console logging level
        file_level: File logging level
    
    Returns:
        CustomLogger instance with enhanced capabilities
    """
    # Default log directory relative to project root
    if log_dir is None:
        # Try to find a reasonable default log directory
        current_dir = Path.cwd()
        log_dir = current_dir / "logs"
    
    return create_logger(
        name=name,
        log_dir=log_dir,
        console_level=console_level,
        file_level=file_level
    )


def create_model(config):
    """Create model instance based on configuration."""
    model_type = config.get("model_type", "transformers")

    if model_type == "transformers":
        return TransformersModel(config)
    elif model_type == "vllm":
        return VLLMModel(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
