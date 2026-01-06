"""Results saving and output handling utilities for Video Question Answering."""

import json
import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from .data_loader import get_video_info


def create_output_directory(output_path: str) -> Path:
    """Create output directory structure for storing inference results."""
    timestamp = datetime.now().strftime("%m%d_%H%M")

    if not output_path:
        output_path = f"inference_output_{timestamp}"

    # Ensure output_path is a directory
    if not output_path.endswith("/"):
        output_path = f"{output_path}_{timestamp}"

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "individual_results").mkdir(exist_ok=True)
    (output_dir / "summary").mkdir(exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)

    return output_dir


def save_individual_result(
    result: Dict, config: Dict, output_dir: Path, model_info: Dict = None
) -> None:
    """Save individual video inference result with complete reproducibility info."""
    file_stem = result["file_stem"]
    individual_file = output_dir / "individual_results" / f"{file_stem}.json"

    # Create comprehensive result data
    individual_data = {
        "metadata": {
            "inference_timestamp": datetime.now().isoformat(),
            "file_stem": file_stem,
            "video_path": result.get("video_path", ""),
            "video_url": result.get("video_url", ""),
            "video_exists": (
                os.path.exists(result.get("video_path", ""))
                if result.get("video_path")
                else False
            ),
        },
        "input": {
            "question": result["question"],
            "video_duration_info": (
                get_video_info(result.get("video_path"))
                if result.get("video_path")
                else None
            ),
        },
        "output": {
            "predicted_answer": result["predicted_answer"],
            "inference_time": result.get("inference_time", 0),
            "all_answers": result.get("all_answers", []),  # Include all generated answers
        },
        "ground_truth": {
            "correct_answer": result["correct_answer"],
            "incorrect_answers": result.get("incorrect_answers", []),
        },
        "evaluation_metrics": result.get("metrics", {}),
        "all_evaluation_metrics": result.get("all_metrics", []),  # Include metrics for all answers
        "model_configuration": {
            "model_type": config.get("model_type", "unknown"),
            "model_name": config.get("model_name", "unknown"),
            "model_parameters": {
                "num_frames": config.get("num_frames", 16),
                "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.8),
                "max_model_len": config.get("max_model_len", 38400),
                "cache_dir": config.get("cache_dir", "caches"),
            },
            "sampling_parameters": {
                "max_tokens": 1024,
                "temperature": 0.1,
                "top_p": 0.95,
                "stop_tokens": ["<|im_end|>", "\n\n", "---"],
            },
        },
        "system_info": {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", "not_set"
                ),
                "HF_HOME": os.environ.get("HF_HOME", "not_set"),
            },
        },
        "reproducibility": {
            "config_file_used": result.get("config_file_path", ""),
            "command_line_args": result.get("command_line_args", {}),
            "random_seed": config.get("random_seed", None),
        },
    }

    # Add model-specific information if available
    if model_info:
        individual_data["model_configuration"]["model_info"] = model_info

    # Save the individual result
    with open(individual_file, "w", encoding="utf-8") as f:
        json.dump(individual_data, f, indent=2, ensure_ascii=False)


def save_config_snapshot(config: Dict, config_file_path: str, output_dir: Path) -> None:
    """Save a snapshot of the configuration used for the inference."""
    config_snapshot = {
        "original_config_file": config_file_path,
        "config_content": config,
        "snapshot_timestamp": datetime.now().isoformat(),
    }

    config_file = output_dir / "configs" / "config_snapshot.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # Also copy the original config file if it exists
    if os.path.exists(config_file_path):
        original_config_copy = (
            output_dir / "configs" / f"original_{os.path.basename(config_file_path)}"
        )
        shutil.copy2(config_file_path, original_config_copy)


def save_prompt_template(config: Dict, output_dir: Path, model=None) -> None:
    """Save the prompt template used for inference to PROMPT.txt file."""
    try:
        # Try to get the prompt template from the model if available
        prompt_template = None
        
        if model and hasattr(model, 'prompt_handler'):
            # Get the actual prompt template being used
            if hasattr(model.prompt_handler, 'config_manager') and model.prompt_handler.config_manager:
                # Using YAML-based prompt configuration
                template_name = model.prompt_handler.template_name
                try:
                    template_info = model.prompt_handler.config_manager.get_template_info(template_name)
                    prompt_template = template_info.get('template', '')
                except Exception as e:
                    print(f"Warning: Could not get YAML template info: {e}")
            
            # If YAML config not available, try to get from prompt_builder
            if not prompt_template:
                try:
                    # Import the prompt templates from prompt_builder
                    from prompt_builder import VISUAL_ONLY_PROMPT, AUDIO_VISUAL_PROMPT
                    
                    # Determine which template to use based on the model's preference
                    if hasattr(model.prompt_handler, 'prefer_audio_visual') and model.prompt_handler.prefer_audio_visual:
                        prompt_template = AUDIO_VISUAL_PROMPT
                    else:
                        prompt_template = VISUAL_ONLY_PROMPT
                except ImportError:
                    print("Warning: Could not import prompt templates from prompt_builder")
        
        # If we still don't have a template, try to get it from config
        if not prompt_template:
            prompt_template = config.get("prompt", "Please answer the question about this video.")
            # If it's just a simple string, create a basic template format
            if prompt_template and "{question}" not in prompt_template:
                prompt_template = "Analyze the video content and audio transcription, then respond with only the answer, concisely and without restating the question.\n\n{question}"
        
        # Save the prompt template to PROMPT.txt
        prompt_file = output_dir / "PROMPT.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_template)
        
        print(f"✓ Prompt template saved to: {prompt_file}")
        
    except Exception as e:
        print(f"Warning: Could not save prompt template: {e}")
        # Create a fallback PROMPT.txt with basic information
        try:
            prompt_file = output_dir / "PROMPT.txt"
            fallback_content = f"# Prompt Template\n\nTemplate: {config.get('prompt_template', 'unknown')}\nFallback prompt: {config.get('prompt', 'Please answer the question about this video.')}\n\nError occurred while extracting detailed template: {str(e)}"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(fallback_content)
            print(f"✓ Fallback prompt info saved to: {prompt_file}")
        except Exception as fallback_error:
            print(f"Error: Could not save even fallback prompt info: {fallback_error}")


def save_summary_results(
    results: List[Dict], config: Dict, output_dir: Path, args
) -> None:
    """Save summary results and statistics."""
    summary_dir = output_dir / "summary"

    # Save detailed results as JSON
    detailed_json = summary_dir / "detailed_results.json"
    with open(detailed_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create summary CSV
    csv_data = []
    for result in results:
        row = {
            "file_stem": result["file_stem"],
            "question": result["question"],
            "correct_answer": result["correct_answer"],
            "predicted_answer": result["predicted_answer"],
            "inference_time": result.get("inference_time", 0),
            **result.get("metrics", {}),
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    summary_csv = summary_dir / "summary_results.csv"
    df.to_csv(summary_csv, index=False)

    # Generate statistics summary
    stats_summary = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "successful_inferences": len(
                [r for r in results if r.get("predicted_answer")]
            ),
            "model_type": config.get("model_type", "unknown"),
            "model_name": config.get("model_name", "unknown"),
            "command_line_args": vars(args) if args else {},
        },
        "performance_metrics": {},
        "timing_info": {
            "total_inference_time": sum(r.get("inference_time", 0) for r in results),
            "average_inference_time": (
                np.mean([r.get("inference_time", 0) for r in results]) if results else 0
            ),
        },
    }

    # Calculate performance statistics if metrics are available
    if results and results[0].get("metrics"):
        metrics_df = pd.DataFrame([r["metrics"] for r in results])
        stats_summary["performance_metrics"] = {
            "sts_score": {
                "mean": float(metrics_df["sts_score"].mean()),
                "std": float(metrics_df["sts_score"].std()),
                "min": float(metrics_df["sts_score"].min()),
                "max": float(metrics_df["sts_score"].max()),
            },
            "meteor_score": {
                "mean": float(metrics_df["meteor_score"].mean()),
                "std": float(metrics_df["meteor_score"].std()),
                "min": float(metrics_df["meteor_score"].min()),
                "max": float(metrics_df["meteor_score"].max()),
            },
            "bert_f1": {
                "mean": float(metrics_df["bert_f1"].mean()),
                "std": float(metrics_df["bert_f1"].std()),
                "min": float(metrics_df["bert_f1"].min()),
                "max": float(metrics_df["bert_f1"].max()),
            },
            "ndcg_score": {
                "mean": float(metrics_df["ndcg_score"].mean()),
                "std": float(metrics_df["ndcg_score"].std()),
                "min": float(metrics_df["ndcg_score"].min()),
                "max": float(metrics_df["ndcg_score"].max()),
            },
        }

    stats_file = summary_dir / "experiment_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)

    # Create a README file for the experiment
    readme_content = f"""# Video Question Answering Inference Results

## Experiment Overview
- **Timestamp**: {datetime.now().isoformat()}
- **Model Type**: {config.get("model_type", "unknown")}
- **Model Name**: {config.get("model_name", "unknown")}
- **Total Samples**: {len(results)}
- **Successful Inferences**: {len([r for r in results if r.get("predicted_answer")])}

## Directory Structure
{output_dir.name}/
├── individual_results/      # Individual JSON files for each video
├── summary/                 # Summary statistics and aggregated results
│   ├── detailed_results.json
│   ├── summary_results.csv
│   ├── experiment_statistics.json
│   └── README.md (this file)
├── configs/                 # Configuration snapshots
│   ├── config_snapshot.json
│   └── original_*.yaml


## Files Description
- **individual_results/**: Contains one JSON file per video with complete inference details, metrics, and reproducibility information
- **summary/detailed_results.json**: All results in a single JSON file
- **summary/summary_results.csv**: CSV format for easy analysis in spreadsheet applications
- **summary/experiment_statistics.json**: Aggregated statistics and performance metrics
- **configs/**: Configuration files used for this experiment

## Reproducing Results
To reproduce these results:
1. Use the configuration file in `configs/original_*.yaml`
2. Ensure the same model and parameters as specified in the config snapshot
3. Run inference on the same video files as listed in the individual result files

## Performance Summary
"""

    if stats_summary.get("performance_metrics"):
        readme_content += f"""
- **Average STS Score**: {stats_summary['performance_metrics']['sts_score']['mean']:.4f} ± {stats_summary['performance_metrics']['sts_score']['std']:.4f}
- **Average METEOR Score**: {stats_summary['performance_metrics']['meteor_score']['mean']:.4f} ± {stats_summary['performance_metrics']['meteor_score']['std']:.4f}
- **Average BERTScore F1**: {stats_summary['performance_metrics']['bert_f1']['mean']:.4f} ± {stats_summary['performance_metrics']['bert_f1']['std']:.4f}
- **Average NDCG Score**: {stats_summary['performance_metrics']['ndcg_score']['mean']:.4f} ± {stats_summary['performance_metrics']['ndcg_score']['std']:.4f}
"""

    readme_content += f"""
## Timing Information
- **Total Inference Time**: {stats_summary['timing_info']['total_inference_time']:.2f} seconds
- **Average Inference Time per Video**: {stats_summary['timing_info']['average_inference_time']:.2f} seconds
"""

    readme_file = summary_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)
