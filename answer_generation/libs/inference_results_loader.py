"""Inference results loading utilities for Video Question Answering evaluation."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


class InferenceResultsLoader:
    """Load and parse inference results from infer.py output."""

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self._validate_directory()

    def _validate_directory(self):
        """Validate that input directory has expected structure."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        expected_subdirs = ["individual_results", "summary", "configs"]
        missing_dirs = []

        for subdir in expected_subdirs:
            if not (self.input_dir / subdir).exists():
                missing_dirs.append(subdir)

        if missing_dirs:
            print(f"Warning: Missing expected subdirectories: {missing_dirs}")

    def load_individual_results(self) -> List[Dict]:
        """Load all individual result files."""
        individual_dir = self.input_dir / "individual_results"
        if not individual_dir.exists():
            print("Warning: individual_results directory not found")
            return []

        results = []
        json_files = list(individual_dir.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    result['file_stem'] = json_file.stem
                    results.append(result)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

        return results

    def load_summary_data(self) -> Dict:
        """Load summary data if available."""
        summary_dir = self.input_dir / "summary"
        summary_data = {}

        # Load detailed results
        detailed_file = summary_dir / "detailed_results.json"
        if detailed_file.exists():
            try:
                with open(detailed_file, 'r', encoding='utf-8') as f:
                    summary_data['detailed_results'] = json.load(f)
            except Exception as e:
                print(f"Error loading detailed results: {e}")

        # Load experiment statistics
        stats_file = summary_dir / "experiment_statistics.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    summary_data['experiment_statistics'] = json.load(f)
            except Exception as e:
                print(f"Error loading experiment statistics: {e}")

        # Load CSV summary
        csv_file = summary_dir / "summary_results.csv"
        if csv_file.exists():
            try:
                summary_data['summary_csv'] = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error loading summary CSV: {e}")

        return summary_data

    def load_config_info(self) -> Dict:
        """Load configuration information."""
        config_dir = self.input_dir / "configs"
        config_info = {}

        config_snapshot = config_dir / "config_snapshot.json"
        if config_snapshot.exists():
            try:
                with open(config_snapshot, 'r', encoding='utf-8') as f:
                    config_info['config_snapshot'] = json.load(f)
            except Exception as e:
                print(f"Error loading config snapshot: {e}")

        return config_info
