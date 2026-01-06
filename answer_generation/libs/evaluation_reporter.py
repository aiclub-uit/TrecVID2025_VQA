"""Evaluation reporting tools for Video Question Answering."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available for plotting")
    PLOTTING_AVAILABLE = False


class EvaluationReporter:
    """Generate comprehensive evaluation reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "detailed_analysis").mkdir(exist_ok=True)

    def generate_report(self, analysis: Dict, input_dir: Path, results: List[Dict] = None):
        """Generate comprehensive evaluation report."""
        timestamp = datetime.now().isoformat()

        # Generate main analysis report
        self._save_analysis_json(analysis, timestamp)

        # Generate plots if matplotlib is available
        if PLOTTING_AVAILABLE and results:
            try:
                self._generate_plots(analysis, results)
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")

        # Generate markdown report
        self._generate_markdown_report(analysis, input_dir, timestamp)

        # Generate detailed CSV files
        if results:
            self._generate_detailed_csvs(results, analysis)

        print(f"Evaluation report generated in: {self.output_dir}")

    def _save_analysis_json(self, analysis: Dict, timestamp: str):
        """Save analysis results as JSON."""
        analysis_file = self.output_dir / "evaluation_analysis.json"

        analysis_with_meta = {
            "evaluation_metadata": {
                "timestamp": timestamp,
                "evaluator_version": "1.0.0"
            },
            "analysis": analysis
        }

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_with_meta, f, indent=2, ensure_ascii=False)

    def _generate_plots(self, analysis: Dict, results: List[Dict]):
        """Generate visualization plots."""
        if not PLOTTING_AVAILABLE:
            return

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Plot metric distributions
        metrics_data = []
        for result in results:
            if result.get('evaluation_metrics'):
                metrics_data.append(result['evaluation_metrics'])

        if metrics_data:
            self._plot_metric_distributions(metrics_data)
            self._plot_metric_correlations(metrics_data)

        # Plot timing analysis if available
        timing_data = []
        for result in results:
            if result.get('output', {}).get('inference_time'):
                timing_data.append(result['output']['inference_time'])

        if timing_data:
            self._plot_timing_analysis(timing_data)

    def _plot_metric_distributions(self, metrics_data: List[Dict]):
        """Plot distributions of evaluation metrics."""
        df = pd.DataFrame(metrics_data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            return

        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, column in enumerate(numeric_columns):
            if i < len(axes):
                axes[i].hist(df[column], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {column}')
                axes[i].set_xlabel(column)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "metric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metric_correlations(self, metrics_data: List[Dict]):
        """Plot correlation matrix of evaluation metrics."""
        df = pd.DataFrame(metrics_data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) < 2:
            return

        correlation_matrix = df[numeric_columns].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.3f')
        plt.title('Correlation Matrix of Evaluation Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "metric_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_timing_analysis(self, timing_data: List[float]):
        """Plot timing analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(timing_data, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Inference Times')
        ax1.set_xlabel('Inference Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(timing_data)
        ax2.set_title('Inference Time Box Plot')
        ax2.set_ylabel('Inference Time (seconds)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "timing_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_markdown_report(self, analysis: Dict, input_dir: Path, timestamp: str):
        """Generate markdown summary report."""
        report_file = self.output_dir / "evaluation_report.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Video Question Answering Evaluation Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Input Directory:** {input_dir}\n\n")

            # Basic statistics
            if 'basic_statistics' in analysis:
                stats = analysis['basic_statistics']
                f.write("## Basic Statistics\n\n")
                f.write(f"- **Total Samples:** {stats.get('total_samples', 'N/A')}\n")
                f.write(f"- **Successful Inferences:** {stats.get('successful_inferences', 'N/A')}\n")
                f.write(f"- **Success Rate:** {stats.get('success_rate', 0):.2%}\n\n")

                if stats.get('metrics_available') and 'metric_statistics' in stats:
                    f.write("### Metric Statistics\n\n")
                    for metric, stats_dict in stats['metric_statistics'].items():
                        f.write(f"**{metric}:**\n")
                        f.write(f"- Mean: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}\n")
                        f.write(f"- Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]\n")
                        f.write(f"- Median: {stats_dict['median']:.4f}\n\n")

            # Performance by categories
            if 'performance_by_categories' in analysis and analysis['performance_by_categories']:
                f.write("## Performance by Categories\n\n")
                for category_name, category_data in analysis['performance_by_categories'].items():
                    f.write(f"### {category_name.replace('_', ' ').title()}\n\n")
                    for category, stats in category_data.items():
                        f.write(f"**{category}** ({stats['count']} samples):\n")
                        if 'metrics' in stats:
                            for metric, metric_stats in stats['metrics'].items():
                                f.write(f"- {metric}: {metric_stats['mean']:.4f} ± {metric_stats['std']:.4f}\n")
                        f.write("\n")

            # Error analysis
            if 'error_analysis' in analysis:
                error_data = analysis['error_analysis']
                f.write("## Error Analysis\n\n")
                f.write(f"- **Failed Inferences:** {len(error_data.get('failed_inferences', []))}\n")
                f.write(f"- **Low Performance Samples:** {len(error_data.get('low_performance_samples', []))}\n\n")

            # Timing analysis
            if 'timing_analysis' in analysis and 'error' not in analysis['timing_analysis']:
                timing = analysis['timing_analysis']
                f.write("## Timing Analysis\n\n")
                f.write(f"- **Total Inference Time:** {timing.get('total_time', 0):.2f} seconds\n")
                f.write(f"- **Average Time per Sample:** {timing.get('mean_time', 0):.2f} seconds\n")
                f.write(f"- **Fastest Inference:** {timing.get('min_time', 0):.2f} seconds\n")
                f.write(f"- **Slowest Inference:** {timing.get('max_time', 0):.2f} seconds\n\n")

            f.write("## Files Generated\n\n")
            f.write("- `evaluation_analysis.json`: Complete analysis results\n")
            f.write("- `plots/`: Visualization plots\n")
            f.write("- `detailed_analysis/`: Detailed CSV files\n")

    def _generate_detailed_csvs(self, results: List[Dict], analysis: Dict):
        """Generate detailed CSV files for further analysis."""
        detailed_dir = self.output_dir / "detailed_analysis"

        # Extract detailed results
        detailed_results = []
        for result in results:
            if result.get('evaluation_metrics'):
                row = {
                    'file_stem': result.get('file_stem', ''),
                    'question': result.get('input', {}).get('question', ''),
                    'correct_answer': result.get('ground_truth', {}).get('correct_answer', ''),
                    'predicted_answer': result.get('output', {}).get('predicted_answer', ''),
                    'inference_time': result.get('output', {}).get('inference_time', 0),
                    **result['evaluation_metrics']
                }
                detailed_results.append(row)

        if detailed_results:
            df = pd.DataFrame(detailed_results)
            df.to_csv(detailed_dir / "detailed_results.csv", index=False)

        # Save error analysis if available
        if 'error_analysis' in analysis:
            error_data = analysis['error_analysis']

            # Low performance samples
            if error_data.get('low_performance_samples'):
                low_perf_df = pd.DataFrame(error_data['low_performance_samples'])
                low_perf_df.to_csv(detailed_dir / "low_performance_samples.csv", index=False)
