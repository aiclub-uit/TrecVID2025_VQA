"""Analysis tools for Video Question Answering evaluation results."""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy import stats

from .evaluation_metrics import EvaluationMetrics


class EvaluationAnalyzer:
    """Comprehensive analysis of inference results."""

    def __init__(self, evaluator: EvaluationMetrics):
        self.evaluator = evaluator

    def analyze_results(self, results: List[Dict], recompute_metrics: bool = False) -> Dict:
        """Perform comprehensive analysis on results."""
        recomputed_data = None
        if recompute_metrics:
            recomputed_data = self._recompute_metrics(results)
        
        analysis = {
            "basic_statistics": self._compute_basic_statistics(results, recomputed_data),
            "metric_distributions": self._analyze_metric_distributions(results),
            "correlation_analysis": self._compute_correlations(results),
            "performance_by_categories": self._analyze_by_categories(results),
            "error_analysis": self._analyze_errors(results),
            "timing_analysis": self._analyze_timing(results)
        }

        if recompute_metrics:
            analysis["recomputed_metrics"] = recomputed_data

        return analysis

    def _compute_basic_statistics(self, results: List[Dict], recomputed_data: List[Dict] = None) -> Dict:
        """Compute basic statistics about the results."""
        total_samples = len(results)
        successful_inferences = len([r for r in results if self._has_valid_prediction(r)])

        # Use recomputed metrics if available, otherwise use existing metrics
        if recomputed_data:
            print("Using recomputed metrics for basic statistics")
            metrics_data = [item['recomputed_metrics'] for item in recomputed_data]
        else:
            # Extract metrics if available
            metrics_data = []
            for result in results:
                if self._has_evaluation_metrics(result):
                    metrics_data.append(result['evaluation_metrics'])

        if not metrics_data:
            return {
                "total_samples": total_samples,
                "successful_inferences": successful_inferences,
                "success_rate": successful_inferences / total_samples if total_samples > 0 else 0,
                "metrics_available": False
            }

        metrics_df = pd.DataFrame(metrics_data)

        return {
            "total_samples": total_samples,
            "successful_inferences": successful_inferences,
            "success_rate": successful_inferences / total_samples if total_samples > 0 else 0,
            "metrics_available": True,
            "metric_statistics": {
                metric: {
                    "mean": float(metrics_df[metric].mean()),
                    "std": float(metrics_df[metric].std()),
                    "min": float(metrics_df[metric].min()),
                    "max": float(metrics_df[metric].max()),
                    "median": float(metrics_df[metric].median()),
                    "q25": float(metrics_df[metric].quantile(0.25)),
                    "q75": float(metrics_df[metric].quantile(0.75))
                } for metric in metrics_df.columns if metrics_df[metric].dtype in ['float64', 'int64']
            }
        }

    def _analyze_metric_distributions(self, results: List[Dict]) -> Dict:
        """Analyze the distribution of metrics."""
        metrics_data = []
        for result in results:
            if self._has_evaluation_metrics(result):
                metrics_data.append(result['evaluation_metrics'])

        if not metrics_data:
            return {"error": "No evaluation metrics found"}

        metrics_df = pd.DataFrame(metrics_data)
        distributions = {}

        for metric in metrics_df.columns:
            if metrics_df[metric].dtype in ['float64', 'int64']:
                distributions[metric] = {
                    "histogram_bins": 20,
                    "histogram_counts": np.histogram(metrics_df[metric], bins=20)[0].tolist(),
                    "histogram_edges": np.histogram(metrics_df[metric], bins=20)[1].tolist(),
                    "skewness": float(stats.skew(metrics_df[metric])),
                    "kurtosis": float(stats.kurtosis(metrics_df[metric]))
                }

        return distributions

    def _compute_correlations(self, results: List[Dict]) -> Dict:
        """Compute correlations between different metrics."""
        metrics_data = []
        for result in results:
            if self._has_evaluation_metrics(result) and self._has_valid_prediction(result):
                metrics = result['evaluation_metrics'].copy()

                # Add timing information if available
                if 'output' in result and 'inference_time' in result['output']:
                    metrics['inference_time'] = result['output']['inference_time']

                # Add length metrics
                if 'output' in result and 'ground_truth' in result:
                    pred = result['output'].get('predicted_answer', '')
                    ref = result['ground_truth'].get('correct_answer', '')
                    length_metrics = self.evaluator.compute_length_metrics(pred, ref)
                    metrics.update(length_metrics)

                metrics_data.append(metrics)

        if len(metrics_data) < 2:
            return {"error": "Insufficient data for correlation analysis"}

        metrics_df = pd.DataFrame(metrics_data)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {"error": "Insufficient numeric metrics for correlation analysis"}

        correlation_matrix = metrics_df[numeric_cols].corr()

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(correlation_matrix)
        }

    def _find_strong_correlations(self, corr_matrix, threshold=0.7):
        """Find strongly correlated metric pairs."""
        strong_corrs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    strong_corrs.append({
                        "metric1": corr_matrix.columns[i],
                        "metric2": corr_matrix.columns[j],
                        "correlation": float(corr_val)
                    })

        return strong_corrs

    def _analyze_by_categories(self, results: List[Dict]) -> Dict:
        """Analyze performance by different categories (if metadata is available)."""
        # This could be extended to analyze by video duration, question type, etc.
        categories_analysis = {}

        # Analyze by question length
        if self._has_questions(results):
            question_lengths = []
            metrics_by_q_length = []

            for result in results:
                if self._has_evaluation_metrics(result) and 'input' in result:
                    question = result['input'].get('question', '')
                    q_length = len(question.split())
                    question_lengths.append(q_length)
                    metrics_by_q_length.append(result['evaluation_metrics'])

            if question_lengths and metrics_by_q_length:
                # Categorize by question length
                q_length_categories = self._categorize_by_quantiles(question_lengths, ["Short", "Medium", "Long"])
                categories_analysis['question_length'] = self._analyze_metrics_by_category(
                    metrics_by_q_length, q_length_categories
                )

        # Analyze by answer length
        if self._has_answers(results):
            answer_lengths = []
            metrics_by_a_length = []

            for result in results:
                if self._has_evaluation_metrics(result) and 'ground_truth' in result:
                    answer = result['ground_truth'].get('correct_answer', '')
                    a_length = len(answer.split())
                    answer_lengths.append(a_length)
                    metrics_by_a_length.append(result['evaluation_metrics'])

            if answer_lengths and metrics_by_a_length:
                a_length_categories = self._categorize_by_quantiles(answer_lengths, ["Short", "Medium", "Long"])
                categories_analysis['answer_length'] = self._analyze_metrics_by_category(
                    metrics_by_a_length, a_length_categories
                )

        return categories_analysis

    def _categorize_by_quantiles(self, values: List[float], labels: List[str]) -> List[str]:
        """Categorize values by quantiles."""
        if len(labels) != 3:
            raise ValueError("Currently only supports 3 categories")

        q33, q67 = np.percentile(values, [33, 67])
        categories = []

        for value in values:
            if value <= q33:
                categories.append(labels[0])
            elif value <= q67:
                categories.append(labels[1])
            else:
                categories.append(labels[2])

        return categories

    def _analyze_metrics_by_category(self, metrics_list: List[Dict], categories: List[str]) -> Dict:
        """Analyze metrics grouped by categories."""
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df['category'] = categories

        category_analysis = {}
        for category in set(categories):
            category_data = metrics_df[metrics_df['category'] == category]
            category_analysis[category] = {
                "count": len(category_data),
                "metrics": {
                    metric: {
                        "mean": float(category_data[metric].mean()),
                        "std": float(category_data[metric].std()),
                        "min": float(category_data[metric].min()),
                        "max": float(category_data[metric].max())
                    } for metric in category_data.columns if category_data[metric].dtype in ['float64', 'int64'] and metric != 'category'
                }
            }

        return category_analysis

    def _analyze_errors(self, results: List[Dict]) -> Dict:
        """Analyze common error patterns and failure cases."""
        error_analysis = {
            "failed_inferences": [],
            "low_performance_samples": [],
            "common_patterns": {}
        }

        # Find failed inferences
        for result in results:
            if not self._has_valid_prediction(result):
                error_analysis["failed_inferences"].append({
                    "file_stem": result.get('file_stem', 'unknown'),
                    "error_info": "No valid prediction generated"
                })

        # Find low performance samples (bottom 10% in main metrics)
        metrics_data = []
        for result in results:
            if self._has_evaluation_metrics(result):
                metrics = result['evaluation_metrics']
                metrics['file_stem'] = result.get('file_stem', 'unknown')
                metrics_data.append(metrics)

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)

            # Find samples in bottom 10% for key metrics
            key_metrics = ['sts_score', 'meteor_score', 'bert_f1', 'ndcg_score']
            for metric in key_metrics:
                if metric in metrics_df.columns:
                    threshold = metrics_df[metric].quantile(0.1)
                    low_perf = metrics_df[metrics_df[metric] <= threshold]

                    error_analysis["low_performance_samples"].extend([
                        {
                            "file_stem": row['file_stem'],
                            "metric": metric,
                            "value": row[metric],
                            "threshold": threshold
                        } for _, row in low_perf.iterrows()
                    ])

        return error_analysis

    def _analyze_timing(self, results: List[Dict]) -> Dict:
        """Analyze inference timing patterns."""
        timing_data = []

        for result in results:
            if 'output' in result and 'inference_time' in result['output']:
                timing_data.append(result['output']['inference_time'])

        if not timing_data:
            return {"error": "No timing data available"}

        timing_array = np.array(timing_data)

        return {
            "total_samples": len(timing_data),
            "total_time": float(timing_array.sum()),
            "mean_time": float(timing_array.mean()),
            "std_time": float(timing_array.std()),
            "min_time": float(timing_array.min()),
            "max_time": float(timing_array.max()),
            "median_time": float(np.median(timing_array)),
            "q95_time": float(np.percentile(timing_array, 95)),
            "outlier_threshold": float(np.percentile(timing_array, 95)),
            "outliers": len(timing_array[timing_array > np.percentile(timing_array, 95)])
        }

    def _recompute_metrics(self, results: List[Dict]) -> Dict:
        """Recompute metrics with current evaluation setup."""
        print("Recomputing metrics...")

        recomputed_results = []
        for i, result in enumerate(results):
            if not self._has_valid_prediction(result) or not self._has_ground_truth(result):
                continue

            if i % 10 == 0:
                print(f"Recomputing metrics for sample {i + 1}/{len(results)}")

            predicted = result['output'].get('predicted_answer', '')
            correct_answer = result['ground_truth'].get('correct_answer', '')
            incorrect_answers = result['ground_truth'].get('incorrect_answers', [])

            # Recompute all metrics
            new_metrics = {
                "sts_score": self.evaluator.compute_sts(predicted, correct_answer),
                "meteor_score": self.evaluator.compute_meteor(predicted, correct_answer),
                **self.evaluator.compute_bertscore(predicted, correct_answer),
                "ndcg_score": self.evaluator.compute_ndcg(predicted, correct_answer, incorrect_answers),
                **self.evaluator.compute_rouge(predicted, correct_answer),
                **self.evaluator.compute_readability(predicted),
                **self.evaluator.compute_length_metrics(predicted, correct_answer)
            }

            recomputed_results.append({
                "file_stem": result.get('file_stem', 'unknown'),
                "original_metrics": result.get('evaluation_metrics', {}),
                "recomputed_metrics": new_metrics
            })

        return recomputed_results

    # Helper methods
    def _has_valid_prediction(self, result: Dict) -> bool:
        """Check if result has a valid prediction."""
        return (
            'output' in result
            and 'predicted_answer' in result['output']
            and result['output']['predicted_answer']
        )

    def _has_evaluation_metrics(self, result: Dict) -> bool:
        """Check if result has evaluation metrics."""
        return 'evaluation_metrics' in result and result['evaluation_metrics']

    def _has_ground_truth(self, result: Dict) -> bool:
        """Check if result has ground truth data."""
        return (
            'ground_truth' in result
            and 'correct_answer' in result['ground_truth']
        )

    def _has_questions(self, results: List[Dict]) -> bool:
        """Check if results have question data."""
        return any('input' in r and 'question' in r['input'] for r in results)

    def _has_answers(self, results: List[Dict]) -> bool:
        """Check if results have answer data."""
        return any(self._has_ground_truth(r) for r in results)
