"""Results comparison tools for Video Question Answering evaluation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats


class ResultsComparator:
    """Compare results from different experiments."""

    def __init__(self):
        pass

    def compare_experiments(
        self,
        results1: List[Dict],
        results2: List[Dict],
        exp1_name: str = "Experiment 1",
        exp2_name: str = "Experiment 2"
    ) -> Dict:
        """Compare two sets of experimental results."""
        # Match results by file_stem
        matched_results = self._match_results(results1, results2)

        if not matched_results:
            return {"error": "No matching samples found between experiments"}

        comparison = {
            "experiment_names": [exp1_name, exp2_name],
            "matched_samples": len(matched_results),
            "metric_comparisons": self._compare_metrics(matched_results),
            "statistical_significance": self._test_significance(matched_results),
            "performance_changes": self._analyze_performance_changes(matched_results)
        }

        return comparison

    def _match_results(
        self, results1: List[Dict], results2: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """Match results from two experiments by file_stem."""
        results1_dict = {
            r.get('file_stem', r.get('metadata', {}).get('file_stem', '')): r
            for r in results1
        }
        results2_dict = {
            r.get('file_stem', r.get('metadata', {}).get('file_stem', '')): r
            for r in results2
        }

        matched = []
        for file_stem in results1_dict:
            if file_stem in results2_dict:
                matched.append((results1_dict[file_stem], results2_dict[file_stem]))

        return matched

    def _compare_metrics(self, matched_results: List[Tuple[Dict, Dict]]) -> Dict:
        """Compare metrics between matched results."""
        metrics1_list = []
        metrics2_list = []

        for r1, r2 in matched_results:
            m1 = r1.get('evaluation_metrics', {})
            m2 = r2.get('evaluation_metrics', {})

            if m1 and m2:
                metrics1_list.append(m1)
                metrics2_list.append(m2)

        if not metrics1_list or not metrics2_list:
            return {"error": "No metrics available for comparison"}

        df1 = pd.DataFrame(metrics1_list)
        df2 = pd.DataFrame(metrics2_list)

        comparisons = {}
        for metric in df1.columns:
            if metric in df2.columns:
                mean1 = df1[metric].mean()
                mean2 = df2[metric].mean()
                diff = mean2 - mean1
                percent_change = (diff / mean1 * 100) if mean1 != 0 else 0

                comparisons[metric] = {
                    "exp1_mean": float(mean1),
                    "exp2_mean": float(mean2),
                    "difference": float(diff),
                    "percent_change": float(percent_change),
                    "exp1_std": float(df1[metric].std()),
                    "exp2_std": float(df2[metric].std())
                }

        return comparisons

    def _test_significance(self, matched_results: List[Tuple[Dict, Dict]]) -> Dict:
        """Test statistical significance of differences."""
        metrics1_list = []
        metrics2_list = []

        for r1, r2 in matched_results:
            m1 = r1.get('evaluation_metrics', {})
            m2 = r2.get('evaluation_metrics', {})

            if m1 and m2:
                metrics1_list.append(m1)
                metrics2_list.append(m2)

        if len(metrics1_list) < 3:  # Need at least 3 samples for meaningful statistics
            return {"error": "Insufficient samples for significance testing"}

        df1 = pd.DataFrame(metrics1_list)
        df2 = pd.DataFrame(metrics2_list)

        significance_tests = {}
        for metric in df1.columns:
            if metric in df2.columns:
                try:
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(df2[metric], df1[metric])

                    # Wilcoxon signed-rank test (non-parametric alternative)
                    w_stat, w_p_value = stats.wilcoxon(df2[metric], df1[metric])

                    significance_tests[metric] = {
                        "paired_t_test": {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant_005": p_value < 0.05,
                            "significant_001": p_value < 0.01
                        },
                        "wilcoxon_test": {
                            "statistic": float(w_stat),
                            "p_value": float(w_p_value),
                            "significant_005": w_p_value < 0.05,
                            "significant_001": w_p_value < 0.01
                        }
                    }
                except Exception as e:
                    significance_tests[metric] = {"error": str(e)}

        return significance_tests

    def _analyze_performance_changes(self, matched_results: List[Tuple[Dict, Dict]]) -> Dict:
        """Analyze how performance changed between experiments."""
        improvements = 0
        degradations = 0
        no_change = 0

        metric_changes = defaultdict(list)

        for r1, r2 in matched_results:
            m1 = r1.get('evaluation_metrics', {})
            m2 = r2.get('evaluation_metrics', {})

            if m1 and m2:
                # Use STS score as primary metric for overall comparison
                if 'sts_score' in m1 and 'sts_score' in m2:
                    change = m2['sts_score'] - m1['sts_score']
                    if abs(change) < 0.001:  # Negligible change threshold
                        no_change += 1
                    elif change > 0:
                        improvements += 1
                    else:
                        degradations += 1

                # Track changes for all metrics
                for metric in m1:
                    if metric in m2:
                        change = m2[metric] - m1[metric]
                        metric_changes[metric].append(change)

        # Analyze metric changes
        metric_change_analysis = {}
        for metric, changes in metric_changes.items():
            if changes:
                changes_array = np.array(changes)
                metric_change_analysis[metric] = {
                    "mean_change": float(changes_array.mean()),
                    "std_change": float(changes_array.std()),
                    "improvements": int(np.sum(changes_array > 0.001)),
                    "degradations": int(np.sum(changes_array < -0.001)),
                    "no_change": int(np.sum(np.abs(changes_array) <= 0.001))
                }

        return {
            "overall_summary": {
                "improvements": improvements,
                "degradations": degradations,
                "no_change": no_change,
                "total_compared": len(matched_results)
            },
            "metric_changes": metric_change_analysis
        }
