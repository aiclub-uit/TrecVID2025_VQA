#!/usr/bin/env python3
"""
Comprehensive evaluation script for Multiple Choice Video QA results.
Supports evaluation of single folders or batch evaluation of multiple result folders.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))


def load_result_files(result_dir: Path) -> Tuple[Dict, List[Dict]]:
    """Load all result files from a directory."""
    summary_file = result_dir / "summary.json"
    results = []
    summary = {}
    
    # Load summary if exists
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                if "results" in summary:
                    results = summary["results"]
        except Exception as e:
            print(f"Warning: Could not load summary file {summary_file}: {e}")
    
    # If no results in summary, load individual files
    if not results:
        print(f"Loading individual result files from {result_dir}")
        for json_file in result_dir.glob("*.json"):
            if json_file.name == "summary.json":
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
    
    return summary, results


def load_multiple_result_folders(result_paths: List[str]) -> Dict[str, List[Dict]]:
    """Load results from multiple folders for ensemble evaluation."""
    all_results = {}
    
    for path_str in result_paths:
        result_path = Path(path_str)
        if not result_path.exists():
            print(f"Warning: Path does not exist: {path_str}")
            continue
            
        folder_name = result_path.name
        print(f"Loading results from: {folder_name}")
        
        summary, results = load_result_files(result_path)
        if results:
            all_results[folder_name] = results
            print(f"  Loaded {len(results)} samples")
        else:
            print(f"  No results found in {path_str}")
    
    return all_results


def aggregate_rankings_by_voting(rankings_list: List[List[str]], method: str = "borda") -> List[str]:
    """Aggregate multiple rankings using voting methods."""
    if not rankings_list or not rankings_list[0]:
        return []
    
    # Get all unique options
    all_options = set()
    for ranking in rankings_list:
        all_options.update(ranking)
    all_options = sorted(list(all_options))
    
    if method == "borda":
        # Borda count: assign points based on position (higher position = more points)
        option_scores = defaultdict(float)
        
        for ranking in rankings_list:
            n_options = len(ranking)
            for i, option in enumerate(ranking):
                # Points: n_options - position (1-indexed)
                points = n_options - i
                option_scores[option] += points
        
        # Sort by score (descending)
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        return [option for option, _ in sorted_options]
    
    elif method == "plurality":
        # Plurality voting: count first-place votes
        first_place_votes = Counter()
        
        for ranking in rankings_list:
            if ranking:
                first_place_votes[ranking[0]] += 1
        
        # Sort by vote count, then by original option order for ties
        sorted_options = sorted(first_place_votes.items(), key=lambda x: (-x[1], all_options.index(x[0])))
        voted_options = [option for option, _ in sorted_options]
        
        # Add remaining options
        for option in all_options:
            if option not in voted_options:
                voted_options.append(option)
                
        return voted_options
    
    elif method == "approval":
        # Approval voting: count top-2 positions
        approval_votes = Counter()
        
        for ranking in rankings_list:
            # Consider top-2 as "approved"
            top_k = min(2, len(ranking))
            for i in range(top_k):
                approval_votes[ranking[i]] += 1
        
        # Sort by approval count
        sorted_options = sorted(approval_votes.items(), key=lambda x: (-x[1], all_options.index(x[0])))
        voted_options = [option for option, _ in sorted_options]
        
        # Add remaining options
        for option in all_options:
            if option not in voted_options:
                voted_options.append(option)
                
        return voted_options
    
    elif method == "condorcet":
        # Simplified Condorcet: pairwise comparisons
        option_wins = defaultdict(int)
        
        for opt1 in all_options:
            for opt2 in all_options:
                if opt1 == opt2:
                    continue
                
                opt1_wins = 0
                total_comparisons = 0
                
                for ranking in rankings_list:
                    if opt1 in ranking and opt2 in ranking:
                        total_comparisons += 1
                        if ranking.index(opt1) < ranking.index(opt2):
                            opt1_wins += 1
                
                if total_comparisons > 0 and opt1_wins > total_comparisons / 2:
                    option_wins[opt1] += 1
        
        # Sort by number of pairwise wins
        sorted_options = sorted(option_wins.items(), key=lambda x: (-x[1], all_options.index(x[0])))
        voted_options = [option for option, _ in sorted_options]
        
        # Add remaining options
        for option in all_options:
            if option not in voted_options:
                voted_options.append(option)
                
        return voted_options
    
    else:
        raise ValueError(f"Unknown voting method: {method}")


def create_ensemble_results(all_results: Dict[str, List[Dict]], voting_method: str = "borda") -> List[Dict]:
    """Create ensemble results by aggregating predictions from multiple models."""
    print(f"Creating ensemble using {voting_method} voting...")
    
    # Group results by video_id
    samples_by_video = defaultdict(list)
    
    for folder_name, results in all_results.items():
        for result in results:
            video_id = result.get('video_id', result.get('sample_id', 'unknown'))
            samples_by_video[video_id].append((folder_name, result))
    
    ensemble_results = []
    successful_ensembles = 0
    
    for video_id, model_results in samples_by_video.items():
        if len(model_results) < 2:
            print(f"Warning: Only {len(model_results)} model(s) for video {video_id}, skipping ensemble")
            continue
        
        # Extract rankings from each model
        rankings = []
        model_names = []
        base_result = None
        
        for folder_name, result in model_results:
            evaluation = result.get('evaluation', {})
            if evaluation and 'ranked_options' in evaluation:
                ranked_options = evaluation['ranked_options']
                if ranked_options:
                    rankings.append(ranked_options)
                    model_names.append(folder_name)
                    if base_result is None:
                        base_result = result.copy()
        
        if len(rankings) < 2:
            print(f"Warning: Less than 2 valid rankings for video {video_id}, skipping")
            continue
        
        # Aggregate rankings using voting
        try:
            ensemble_ranking = aggregate_rankings_by_voting(rankings, voting_method)
            
            if not ensemble_ranking:
                print(f"Warning: Empty ensemble ranking for video {video_id}")
                continue
            
            # Convert back to letter ranking
            options = base_result.get('options', [])
            correct_answer = base_result.get('correct_answer', '')
            
            if not options:
                print(f"Warning: No options found for video {video_id}")
                continue
            
            # Map ensemble ranking back to letters
            ensemble_letters = []
            for ranked_option in ensemble_ranking:
                if ranked_option in options:
                    letter_idx = options.index(ranked_option)
                    if letter_idx < 4:  # A, B, C, D
                        ensemble_letters.append(chr(ord('A') + letter_idx))
            
            # Find correct letter
            correct_letter = None
            if correct_answer in options:
                correct_idx = options.index(correct_answer)
                if correct_idx < 4:
                    correct_letter = chr(ord('A') + correct_idx)
            
            # Calculate metrics
            top1_correct = (ensemble_letters[0] == correct_letter) if ensemble_letters and correct_letter else False
            
            reciprocal_rank = 0.0
            if correct_letter and correct_letter in ensemble_letters:
                rank = ensemble_letters.index(correct_letter) + 1
                reciprocal_rank = 1.0 / rank
            
            # Create ensemble evaluation
            ensemble_evaluation = {
                "predicted_ranking": ensemble_letters,
                "correct_letter": correct_letter,
                "top1_correct": top1_correct,
                "reciprocal_rank": reciprocal_rank,
                "ranked_options": ensemble_ranking,
                "correct_option": correct_answer,
                "predicted_top1_option": ensemble_ranking[0] if ensemble_ranking else "Unknown",
                "approach": f"ensemble_{voting_method}",
                "contributing_models": model_names,
                "num_models": len(model_names)
            }
            
            # Create ensemble result
            ensemble_result = base_result.copy()
            ensemble_result.update({
                "evaluation": ensemble_evaluation,
                "top1_correct": top1_correct,
                "reciprocal_rank": reciprocal_rank,
                "ensemble_method": voting_method,
                "contributing_models": model_names,
                "individual_rankings": rankings,
                "generated_text": f"Ensemble {voting_method}: {' '.join(ensemble_letters)}"
            })
            
            ensemble_results.append(ensemble_result)
            successful_ensembles += 1
            
        except Exception as e:
            print(f"Error creating ensemble for video {video_id}: {e}")
            continue
    
    print(f"Successfully created {successful_ensembles} ensemble predictions")
    return ensemble_results


def evaluate_ensemble(result_paths: List[str], voting_methods: List[str] = None, output_dir: Path = None) -> Dict:
    """Evaluate ensemble results using multiple voting methods."""
    if voting_methods is None:
        voting_methods = ["borda", "plurality", "approval", "condorcet"]
    
    print(f"\n{'='*80}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*80}")
    print(f"Input paths: {result_paths}")
    print(f"Voting methods: {voting_methods}")
    
    # Load results from all folders
    all_results = load_multiple_result_folders(result_paths)
    
    if len(all_results) < 2:
        print("Error: Need at least 2 result folders for ensemble evaluation")
        return {}
    
    print(f"\nLoaded results from {len(all_results)} folders:")
    for folder_name, results in all_results.items():
        print(f"  {folder_name}: {len(results)} samples")
    
    ensemble_evaluations = {}
    comparison_data = []
    
    # Evaluate each voting method
    for method in voting_methods:
        print(f"\n{'-'*50}")
        print(f"Evaluating {method.upper()} voting method")
        print(f"{'-'*50}")
        
        # Create ensemble results
        ensemble_results = create_ensemble_results(all_results, method)
        
        if not ensemble_results:
            print(f"No ensemble results generated for {method}")
            continue
        
        # Evaluate ensemble
        evaluation = {
            'method': method,
            'ensemble_size': len(all_results),
            'contributing_folders': list(all_results.keys()),
            'evaluation_timestamp': datetime.now().isoformat(),
            'basic_metrics': calculate_basic_metrics(ensemble_results),
            'ranking_metrics': calculate_ranking_metrics(ensemble_results),
            'option_analysis': calculate_option_analysis(ensemble_results),
            'temporal_analysis': calculate_temporal_analysis(ensemble_results),
            'error_analysis': calculate_error_analysis(ensemble_results),
            'ensemble_results': ensemble_results,
        }
        
        ensemble_evaluations[method] = evaluation
        
        # Print summary
        basic = evaluation['basic_metrics']
        ranking = evaluation['ranking_metrics']
        
        print(f"Ensemble samples: {basic.get('total_samples', 0)}")
        print(f"Top-1 Accuracy: {basic.get('top1_accuracy', 0):.3f} ({basic.get('top1_correct', 0)} correct)")
        print(f"Top-2 Accuracy: {basic.get('top2_accuracy', 0):.3f} ({basic.get('top2_correct', 0)} correct)")
        print(f"Top-3 Accuracy: {basic.get('top3_accuracy', 0):.3f} ({basic.get('top3_correct', 0)} correct)")
        print(f"Mean Reciprocal Rank: {basic.get('mean_reciprocal_rank', 0):.3f}")
        print(f"Median Reciprocal Rank: {basic.get('median_reciprocal_rank', 0):.3f}")
        
        if ranking.get('mean_rank'):
            print(f"Mean Rank: {ranking.get('mean_rank', 0):.2f}")
            print(f"Median Rank: {ranking.get('median_rank', 0):.2f}")
        
        # Add to comparison data
        comparison_row = {
            'method': method,
            'ensemble_size': len(all_results),
            'total_samples': basic.get('total_samples', 0),
            'top1_accuracy': basic.get('top1_accuracy', 0),
            'top2_accuracy': basic.get('top2_accuracy', 0),
            'top3_accuracy': basic.get('top3_accuracy', 0),
            'top4_accuracy': basic.get('top4_accuracy', 0),
            'mean_reciprocal_rank': basic.get('mean_reciprocal_rank', 0),
            'median_reciprocal_rank': basic.get('median_reciprocal_rank', 0),
            'mean_rank': ranking.get('mean_rank', 0),
            'median_rank': ranking.get('median_rank', 0),
        }
        comparison_data.append(comparison_row)
    
    # Compare voting methods
    if comparison_data:
        print(f"\n{'='*80}")
        print("ENSEMBLE METHODS COMPARISON")
        print(f"{'='*80}")
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display comparison table
        display_cols = ['method', 'total_samples', 'top1_accuracy', 'top2_accuracy', 
                       'mean_reciprocal_rank', 'mean_rank']
        
        if all(col in df_comparison.columns for col in display_cols):
            display_df = df_comparison[display_cols].copy()
            display_df = display_df.round(3)
            print(display_df.to_string(index=False))
        
        # Find best method
        print(f"\n{'='*50}")
        print("BEST ENSEMBLE METHODS")
        print(f"{'='*50}")
        
        for metric in ['top1_accuracy', 'mean_reciprocal_rank']:
            if metric in df_comparison.columns:
                best_idx = df_comparison[metric].idxmax()
                best_method = df_comparison.loc[best_idx, 'method']
                best_value = df_comparison.loc[best_idx, metric]
                print(f"{metric.replace('_', ' ').title()}: {best_method} ({best_value:.3f})")
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed ensemble evaluations
        ensemble_file = output_dir / "ensemble_evaluations.json"
        with open(ensemble_file, 'w', encoding='utf-8') as f:
            json.dump(ensemble_evaluations, f, indent=2, ensure_ascii=False)
        print(f"\nEnsemble evaluations saved to: {ensemble_file}")
        
        # Save comparison table
        if comparison_data:
            comparison_file = output_dir / "ensemble_comparison.csv"
            df_comparison.to_csv(comparison_file, index=False)
            print(f"Ensemble comparison saved to: {comparison_file}")
        
        # Save individual ensemble results
        for method, evaluation in ensemble_evaluations.items():
            method_file = output_dir / f"ensemble_{method}_results.json"
            with open(method_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation['ensemble_results'], f, indent=2, ensure_ascii=False)
            print(f"Ensemble {method} results saved to: {method_file}")
    
    return {
        'ensemble_evaluations': ensemble_evaluations,
        'comparison_data': comparison_data,
        'contributing_folders': list(all_results.keys()),
        'summary': {
            'total_methods_evaluated': len(ensemble_evaluations),
            'ensemble_size': len(all_results),
            'evaluation_timestamp': datetime.now().isoformat(),
        }
    }


def calculate_basic_metrics(results: List[Dict]) -> Dict:
    """Calculate basic accuracy and ranking metrics."""
    if not results:
        return {}
    
    total_samples = len(results)
    top1_correct = sum(1 for r in results if r.get('top1_correct', False))
    
    # Calculate reciprocal ranks
    reciprocal_ranks = []
    top_k_counts = {2: 0, 3: 0, 4: 0}  # Top-2, Top-3, Top-4 accuracy
    
    for result in results:
        rr = result.get('reciprocal_rank', 0.0)
        reciprocal_ranks.append(rr)
        
        # Calculate top-k accuracy from evaluation data
        evaluation = result.get('evaluation', {})
        if evaluation:
            predicted_ranking = evaluation.get('predicted_ranking', [])
            correct_letter = evaluation.get('correct_letter', '')
            
            if correct_letter and correct_letter in predicted_ranking:
                rank = predicted_ranking.index(correct_letter) + 1
                for k in top_k_counts:
                    if rank <= k:
                        top_k_counts[k] += 1
    
    metrics = {
        'total_samples': total_samples,
        'top1_correct': top1_correct,
        'top1_accuracy': top1_correct / total_samples if total_samples > 0 else 0.0,
        'mean_reciprocal_rank': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        'median_reciprocal_rank': np.median(reciprocal_ranks) if reciprocal_ranks else 0.0,
        'std_reciprocal_rank': np.std(reciprocal_ranks) if reciprocal_ranks else 0.0,
    }
    
    # Add top-k accuracies
    for k, count in top_k_counts.items():
        metrics[f'top{k}_accuracy'] = count / total_samples if total_samples > 0 else 0.0
        metrics[f'top{k}_correct'] = count
    
    return metrics


def calculate_ranking_metrics(results: List[Dict]) -> Dict:
    """Calculate detailed ranking-specific metrics."""
    ranking_data = []
    position_distributions = defaultdict(Counter)
    
    for result in results:
        evaluation = result.get('evaluation', {})
        if not evaluation:
            continue
            
        predicted_ranking = evaluation.get('predicted_ranking', [])
        correct_letter = evaluation.get('correct_letter', '')
        
        if correct_letter and correct_letter in predicted_ranking:
            rank = predicted_ranking.index(correct_letter) + 1
            ranking_data.append(rank)
            position_distributions[correct_letter][rank] += 1
    
    if not ranking_data:
        return {}
    
    # Calculate rank distribution
    rank_counts = Counter(ranking_data)
    total_ranked = len(ranking_data)
    
    metrics = {
        'rank_distribution': dict(rank_counts),
        'rank_percentages': {k: v / total_ranked for k, v in rank_counts.items()},
        'mean_rank': np.mean(ranking_data),
        'median_rank': np.median(ranking_data),
        'std_rank': np.std(ranking_data),
        'min_rank': min(ranking_data),
        'max_rank': max(ranking_data),
        'total_ranked_samples': total_ranked,
    }
    
    return metrics


def calculate_option_analysis(results: List[Dict]) -> Dict:
    """Analyze performance by option characteristics."""
    option_analysis = {
        'by_num_options': defaultdict(list),
        'by_correct_option_text': defaultdict(list),
        'common_correct_options': Counter(),
        'common_incorrect_predictions': Counter(),
    }
    
    for result in results:
        evaluation = result.get('evaluation', {})
        if not evaluation:
            continue
            
        options = result.get('options', [])
        correct_option = evaluation.get('correct_option', '')
        predicted_top1 = evaluation.get('predicted_top1_option', '')
        top1_correct = result.get('top1_correct', False)
        reciprocal_rank = result.get('reciprocal_rank', 0.0)
        
        # Group by number of options
        num_options = len(options)
        option_analysis['by_num_options'][num_options].append({
            'top1_correct': top1_correct,
            'reciprocal_rank': reciprocal_rank
        })
        
        # Track correct options
        option_analysis['common_correct_options'][correct_option] += 1
        
        # Track incorrect predictions
        if not top1_correct and predicted_top1:
            option_analysis['common_incorrect_predictions'][predicted_top1] += 1
    
    # Calculate metrics per option count
    option_metrics = {}
    for num_opts, data in option_analysis['by_num_options'].items():
        if data:
            option_metrics[f'{num_opts}_options'] = {
                'count': len(data),
                'top1_accuracy': sum(d['top1_correct'] for d in data) / len(data),
                'mean_rr': np.mean([d['reciprocal_rank'] for d in data]),
            }
    
    option_analysis['metrics_by_option_count'] = option_metrics
    
    return option_analysis


def calculate_temporal_analysis(results: List[Dict]) -> Dict:
    """Analyze performance over processing order (if available)."""
    temporal_data = []
    
    for i, result in enumerate(results):
        sample_id = result.get('sample_id', i)
        top1_correct = result.get('top1_correct', False)
        reciprocal_rank = result.get('reciprocal_rank', 0.0)
        inference_time = result.get('inference_time', 0.0)
        
        temporal_data.append({
            'sample_id': sample_id,
            'processing_order': i,
            'top1_correct': top1_correct,
            'reciprocal_rank': reciprocal_rank,
            'inference_time': inference_time
        })
    
    if not temporal_data:
        return {}
    
    # Calculate moving averages
    window_size = min(50, len(temporal_data) // 10) if len(temporal_data) > 10 else len(temporal_data)
    df = pd.DataFrame(temporal_data)
    
    metrics = {
        'total_inference_time': df['inference_time'].sum(),
        'mean_inference_time': df['inference_time'].mean(),
        'median_inference_time': df['inference_time'].median(),
        'std_inference_time': df['inference_time'].std(),
        'samples_per_second': len(temporal_data) / df['inference_time'].sum() if df['inference_time'].sum() > 0 else 0,
    }
    
    if len(temporal_data) > window_size:
        df['accuracy_ma'] = df['top1_correct'].rolling(window=window_size).mean()
        df['rr_ma'] = df['reciprocal_rank'].rolling(window=window_size).mean()
        metrics['accuracy_trend'] = df['accuracy_ma'].dropna().tolist()
        metrics['rr_trend'] = df['rr_ma'].dropna().tolist()
    
    return metrics


def calculate_error_analysis(results: List[Dict]) -> Dict:
    """Analyze types and patterns of errors."""
    error_analysis = {
        'by_question_type': defaultdict(list),
        'by_question_length': defaultdict(list),
        'prediction_patterns': Counter(),
        'confidence_analysis': defaultdict(list),
    }
    
    for result in results:
        question = result.get('question', '')
        evaluation = result.get('evaluation', {})
        top1_correct = result.get('top1_correct', False)
        reciprocal_rank = result.get('reciprocal_rank', 0.0)
        
        # Analyze by question length
        q_length_bucket = f"{(len(question) // 50) * 50}-{(len(question) // 50 + 1) * 50}"
        error_analysis['by_question_length'][q_length_bucket].append({
            'top1_correct': top1_correct,
            'reciprocal_rank': reciprocal_rank,
        })
        
        # Analyze prediction patterns
        if evaluation:
            predicted_ranking = evaluation.get('predicted_ranking', [])
            pattern = ''.join(predicted_ranking)
            error_analysis['prediction_patterns'][pattern] += 1
            
            # Confidence analysis based on reciprocal rank
            confidence_bucket = f"{int(reciprocal_rank * 4) / 4:.2f}-{int(reciprocal_rank * 4 + 1) / 4:.2f}"
            error_analysis['confidence_analysis'][confidence_bucket].append(top1_correct)
    
    # Calculate metrics
    error_metrics = {}
    
    # Question length analysis
    for length_range, data in error_analysis['by_question_length'].items():
        if data:
            error_metrics[f'length_{length_range}'] = {
                'count': len(data),
                'accuracy': sum(d['top1_correct'] for d in data) / len(data),
                'mean_rr': np.mean([d['reciprocal_rank'] for d in data]),
            }
    
    # Confidence analysis
    for conf_range, correctness in error_analysis['confidence_analysis'].items():
        if correctness:
            error_metrics[f'confidence_{conf_range}'] = {
                'count': len(correctness),
                'accuracy': sum(correctness) / len(correctness),
            }
    
    error_analysis['metrics'] = error_metrics
    
    return error_analysis


def evaluate_single_folder(result_dir: Path, output_dir: Path = None) -> Dict:
    """Evaluate results from a single folder."""
    print(f"\nEvaluating: {result_dir}")
    
    # Load results
    summary, results = load_result_files(result_dir)
    
    if not results:
        print(f"No results found in {result_dir}")
        return {}
    
    # Calculate comprehensive metrics
    evaluation = {
        'folder_name': result_dir.name,
        'folder_path': str(result_dir),
        'evaluation_timestamp': datetime.now().isoformat(),
        'basic_metrics': calculate_basic_metrics(results),
        'ranking_metrics': calculate_ranking_metrics(results),
        'option_analysis': calculate_option_analysis(results),
        'temporal_analysis': calculate_temporal_analysis(results),
        'error_analysis': calculate_error_analysis(results),
        'original_summary': summary,
    }
    
    # Print summary
    basic = evaluation['basic_metrics']
    ranking = evaluation['ranking_metrics']
    
    print(f"Samples: {basic.get('total_samples', 0)}")
    print(f"Top-1 Accuracy: {basic.get('top1_accuracy', 0):.3f} ({basic.get('top1_correct', 0)} correct)")
    print(f"Top-2 Accuracy: {basic.get('top2_accuracy', 0):.3f} ({basic.get('top2_correct', 0)} correct)")
    print(f"Top-3 Accuracy: {basic.get('top3_accuracy', 0):.3f} ({basic.get('top3_correct', 0)} correct)")
    print(f"Mean Reciprocal Rank: {basic.get('mean_reciprocal_rank', 0):.3f}")
    print(f"Median Reciprocal Rank: {basic.get('median_reciprocal_rank', 0):.3f}")
    
    if ranking.get('mean_rank'):
        print(f"Mean Rank: {ranking.get('mean_rank', 0):.2f}")
        print(f"Median Rank: {ranking.get('median_rank', 0):.2f}")
    
    # Save detailed evaluation if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_file = output_dir / f"{result_dir.name}_evaluation.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        print(f"Detailed evaluation saved to: {eval_file}")
    
    return evaluation


def evaluate_multiple_folders(base_dir: Path, pattern: str = "*", output_dir: Path = None) -> Dict:
    """Evaluate multiple result folders and create comparison."""
    print(f"Evaluating multiple folders in: {base_dir}")
    print(f"Pattern: {pattern}")
    
    result_dirs = sorted([d for d in base_dir.glob(pattern) if d.is_dir()])
    print(f"Found {len(result_dirs)} directories")
    
    evaluations = {}
    comparison_data = []
    
    for result_dir in result_dirs:
        evaluation = evaluate_single_folder(result_dir, output_dir)
        if evaluation:
            evaluations[result_dir.name] = evaluation
            
            # Extract key metrics for comparison
            basic = evaluation.get('basic_metrics', {})
            ranking = evaluation.get('ranking_metrics', {})
            temporal = evaluation.get('temporal_analysis', {})
            
            comparison_row = {
                'folder_name': result_dir.name,
                'total_samples': basic.get('total_samples', 0),
                'top1_accuracy': basic.get('top1_accuracy', 0),
                'top2_accuracy': basic.get('top2_accuracy', 0),
                'top3_accuracy': basic.get('top3_accuracy', 0),
                'top4_accuracy': basic.get('top4_accuracy', 0),
                'mean_reciprocal_rank': basic.get('mean_reciprocal_rank', 0),
                'median_reciprocal_rank': basic.get('median_reciprocal_rank', 0),
                'std_reciprocal_rank': basic.get('std_reciprocal_rank', 0),
                'mean_rank': ranking.get('mean_rank', 0),
                'median_rank': ranking.get('median_rank', 0),
                'mean_inference_time': temporal.get('mean_inference_time', 0),
                'total_inference_time': temporal.get('total_inference_time', 0),
                'samples_per_second': temporal.get('samples_per_second', 0),
            }
            comparison_data.append(comparison_row)
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # Print comparison table
    if not df_comparison.empty:
        print(f"\n{'='*100}")
        print("COMPARISON TABLE")
        print(f"{'='*100}")
        
        # Select key columns for display
        display_cols = ['folder_name', 'total_samples', 'top1_accuracy', 'top2_accuracy', 
                       'mean_reciprocal_rank', 'mean_rank', 'samples_per_second']
        
        if all(col in df_comparison.columns for col in display_cols):
            display_df = df_comparison[display_cols].copy()
            display_df = display_df.round(3)
            print(display_df.to_string(index=False))
        
        # Find best performers
        print(f"\n{'='*50}")
        print("BEST PERFORMERS")
        print(f"{'='*50}")
        
        metrics_to_check = [
            ('top1_accuracy', 'Top-1 Accuracy', 'max'),
            ('mean_reciprocal_rank', 'Mean Reciprocal Rank', 'max'),
            ('samples_per_second', 'Speed (samples/sec)', 'max'),
        ]
        
        for metric, name, direction in metrics_to_check:
            if metric in df_comparison.columns:
                if direction == 'max':
                    best_idx = df_comparison[metric].idxmax()
                else:
                    best_idx = df_comparison[metric].idxmin()
                    
                best_folder = df_comparison.loc[best_idx, 'folder_name']
                best_value = df_comparison.loc[best_idx, metric]
                print(f"{name}: {best_folder} ({best_value:.3f})")
    
    # Save comprehensive comparison
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        if not df_comparison.empty:
            comparison_file = output_dir / "comparison_table.csv"
            df_comparison.to_csv(comparison_file, index=False)
            print(f"\nComparison table saved to: {comparison_file}")
        
        # Save detailed evaluations
        detailed_file = output_dir / "detailed_evaluations.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
        print(f"Detailed evaluations saved to: {detailed_file}")
        
        # Generate plots if matplotlib available
        try:
            create_comparison_plots(df_comparison, output_dir)
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    return {
        'evaluations': evaluations,
        'comparison_table': df_comparison.to_dict('records') if not df_comparison.empty else [],
        'summary': {
            'total_folders_evaluated': len(evaluations),
            'evaluation_timestamp': datetime.now().isoformat(),
        }
    }


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    if df.empty:
        return
    
    plt.style.use('default')
    
    # Top-1 Accuracy comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('top1_accuracy', ascending=True)
    plt.barh(range(len(df_sorted)), df_sorted['top1_accuracy'])
    plt.yticks(range(len(df_sorted)), df_sorted['folder_name'], fontsize=8)
    plt.xlabel('Top-1 Accuracy')
    plt.title('Top-1 Accuracy Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'top1_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # MRR comparison
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('mean_reciprocal_rank', ascending=True)
    plt.barh(range(len(df_sorted)), df_sorted['mean_reciprocal_rank'])
    plt.yticks(range(len(df_sorted)), df_sorted['folder_name'], fontsize=8)
    plt.xlabel('Mean Reciprocal Rank')
    plt.title('Mean Reciprocal Rank Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'mrr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Speed comparison (if inference time data available)
    if 'samples_per_second' in df.columns and df['samples_per_second'].sum() > 0:
        plt.figure(figsize=(12, 6))
        df_sorted = df.sort_values('samples_per_second', ascending=True)
        plt.barh(range(len(df_sorted)), df_sorted['samples_per_second'])
        plt.yticks(range(len(df_sorted)), df_sorted['folder_name'], fontsize=8)
        plt.xlabel('Samples per Second')
        plt.title('Inference Speed Comparison')
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Multiple Choice Video QA Evaluation")
    parser.add_argument("input_path", type=str, nargs="*", help="Path(s) to result folder(s). For ensemble: provide multiple paths")
    parser.add_argument("--output", "-o", type=str, help="Output directory for evaluation results")
    parser.add_argument("--pattern", "-p", type=str, default="*", help="Pattern to match folders when evaluating multiple (default: '*')")
    parser.add_argument("--single", "-s", action="store_true", help="Evaluate single folder only (default: auto-detect)")
    parser.add_argument("--multiple", "-m", action="store_true", help="Evaluate multiple folders (default: auto-detect)")
    parser.add_argument("--ensemble", "-e", action="store_true", help="Perform ensemble evaluation with voting")
    parser.add_argument("--voting-methods", "--vm", type=str, nargs="*",
                        default=["borda", "plurality", "approval", "condorcet"],
                        help="Voting methods for ensemble (default: borda plurality approval condorcet)")
    parser.add_argument("--ensemble-paths", "--ep", type=str, nargs="+",
                        help="Explicit paths for ensemble evaluation (alternative to input_path)")
    
    args = parser.parse_args()
    
    # Handle ensemble evaluation
    if args.ensemble or args.ensemble_paths:
        print("Ensemble evaluation mode")
        
        # Get ensemble paths
        if args.ensemble_paths:
            ensemble_paths = args.ensemble_paths
        elif len(args.input_path) >= 2:
            ensemble_paths = args.input_path
        else:
            print("Error: Ensemble evaluation requires at least 2 result paths")
            print("Use --ensemble-paths path1 path2 path3... or provide multiple input_path arguments")
            return
        
        output_dir = Path(args.output) if args.output else None
        evaluate_ensemble(ensemble_paths, args.voting_methods, output_dir)
        print("\nEnsemble evaluation completed!")
        return
    
    # Handle regular evaluation
    if not args.input_path:
        print("Error: No input path provided")
        return
    
    input_path = Path(args.input_path[0])  # Take first path for regular evaluation
    output_dir = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Auto-detect evaluation mode
    if args.single:
        mode = "single"
    elif args.multiple:
        mode = "multiple"
    else:
        # Auto-detect: if it contains summary.json or individual result files, treat as single
        # Otherwise, treat as multiple
        if (input_path / "summary.json").exists() or any(input_path.glob("*.json")):
            mode = "single"
        else:
            mode = "multiple"
    
    print(f"Evaluation mode: {mode}")
    
    if mode == "single":
        evaluate_single_folder(input_path, output_dir)
    else:
        evaluate_multiple_folders(input_path, args.pattern, output_dir)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
