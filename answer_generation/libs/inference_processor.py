"""Inference processing utilities for Video Question Answering."""

import os
import time
from pathlib import Path
from typing import Dict, List

from .evaluation_metrics import EvaluationMetrics
from .result_saver import save_individual_result


def run_batch_inference(
    model,
    data: List[Dict],
    logger,
    evaluator: EvaluationMetrics,
    output_dir: Path,
    config: Dict,
    config_file_path: str,
    args,
) -> List[Dict]:
    """Run inference on batch of prepared data."""
    results = []
    
    # Get number of answer iterations from config or default to 10
    num_iterations = config.get("multiple_answers", {}).get("num_iterations", 10)
    logger.info(f"Multiple answer generation enabled: {num_iterations} iterations per question")

    for i, item in enumerate(data):
        file_stem = item["file_stem"]

        # Skip if result exists and flag is set
        if args.skip_processed:
            individual_file = output_dir / "individual_results" / f"{file_stem}.json"
            if individual_file.exists():
                logger.info(f"Skipping {file_stem} as result file already exists.")
                continue

        logger.info(f"Processing {i + 1}/{len(data)}: {item['file_stem']}")

        # Skip if video doesn't exist
        if not item.get("video_exists", False) or not item.get("video_path"):
            logger.warning(f"Skipping {item['file_stem']}: video not available")
            continue

        video_path = item["video_path"]
        question = item["question"]
        correct_answer = item["correct_answer"]
        incorrect_answers = item.get("incorrect_answers", [])

        # Check if video file actually exists
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            continue

        try:
            # Generate multiple answers
            all_answers = []
            previous_answers = []
            total_inference_time = 0
            
            for iteration in range(num_iterations):
                logger.info(f"Generating answer {iteration + 1}/{num_iterations} for {file_stem}")
                
                # Run inference with timing
                start_time = time.time()
                trace_id = f"batch_inference_{item['file_stem']}_iter_{iteration + 1}"
                
                # Create modified item with previous answers context
                modified_item = item.copy()
                if previous_answers:
                    # Add context about previous answers to prompt the model differently
                    modified_item["previous_answers"] = previous_answers
                    logger.info(f"Added {len(previous_answers)} previous answers to context")

                # Use the enhanced data format for inference
                # This will include audio information if available
                predicted_answer = model.infer(video_path, modified_item, logger, trace_id)
                
                iteration_time = time.time() - start_time
                total_inference_time += iteration_time
                
                # Store this answer for next iterations
                previous_answers.append(predicted_answer)
                all_answers.append({
                    "iteration": iteration + 1,
                    "answer": predicted_answer,
                    "inference_time": iteration_time
                })
                
                logger.info(f"Answer {iteration + 1}: {predicted_answer[:100]}...")

            # Use the first answer as the main predicted answer for compatibility
            predicted_answer = all_answers[0]["answer"]

            # Compute evaluation metrics for the first answer
            sts_score = evaluator.compute_sts(predicted_answer, correct_answer)
            meteor_score_val = evaluator.compute_meteor(
                predicted_answer, correct_answer
            )
            bert_scores = evaluator.compute_bertscore(predicted_answer, correct_answer)
            ndcg_score_val = evaluator.compute_ndcg(
                predicted_answer, correct_answer, incorrect_answers
            )

            # Compute metrics for all answers
            all_metrics = []
            for answer_data in all_answers:
                answer = answer_data["answer"]
                answer_metrics = {
                    "iteration": answer_data["iteration"],
                    "sts_score": evaluator.compute_sts(answer, correct_answer),
                    "meteor_score": evaluator.compute_meteor(answer, correct_answer),
                    **evaluator.compute_bertscore(answer, correct_answer),
                    "ndcg_score": evaluator.compute_ndcg(answer, correct_answer, incorrect_answers),
                    "inference_time": answer_data["inference_time"]
                }
                all_metrics.append(answer_metrics)

            result = {
                "file_stem": item["file_stem"],
                "video_path": video_path,
                "video_url": item.get("video_url", ""),
                "question": question,
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
                "predicted_answer": predicted_answer,  # First answer for compatibility
                "all_answers": all_answers,  # All generated answers
                "inference_time": total_inference_time,
                "config_file_path": config_file_path,
                "command_line_args": {},  # Will be filled in main
                "audio_info": item.get(
                    "speech_recognition", {}
                ),  # Include audio information
                "metrics": {
                    "sts_score": sts_score,
                    "meteor_score": meteor_score_val,
                    "bert_precision": bert_scores["precision"],
                    "bert_recall": bert_scores["recall"],
                    "bert_f1": bert_scores["f1"],
                    "ndcg_score": ndcg_score_val,
                },
                "all_metrics": all_metrics,  # Metrics for all answers
            }

            results.append(result)

            # Save individual result immediately after processing
            save_individual_result(result, config, output_dir)

            # Enhanced logging with audio information
            speech_recognition = item.get("speech_recognition") or {}
            audio_text = speech_recognition.get("recognized_text", "")
            audio_log = f" | Audio: '{audio_text[:30]}...'" if audio_text else ""
            logger.info(
                f"Completed {item['file_stem']} - STS: {sts_score:.3f}, METEOR: {meteor_score_val:.3f}, BERTScore F1: {bert_scores['f1']:.3f}, NDCG: {ndcg_score_val:.3f}{audio_log}"
            )

        except Exception as e:
            logger.error(f"Error processing {item['file_stem']}: {str(e)}")
            continue

    return results
