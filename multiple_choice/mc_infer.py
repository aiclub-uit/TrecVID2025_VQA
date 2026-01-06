import argparse
import json
import os
import re
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List
import torch
import cv2
import numpy as np
from PIL import Image

# Import model classes
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available")
    VLLM_AVAILABLE = False

# Import custom model classes
from models.vllm import VLLMModel
from models.transformers import TransformersModel
from models.llama_model import LlamaCppModel


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def extract_frames(video_path: str, max_frames: int = 32) -> List[Image.Image]:
    """Extract frames from video uniformly."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Calculate frame indices to extract uniformly
    if total_frames <= max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    return frames


def create_prompt(question: str, options: List[str], template: str = None) -> str:
    """Create multiple choice prompt using template from config."""
    if template:
        # Use template from config
        option_dict = {}
        option_labels = ['A', 'B', 'C', 'D']
        for i, option in enumerate(options):
            if i < len(option_labels):
                option_dict[f'option_{option_labels[i].lower()}'] = option
        
        # Fill in the template
        try:
            prompt = template.format(question=question, **option_dict)
        except KeyError as e:
            print(f"Warning: Template missing key {e}, falling back to default prompt")
            prompt = create_default_prompt(question, options)
    else:
        # Fallback to default prompt
        prompt = create_default_prompt(question, options)
    
    return prompt


def create_default_prompt(question: str, options: List[str]) -> str:
    """Create default multiple choice prompt for ranking."""
    option_labels = ['A', 'B', 'C', 'D']
    prompt = f"Watch the video and answer the following multiple choice question.\n\nQuestion: {question}\n\nOptions:\n"
    
    for i, option in enumerate(options):
        if i < len(option_labels):
            prompt += f"{option_labels[i]}. {option}\n"
    
    prompt += "\nRank all options from most likely correct to least likely correct. Provide your ranking as letters separated by spaces (e.g., 'B A D C'):"
    return prompt


def load_prepared_data(data_dir: str) -> List[Dict]:
    """Load prepared multiple choice data."""
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


def evaluate_answer_no_ground_truth(predicted: str, options: List[str], approach: str = "ranking") -> Dict:
    """Evaluate the predicted ranking without ground truth."""
    # Extract ranking from prediction
    predicted = predicted.strip().upper()
    
    # First try to find a clear ranking pattern like "B A D C" (space-separated letters)
    
    # Look for space-separated letters pattern first
    space_pattern = re.search(r'\b([ABCD](?:\s+[ABCD]){1,3})\b', predicted)
    if space_pattern:
        # Found space-separated pattern, use this order
        letters = space_pattern.group(1).split()
    else:
        # Fallback: extract all letters in order they appear
        letters = re.findall(r'[ABCD]', predicted)
    
    # If no valid letters found, default to A B C D
    option_labels = ['A', 'B', 'C', 'D']
    if not letters:
        letters = option_labels[:len(options)]
    
    # Ensure we have the right number of letters and no duplicates
    unique_letters = []
    for letter in letters:
        if letter not in unique_letters and letter in option_labels[:len(options)]:
            unique_letters.append(letter)
    
    # Add missing letters at the end
    for label in option_labels[:len(options)]:
        if label not in unique_letters:
            unique_letters.append(label)
    
    # Truncate to match number of options
    predicted_ranking = unique_letters[:len(options)]
    
    # Convert ranking back to options
    ranked_options = []
    for letter in predicted_ranking:
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            ranked_options.append(options[idx])
    
    return {
        "predicted_ranking": predicted_ranking,
        "ranked_options": ranked_options,
        "predicted_top1_option": ranked_options[0] if ranked_options else "Unknown",
        "approach": approach
    }


def evaluate_answer(predicted: str, correct_option: str, options: List[str], approach: str = "ranking") -> Dict:
    """Evaluate the predicted ranking."""
    # Extract ranking from prediction
    predicted = predicted.strip().upper()
    
    # First try to find a clear ranking pattern like "B A D C" (space-separated letters)
    
    # Look for space-separated letters pattern first
    space_pattern = re.search(r'\b([ABCD](?:\s+[ABCD]){1,3})\b', predicted)
    if space_pattern:
        # Found space-separated pattern, use this order
        letters = space_pattern.group(1).split()
    else:
        # Fallback: extract all letters in order they appear
        letters = re.findall(r'[ABCD]', predicted)
    
    # If no valid letters found, default to A B C D
    option_labels = ['A', 'B', 'C', 'D']
    if not letters:
        letters = option_labels[:len(options)]
    
    # Ensure we have the right number of letters and no duplicates
    unique_letters = []
    for letter in letters:
        if letter not in unique_letters and letter in option_labels[:len(options)]:
            unique_letters.append(letter)
    
    # Add missing letters at the end
    for label in option_labels[:len(options)]:
        if label not in unique_letters:
            unique_letters.append(label)
    
    # Truncate to match number of options
    predicted_ranking = unique_letters[:len(options)]
    
    # Find correct letter
    correct_letter = None
    for i, option in enumerate(options):
        if option == correct_option and i < len(option_labels):
            correct_letter = option_labels[i]
            break
    
    # Calculate metrics
    top1_correct = (predicted_ranking[0] == correct_letter) if predicted_ranking else False
    
    # Calculate reciprocal rank
    reciprocal_rank = 0.0
    if correct_letter and correct_letter in predicted_ranking:
        rank = predicted_ranking.index(correct_letter) + 1  # 1-based rank
        reciprocal_rank = 1.0 / rank
    
    # Convert ranking back to options
    ranked_options = []
    for letter in predicted_ranking:
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            ranked_options.append(options[idx])
    
    return {
        "predicted_ranking": predicted_ranking,
        "correct_letter": correct_letter,
        "top1_correct": top1_correct,
        "reciprocal_rank": reciprocal_rank,
        "ranked_options": ranked_options,
        "correct_option": correct_option,
        "predicted_top1_option": ranked_options[0] if ranked_options else "Unknown",
        "approach": approach
    }
    
    # Add missing letters at the end
    for label in option_labels[:len(options)]:
        if label not in unique_letters:
            unique_letters.append(label)
    
    # Truncate to match number of options
    predicted_ranking = unique_letters[:len(options)]
    
    # Find correct letter
    correct_letter = None
    for i, option in enumerate(options):
        if option == correct_option and i < len(option_labels):
            correct_letter = option_labels[i]
            break
    
    # Calculate metrics
    top1_correct = (predicted_ranking[0] == correct_letter) if predicted_ranking else False
    
    # Calculate reciprocal rank
    reciprocal_rank = 0.0
    if correct_letter and correct_letter in predicted_ranking:
        rank = predicted_ranking.index(correct_letter) + 1  # 1-based rank
        reciprocal_rank = 1.0 / rank
    
    # Convert ranking back to options
    ranked_options = []
    for letter in predicted_ranking:
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(options):
            ranked_options.append(options[idx])
    
    return {
        "predicted_ranking": predicted_ranking,
        "correct_letter": correct_letter,
        "top1_correct": top1_correct,
        "reciprocal_rank": reciprocal_rank,
        "ranked_options": ranked_options,
        "correct_option": correct_option,
        "predicted_top1_option": ranked_options[0] if ranked_options else "Unknown"
    }


def main():
    parser = argparse.ArgumentParser(description="Multiple Choice Video QA Baseline")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--data_dir", "-d", type=str, required=True, help="Path to prepared data directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for results")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples that already have result files in the output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    print(f"Model: {config.get('model_name', 'Unknown')}")
    print(f"Backend: {config.get('backend', 'Unknown')}")
    
    # Get and display the MC approach
    approach = config.get("mc_config", {}).get("approach", "ranking")
    print(f"Multiple Choice approach: {approach}")
    
    # Display skip existing setting
    if args.skip_existing:
        print("Skip existing results: ENABLED")
    else:
        print("Skip existing results: DISABLED")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model based on model_type
    model_type = config.get("model_type", "vllm")  # Check model_type
    print(f"Initializing {model_type} model...")
    
    if model_type == "vllm":
        if not VLLM_AVAILABLE:
            print("Error: VLLM not available but model_type is 'vllm'")
            return
        model = VLLMModel(config)
    elif model_type == "transformers":
        model = TransformersModel(config)
    elif model_type == "llamacpp":
        model = LlamaCppModel(config)
    else:
        print(f"Error: Unknown model_type: {model_type}")
        return
    
    # Load prepared data
    print(f"Loading data from: {args.data_dir}")
    data = load_prepared_data(args.data_dir)
    print(f"Loaded {len(data)} samples")
    
    if not data:
        print("No data found!")
        return
    
    # Process each sample
    results = []
    top1_correct_count = 0
    total_reciprocal_rank = 0.0
    total_count = 0
    failed_samples = []
    skipped_samples = 0
    
    for i, sample in enumerate(data):
        print(f"\nProcessing sample {i + 1}/{len(data)}: {sample.get('video_id', 'Unknown')}")
        
        # Check if we should skip existing results
        if args.skip_existing:
            video_id = sample.get('video_id', i)
            result_file = output_dir / f"{video_id}.json"
            if result_file.exists():
                print(f"Skipping sample {video_id} - result file already exists")
                skipped_samples += 1
                continue
        
        try:
            # Get video path
            video_path = sample.get("video_path", "")
            if not video_path or not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            
            # Extract question and options
            question = sample.get("question", "")
            correct_answer = sample.get("correct_answer", None)  # May be None for test data
            
            # Extract audio transcription with better error handling
            recognized_text = ""
            try:
                speech_recognition = sample.get("speech_recognition", {})
                if speech_recognition and isinstance(speech_recognition, dict):
                    # First check if there's a direct recognized_text field
                    recognized_text = speech_recognition.get("recognized_text", "")
                    
                    # If not found, check if there's nested speech_recognition (test data format)
                    if not recognized_text and "speech_recognition" in speech_recognition:
                        nested_speech = speech_recognition.get("speech_recognition", {})
                        if isinstance(nested_speech, dict):
                            recognized_text = nested_speech.get("recognized_text", "")
                    
                    if recognized_text:
                        print(f"Audio transcription: '{recognized_text[:100]}...' ({len(recognized_text)} chars)")
                    else:
                        print("No audio transcription text available")
                else:
                    print("No speech recognition data available")
            except Exception as audio_error:
                print(f"Error processing audio transcription: {audio_error}")
                recognized_text = ""
            
            # Get options from choices field (test data) or mc_options (training data)
            options = []
            if "choices" in sample:
                # Test data format - use choices directly
                options = sample.get("choices", [])
            else:
                # Training data format - get shuffled options from mc_options
                mc_options = sample.get("mc_options", [])
                if mc_options:
                    # Sort by rank to get the shuffled order
                    mc_options.sort(key=lambda x: x.get("rank", 0))
                    options = [opt.get("option", "") for opt in mc_options]
            if len(options) < 2:
                print("Not enough options")
                continue
            
            # Extract frames from video (for transformers model)
            if model_type == "transformers":
                frames = model.extract_frames(video_path, config["mc_config"].get("max_frames", 32))
                if not frames:
                    print(f"Could not extract frames from: {video_path}")
                    continue
            
            # Time the inference
            start_time = time.time()
            
            try:
                if model_type == "vllm":
                    # VLLM inference
                    generated_text = model.infer_multiple_choice(video_path, question, options, recognized_text)
                elif model_type == "transformers":
                    # Transformers inference
                    generated_text = model.infer_multiple_choice(video_path, question, options, recognized_text)
                elif model_type == "llamacpp":
                    # LlamaCpp HTTP server inference
                    generated_text = model.infer_multiple_choice(video_path, question, options, recognized_text)
                else:
                    print(f"Unknown model_type: {model_type}")
                    continue
            except Exception as inference_error:
                print(f"Error during model inference: {inference_error}")
                print(f"Sample details - Video: {sample.get('video_id')}, Question length: {len(question)}, Audio length: {len(recognized_text)}")
                failed_samples.append({
                    "video_id": sample.get('video_id'),
                    "error": str(inference_error),
                    "question_length": len(question),
                    "audio_length": len(recognized_text)
                })
                continue
                
            inference_time = time.time() - start_time
            
            if generated_text:
                print(f"Generated: {generated_text}")
                
                # Get approach for evaluation
                approach = config.get("mc_config", {}).get("approach", "ranking")
                
                # Handle evaluation based on whether ground truth is available
                if correct_answer is not None:
                    # Training/validation data with ground truth
                    evaluation = evaluate_answer(generated_text, correct_answer, options, approach)
                    
                    top1_correct = evaluation["top1_correct"]
                    reciprocal_rank = evaluation["reciprocal_rank"]
                    
                    if top1_correct:
                        top1_correct_count += 1
                    total_reciprocal_rank += reciprocal_rank
                    total_count += 1
                    
                    print(f"Predicted ranking: {' '.join(evaluation['predicted_ranking'])}")
                    print(f"Ranked options: {evaluation['ranked_options']}")
                    print(f"Correct: {evaluation['correct_letter']} ({evaluation['correct_option']})")
                    print(f"Top-1 Result: {'✓ Correct' if top1_correct else '✗ Incorrect'}")
                    print(f"Reciprocal Rank: {reciprocal_rank:.3f}")
                    
                    current_accuracy = top1_correct_count / total_count
                    current_mrr = total_reciprocal_rank / total_count
                    print(f"Running Top-1 Accuracy: {top1_correct_count}/{total_count} = {current_accuracy * 100:.1f}%")
                    print(f"Running MRR: {current_mrr:.3f}")
                else:
                    # Test data without ground truth - just extract ranking
                    evaluation = evaluate_answer_no_ground_truth(generated_text, options, approach)
                    
                    print(f"Predicted ranking: {' '.join(evaluation['predicted_ranking'])}")
                    print(f"Ranked options: {evaluation['ranked_options']}")
                    print(f"Top-1 Prediction: {evaluation['predicted_top1_option']}")
                    
                    # Set dummy values for compatibility
                    top1_correct = None
                    reciprocal_rank = None
                
                # Store result
                result = {
                    "sample_id": i,
                    "video_id": sample.get("video_id", ""),
                    "question": question,
                    "options": options,
                    "recognized_text": recognized_text,
                    "generated_text": generated_text,
                    "inference_time": inference_time,
                    "evaluation": evaluation,
                }
                
                # Add ground truth fields only if available
                if correct_answer is not None:
                    result["correct_answer"] = correct_answer
                    result["top1_correct"] = top1_correct
                    result["reciprocal_rank"] = reciprocal_rank
                
                results.append(result)
                
                # Save individual result
                result_file = output_dir / f"{sample.get('video_id', i)}.json"
                result_file.parent.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                print("No output generated")
                continue
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            failed_samples.append({
                "video_id": sample.get('video_id', f'sample_{i}'),
                "error": str(e),
                "error_type": "general_processing_error"
            })
            continue
    
    # Save summary results
    final_top1_accuracy = top1_correct_count / total_count if total_count > 0 else None
    final_mrr = total_reciprocal_rank / total_count if total_count > 0 else None
    approach = config.get("mc_config", {}).get("approach", "ranking")

    summary = {
        "total_samples": len(data),
        "processed_samples": len(results),  # Use actual results count
        "skipped_samples": skipped_samples,
        "failed_samples": len(failed_samples),
        "failed_sample_details": failed_samples,
        "config_file": args.config,
        "model_name": config.get("model_name", "Unknown"),
        "approach": approach,
        "skip_existing_enabled": args.skip_existing,
        "results": results
    }
    
    # Add evaluation metrics only if we have ground truth data
    if total_count > 0:
        summary.update({
            "samples_with_ground_truth": total_count,
            "top1_correct_predictions": top1_correct_count,
            "top1_accuracy": final_top1_accuracy,
            "mean_reciprocal_rank": final_mrr,
            "total_reciprocal_rank": total_reciprocal_rank,
        })
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS")
    print(f"{'=' * 50}")
    print(f"Total samples: {len(data)}")
    print(f"Processed samples: {len(results)}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Failed samples: {len(failed_samples)}")
    
    if total_count > 0:
        # Ground truth metrics available
        print(f"Samples with ground truth: {total_count}")
        print(f"Top-1 correct predictions: {top1_correct_count}")
        print(f"Top-1 accuracy: {final_top1_accuracy * 100:.2f}%")
        print(f"Mean Reciprocal Rank (MRR): {final_mrr:.3f}")
    else:
        # No ground truth available (test data)
        print("No ground truth available - predictions saved for submission")
    
    if failed_samples:
        print("\nFailed samples breakdown:")
        for failed in failed_samples:
            print(f"  - {failed['video_id']}: {failed['error']}")
    
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
