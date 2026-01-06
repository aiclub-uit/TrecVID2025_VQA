#!/usr/bin/env python3
"""
Script to create submission files for the TREC AG (Answer Generation) task.

This script processes individual result JSON files and creates a submission
file in the required format:

Q_ID, Video_ID, Rank, Answer, Time (sec)

Example:
1, tui89Xr_iri, 1, "she found a surprise birthday party", 5
1, tui89Xr_iri, 2, "she found party", 6
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import csv


def apply_answer_transformations(
    answer: str, add_dot: bool = False, convert_digit_to_alpha: bool = True
) -> str:
    """
    Apply transformations to a single answer.

    Args:
        answer: The original answer string
        add_dot: Whether to add dot to end of answer
        convert_digit_to_alpha: Whether to convert single digits to alpha

    Returns:
        Transformed answer string
    """
    digit_to_alpha = {
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    transformed_answer = answer

    if convert_digit_to_alpha and transformed_answer.strip() in digit_to_alpha:
        transformed_answer = digit_to_alpha[transformed_answer.strip()]

    if add_dot and transformed_answer and not transformed_answer.endswith("."):
        transformed_answer = transformed_answer + "."

    return transformed_answer


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_video_id_from_filename(filename: str) -> str:
    """Extract video ID from filename (remove .json extension)."""
    return Path(filename).stem


def generate_diverse_default_answers(
    question: str = "", max_answers: int = 10
) -> List[str]:
    """
    Generate diverse default answers for a missing video.
    Uses the question text to create contextual answers when available.
    """
    import random

    # Base generic answers
    generic_answers = [
        "I don't know",
        "I'm not sure",
        "It's unclear from the video",
        "Cannot determine from the content",
        "The video doesn't show this clearly",
        "No clear answer is visible",
        "Unable to identify",
        "Not visible in the video",
        "Cannot be determined",
        "The answer is not obvious",
    ]

    # Question-based answers (when question is available)
    question_based_answers = []
    if question:
        question_lower = question.lower()

        # For "what" questions
        if question_lower.startswith("what"):
            question_based_answers.extend(
                [
                    "Something happens",
                    "An action occurs",
                    "A change takes place",
                    "An event unfolds",
                    "It shows something interesting",
                ]
            )

        # For "why" questions
        elif question_lower.startswith("why"):
            question_based_answers.extend(
                [
                    "For a specific reason",
                    "Due to the circumstances",
                    "Because of the situation",
                    "It has a purpose",
                    "There's a logical explanation",
                ]
            )

        # For "how many" questions
        elif "how many" in question_lower:
            question_based_answers.extend(
                ["one", "two", "three", "several", "a few", "multiple", "some", "many"]
            )

        # For "how" questions
        elif question_lower.startswith("how"):
            question_based_answers.extend(
                [
                    "In a specific way",
                    "Through a process",
                    "By doing something",
                    "With a method",
                    "Using a technique",
                ]
            )

        # For "when" questions
        elif question_lower.startswith("when"):
            question_based_answers.extend(
                [
                    "At some point",
                    "During the video",
                    "At a specific time",
                    "Later",
                    "Eventually",
                ]
            )

        # For "where" questions
        elif question_lower.startswith("where"):
            question_based_answers.extend(
                [
                    "In a location",
                    "Somewhere in the scene",
                    "At a place",
                    "In the background",
                    "On screen",
                ]
            )

        # For yes/no questions (contains "is", "does", "can", etc.)
        elif any(
            word in question_lower for word in ["is ", "does ", "can ", "will ", "are "]
        ):
            question_based_answers.extend(
                ["yes", "no", "maybe", "possibly", "likely", "probably", "uncertain"]
            )

    # Combine all possible answers
    all_possible_answers = generic_answers + question_based_answers

    # Shuffle and select up to max_answers unique answers
    random.seed(hash(question))  # Consistent randomness based on question
    selected_answers = []

    # Always include some generic answers
    selected_answers.extend(generic_answers[: max_answers // 2])

    # Add question-based answers if available
    if question_based_answers:
        selected_answers.extend(
            question_based_answers[: max_answers - len(selected_answers)]
        )

    # Fill remaining slots with more generic answers if needed
    while len(selected_answers) < max_answers:
        for answer in generic_answers:
            if answer not in selected_answers:
                selected_answers.append(answer)
                break
        if len(selected_answers) >= max_answers:
            break

    return selected_answers[:max_answers]


def create_question_id_mapping(
    results_dirs: List[str],
) -> tuple[Dict[str, int], Dict[str, str]]:
    """
    Create a mapping from video_id to question_id based on the testing dataset CSV.
    Also returns a mapping from video_id to question text for generating diverse default answers.
    First, try to get the order from the testing dataset CSV file,
    otherwise fall back to alphabetical order of video IDs found in results.
    """
    # First, try to get the canonical order from testing dataset CSV
    testing_csv_path = "../data/test/testing.dataset.vqa.2025.csv"
    if os.path.exists(testing_csv_path):
        print(f"Using question order from testing dataset: {testing_csv_path}")
        video_id_to_qid = {}
        video_id_to_question = {}

        with open(testing_csv_path, "r", encoding="utf-8") as f:
            for line in f:
                # Parse CSV line: Q_ID, Video_ID, Question
                parts = [part.strip() for part in line.split(",", 2)]
                if len(parts) >= 3:
                    try:
                        q_id = int(parts[0])
                        video_id = parts[1]
                        question = parts[2].strip('"')  # Remove quotes
                        video_id_to_qid[video_id] = q_id
                        video_id_to_question[video_id] = question
                    except (ValueError, IndexError):
                        continue

        print(f"Loaded {len(video_id_to_qid)} video-to-question mappings from CSV")
        return video_id_to_qid, video_id_to_question

    # Fall back to collecting from result directories and sorting alphabetically
    print("Testing dataset CSV not found, using alphabetical order from results")
    all_video_ids = set()

    # Collect all video IDs from all result directories
    for results_dir in results_dirs:
        individual_results_dir = os.path.join(results_dir, "individual_results")
        if not os.path.exists(individual_results_dir):
            continue

        for filename in os.listdir(individual_results_dir):
            if filename.endswith(".json"):
                video_id = extract_video_id_from_filename(filename)
                all_video_ids.add(video_id)

    # Sort video IDs alphabetically and assign Q_IDs
    sorted_video_ids = sorted(all_video_ids)
    video_id_to_qid = {
        video_id: idx + 1 for idx, video_id in enumerate(sorted_video_ids)
    }
    video_id_to_question = {}  # Empty since we don't have questions
    return video_id_to_qid, video_id_to_question


def process_result_file(
    filepath: str, video_id: str, question_id: int, max_answers: int = 10
) -> List[Dict[str, Any]]:
    """
    Process a single result JSON file and extract submission entries.
    """
    try:
        data = load_json_file(filepath)
        entries = []

        # Check if the file has the expected structure
        if "output" not in data:
            print(f"Warning: {filepath} does not have expected structure")
            return entries

        output = data["output"]

        # Handle multiple answers format (new format)
        if "all_answers" in output:
            all_answers = output["all_answers"]

            # Limit to max_answers (up to 10 for AG task)
            for rank, answer_data in enumerate(all_answers[:max_answers], 1):
                if "answer" in answer_data and "inference_time" in answer_data:
                    # Clean the answer by removing newlines
                    answer = answer_data["answer"].replace("\n", "")
                    # Apply digit-to-alpha transformation
                    answer = apply_answer_transformations(
                        answer, add_dot=False, convert_digit_to_alpha=True
                    )
                    entry = {
                        "Q_ID": question_id,
                        "Video_ID": video_id,
                        "Rank": rank,
                        "Answer": answer,
                        "Time": round(answer_data["inference_time"], 2),
                    }
                    entries.append(entry)

        # Handle single answer format (old format)
        elif "predicted_answer" in output and "inference_time" in output:
            # Clean the answer by removing newlines
            answer = output["predicted_answer"].replace("\n", "")
            # Apply digit-to-alpha transformation
            answer = apply_answer_transformations(
                answer, add_dot=False, convert_digit_to_alpha=True
            )
            entry = {
                "Q_ID": question_id,
                "Video_ID": video_id,
                "Rank": 1,
                "Answer": answer,
                "Time": round(output["inference_time"], 2),
            }
            entries.append(entry)

        else:
            print(f"Warning: {filepath} does not have expected structure")
            return entries

        return entries

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return []


def create_submission_file(
    results_dirs: List[str],
    output_file: str,
    max_answers: int = 10,
    default_answer: str = "I don't know",
):
    """
    Create a submission file by processing all result directories.
    Note: default_answer parameter is kept for backward compatibility but diverse answers will be generated.
    """
    print("Creating question ID mapping...")
    question_id_mapping, video_id_to_question = create_question_id_mapping(results_dirs)
    print(f"Found {len(question_id_mapping)} unique video IDs")

    all_entries = []
    processed_files = 0

    print("Processing result files...")
    for results_dir in results_dirs:
        print(f"Processing directory: {results_dir}")
        individual_results_dir = os.path.join(results_dir, "individual_results")

        if not os.path.exists(individual_results_dir):
            print(f"Warning: {individual_results_dir} does not exist")
            continue

        for filename in os.listdir(individual_results_dir):
            if filename.endswith(".json"):
                video_id = extract_video_id_from_filename(filename)
                question_id = question_id_mapping.get(video_id)

                if question_id is None:
                    print(f"Warning: No question ID found for video {video_id}")
                    continue

                filepath = os.path.join(individual_results_dir, filename)
                entries = process_result_file(
                    filepath, video_id, question_id, max_answers
                )
                all_entries.extend(entries)
                processed_files += 1

    print(f"Processed {processed_files} files, generated {len(all_entries)} entries")
    print(f"Maximum answers per question: {max_answers}")

    # Add diverse default answers to fill up questions that don't have enough answers
    print("Adding diverse default answers to fill incomplete questions...")

    # Group entries by question ID to count existing answers
    question_entries = {}
    for entry in all_entries:
        q_id = entry["Q_ID"]
        if q_id not in question_entries:
            question_entries[q_id] = []
        question_entries[q_id].append(entry)

    default_time = 1.0  # Default inference time

    # Check each question in the dataset
    for video_id, question_id in question_id_mapping.items():
        current_answer_count = len(question_entries.get(question_id, []))

        if current_answer_count < max_answers:
            # Need to add more answers to reach max_answers
            answers_needed = max_answers - current_answer_count

            # Get the question text for this video (if available)
            question_text = video_id_to_question.get(video_id, "")

            # Generate diverse answers based on the question
            diverse_answers = generate_diverse_default_answers(
                question_text, max_answers
            )

            # If this question has some existing answers, start from diverse answers
            # that are different from the existing ones
            existing_answers = [
                entry["Answer"].lower()
                for entry in question_entries.get(question_id, [])
            ]

            # Filter out diverse answers that are too similar to existing ones
            filtered_diverse_answers = []
            for answer in diverse_answers:
                if answer.lower() not in existing_answers:
                    filtered_diverse_answers.append(answer)

            # Take the needed number of diverse answers
            selected_diverse_answers = filtered_diverse_answers[:answers_needed]

            # If we still don't have enough, add more generic ones
            if len(selected_diverse_answers) < answers_needed:
                generic_fallbacks = [
                    "No clear answer",
                    "Uncertain",
                    "Hard to tell",
                    "Not clear",
                    "Cannot say",
                    "Unclear",
                    "Unknown",
                    "Not obvious",
                ]

                for fallback in generic_fallbacks:
                    if len(selected_diverse_answers) >= answers_needed:
                        break
                    if (
                        fallback.lower() not in existing_answers
                        and fallback not in selected_diverse_answers
                    ):
                        selected_diverse_answers.append(fallback)

            # Create entries for each additional diverse answer
            for answer in selected_diverse_answers[:answers_needed]:
                # Apply transformations to each answer
                transformed_answer = apply_answer_transformations(
                    answer, add_dot=False, convert_digit_to_alpha=True
                )
                entry = {
                    "Q_ID": question_id,
                    "Video_ID": video_id,
                    "Rank": current_answer_count
                    + len(
                        [
                            e
                            for e in all_entries
                            if e["Q_ID"] == question_id
                            and e["Answer"] == transformed_answer
                        ]
                    )
                    + 1,  # Temporary rank
                    "Answer": transformed_answer,
                    "Time": default_time,
                }
                all_entries.append(entry)

            if current_answer_count == 0:
                print(
                    f"  Q_ID {question_id}: Added {answers_needed} diverse answers (no existing answers)"
                )
            else:
                print(
                    f"  Q_ID {question_id}: Added {answers_needed} diverse answers ({current_answer_count} existing answers)"
                )

    print(f"After adding diverse answers: {len(all_entries)} total entries")

    # Sort entries by Q_ID and then by original rank (to preserve order within each result file)
    all_entries.sort(key=lambda x: (x["Q_ID"], x["Rank"]))

    # Reassign ranks to be consecutive and unique per question
    print("Reassigning ranks to ensure uniqueness per question...")
    current_qid = None
    current_rank = 1

    for entry in all_entries:
        if entry["Q_ID"] != current_qid:
            # New question, reset rank
            current_qid = entry["Q_ID"]
            current_rank = 1

        entry["Rank"] = current_rank
        current_rank += 1

    # Write to CSV file
    print(f"Writing submission to {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header (optional, remove if not needed)
        # writer.writerow(['Q_ID', 'Video_ID', 'Rank', 'Answer', 'Time'])

        for entry in all_entries:
            writer.writerow(
                [
                    entry["Q_ID"],
                    entry["Video_ID"],
                    entry["Rank"],
                    entry["Answer"],  # Let CSV writer handle quoting
                    entry["Time"],
                ]
            )

    print(f"Submission file created: {output_file}")
    print(f"Total entries: {len(all_entries)}")

    # Print some statistics
    unique_questions = len(set(entry["Q_ID"] for entry in all_entries))
    unique_videos = len(set(entry["Video_ID"] for entry in all_entries))
    print(f"Unique questions: {unique_questions}")
    print(f"Unique videos: {unique_videos}")

    # Print average answers per question
    if unique_questions > 0:
        avg_answers = len(all_entries) / unique_questions
        print(f"Average answers per question: {avg_answers:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create submission file for TREC AG task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_submission.py -r results/test_final_vl3_ten_no_context_v2_slice_1000_1500_0828_1946 -o submission.csv
  
  python create_submission.py \\
    -r results/test_final_vl3_ten_no_context_v2_slice_1000_1500_0828_1946 \\
    -r results/test_final_vl3_ten_no_context_v2_slice_1000_1500_0828_1950 \\
    -r results/test_final_vl3_ten_no_context_v2_slice_1500_2000_0828_1947 \\
    -r results/test_final_vl3_ten_no_context_v2_slice_1500_2000_0828_1950 \\
    -o combined_submission.csv
        """,
    )

    parser.add_argument(
        "-r",
        "--results-dir",
        action="append",
        required=True,
        help="Path to results directory (can be specified multiple times)",
    )

    parser.add_argument(
        "-o", "--output", required=True, help="Output submission file path"
    )

    parser.add_argument(
        "--max-answers",
        type=int,
        default=10,
        help="Maximum number of answers per question (default: 10)",
    )

    parser.add_argument(
        "--default-answer",
        type=str,
        default="I don't know",
        help="Default answer for videos without results (default: 'I don't know')",
    )

    args = parser.parse_args()

    # Validate input directories
    for results_dir in args.results_dir:
        if not os.path.exists(results_dir):
            print(f"Error: Results directory {results_dir} does not exist")
            sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create submission file
    create_submission_file(
        args.results_dir, args.output, args.max_answers, args.default_answer
    )


if __name__ == "__main__":
    main()
