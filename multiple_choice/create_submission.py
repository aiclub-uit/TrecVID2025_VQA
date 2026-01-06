#!/usr/bin/env python3
"""
Create TREC25 Multiple Choice submission file from JSON result files.
Converts evaluation results to required CSV format: Q_ID,Video_ID,Rank,option_X
Fills missing questions with default A,B,C,D rankings using test dataset.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_test_dataset(csv_path: Path) -> Tuple[Dict[int, Dict], Dict[str, int]]:
    """Load the test dataset CSV to get all questions and their video_id to Q_ID mapping (1-based)."""
    test_data = {}
    video_to_qid = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, quotechar='"')
            
            for row_idx, row in enumerate(csv_reader, 1):
                try:
                    if len(row) != 7:  # q_id, video_id, question, choice1, choice2, choice3, choice4
                        print(f"Warning: Row {row_idx} has {len(row)} columns, expected 7")
                        continue
                    
                    q_id_str, video_id, question, choice_1, choice_2, choice_3, choice_4 = row
                    q_id = int(q_id_str)  # Keep 1-based indexing from the CSV
                    
                    test_data[q_id] = {
                        'video_id': video_id,
                        'question': question,
                        'options': [choice_1, choice_2, choice_3, choice_4]
                    }
                    
                    # Create video_id to Q_ID mapping (1-based)
                    video_to_qid[video_id] = q_id
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing row {row_idx}: {e}")
                    continue
                    
        print(f"Successfully loaded {len(test_data)} questions from test dataset")
        print(f"Created video_id mapping for {len(video_to_qid)} videos")
                    
    except FileNotFoundError:
        print(f"Warning: Test dataset not found at {csv_path}")
        print("Missing questions will be filled with placeholder data")
    
    return test_data, video_to_qid


def extract_ranking_from_result(result_data: dict, video_to_qid: Dict[str, int]) -> List[Tuple[int, str, int, str]]:
    """Extract ranking information from a result JSON file using proper Q_ID mapping."""
    submissions = []
    
    # Get the video_id and find the correct Q_ID
    video_id = result_data.get('video_id', '')
    if not video_id:
        return submissions
        
    # Get the correct Q_ID from the video_to_qid mapping
    q_id = video_to_qid.get(video_id)
    if q_id is None:
        print(f"Warning: No Q_ID mapping found for video_id: {video_id}")
        return submissions
    
    # Get options and ranking
    options = result_data.get('options', [])
    
    # Check for evaluation data
    evaluation = result_data.get('evaluation', {})
    
    # Try to get ranking from ranked_options first
    if 'ranked_options' in evaluation and evaluation['ranked_options']:
        ranked_options = evaluation['ranked_options']
        for rank, option_text in enumerate(ranked_options, 1):
            submissions.append((q_id, video_id, rank, option_text))
    
    # Fallback to predicted_ranking with letter choices
    elif 'predicted_ranking' in evaluation and evaluation['predicted_ranking']:
        predicted_ranking = evaluation['predicted_ranking']
        
        if len(predicted_ranking) == len(options):
            # Convert letter choices to option text
            letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
            
            for rank, letter_choice in enumerate(predicted_ranking, 1):
                letter_choice = letter_choice.strip().upper()
                if letter_choice in letter_to_index:
                    option_idx = letter_to_index[letter_choice]
                    if 0 <= option_idx < len(options):
                        option_text = options[option_idx]
                        submissions.append((q_id, video_id, rank, option_text))
    
    return submissions


def load_result_files(results_dir: Path, video_to_qid: Dict[str, int]) -> List[Tuple[int, str, int, str]]:
    """Load all result JSON files and extract submission data."""
    all_submissions = []
    processed_files = 0
    
    json_files = list(results_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {results_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            submissions = extract_ranking_from_result(result_data, video_to_qid)
            all_submissions.extend(submissions)
            
            if submissions:
                processed_files += 1
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not process file {json_file}: {e}")
    
    print(f"Successfully processed {processed_files} result files")
    return all_submissions


def create_submission_csv(results_dir: Path, csv_path: Path, output_path: Path) -> bool:
    """Create a complete TREC25 submission CSV file."""
    print(f"Loading test dataset from: {csv_path}")
    test_data, video_to_qid = load_test_dataset(csv_path)
    
    if not test_data:
        print("Error: Could not load test dataset")
        return False
        
    print(f"Loaded {len(test_data)} questions from test dataset")
    print(f"Video to Q_ID mapping created for {len(video_to_qid)} videos")
    
    # Load results from JSON files
    print(f"\nLoading results from: {results_dir}")
    all_submissions = load_result_files(results_dir, video_to_qid)
    
    if not all_submissions:
        print("Warning: No submissions found in result files")
    
    # Group submissions by Q_ID
    submissions_by_qid = {}
    for q_id, video_id, rank, option_text in all_submissions:
        if q_id not in submissions_by_qid:
            submissions_by_qid[q_id] = []
        submissions_by_qid[q_id].append((video_id, rank, option_text))
    
    print(f"Found results for {len(submissions_by_qid)} questions")
    
    # Create complete submission data ensuring all (q_id, video_id) pairs match test dataset
    complete_submissions = []
    missing_count = 0
    
    # Iterate through all Q_IDs in test dataset (1-based: 1-2000)
    for q_id in range(1, 2001):  # 1-2000
        if q_id in submissions_by_qid:
            # Use existing results
            for video_id, rank, option_text in submissions_by_qid[q_id]:
                complete_submissions.append((q_id, video_id, rank, option_text))
        elif q_id in test_data:
            # Fill missing with default A,B,C,D ranking using test dataset
            video_id = test_data[q_id]['video_id']
            options = test_data[q_id]['options']
            for rank, option_text in enumerate(options, 1):
                complete_submissions.append((q_id, video_id, rank, option_text))
            missing_count += 1
        else:
            # This should not happen if test_data is complete
            print(f"Warning: No data found for Q_ID {q_id} in test dataset")
            continue
    
    if missing_count > 0:
        print(f"Filled {missing_count} missing questions with default A,B,C,D rankings")
    
    # Sort by Q_ID then rank
    complete_submissions.sort(key=lambda x: (x[0], x[2]))
    
    # Write to CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            
            for q_id, video_id, rank, option_text in complete_submissions:
                csv_writer.writerow([q_id, video_id, rank, option_text])
        
        print(f"\nSubmission file created: {output_path}")
        print(f"Total entries: {len(complete_submissions)}")
        
        # Validate the output
        if validate_submission_format(output_path):
            print("Submission format validation: PASSED")
            return True
        else:
            print("Submission format validation: FAILED")
            return False
            
    except IOError as e:
        print(f"Error writing submission file: {e}")
        return False


def validate_submission_format(csv_path: Path) -> bool:
    """Validate the submission CSV format."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print("Error: Empty CSV file")
            return False
        
        # Check first few lines format
        print("First 5 lines of submission:")
        for i, line in enumerate(lines[:5]):
            print(f"  {i + 1}: {line.strip()}")
            
            # Basic format check - count commas outside quotes
            line_stripped = line.strip()
            comma_count = 0
            in_quotes = False
            
            for char in line_stripped:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    comma_count += 1
            
            if comma_count != 3:  # Should have exactly 3 commas
                print(f"Error: Line {i + 1} doesn't have correct format (expected 3 commas): {line.strip()}")
                return False
        
        print(f"✓ Validation passed: {len(lines)} total lines")
        return True
        
    except Exception as e:
        print(f"Error validating submission: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create TREC25 MC submission file from result folder")
    parser.add_argument("input_folder", type=str,
                        help="Path to folder containing JSON result files")
    parser.add_argument("--output", "-o", type=str,
                        help="Output CSV file path (default: submission.csv)")
    parser.add_argument("--test-csv", type=str,
                        help="Path to test dataset CSV file to fill missing questions")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the output format after creation")
    
    args = parser.parse_args()
    
    # Validate input folder
    input_dir = Path(args.input_folder)
    if not input_dir.exists():
        print(f"Error: Input folder does not exist: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("corrected_submission.csv")
    
    # Set test CSV path
    test_csv_path = None
    if args.test_csv:
        test_csv_path = Path(args.test_csv)
        if not test_csv_path.exists():
            print(f"Warning: Test CSV file not found: {test_csv_path}")
            test_csv_path = None
    else:
        # Try default path
        default_test_csv = Path("raw/vqa.testing.dataset.task2.MC (2).csv")
        if default_test_csv.exists():
            test_csv_path = default_test_csv
    
    if not test_csv_path:
        print("Error: No test dataset CSV found. Please provide --test-csv argument.")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder: {input_dir}")
    print(f"Output file: {output_path}")
    print(f"Test dataset: {test_csv_path}")
    print("-" * 50)
    
    # Create submission with correct Q_ID mapping
    success = create_submission_csv(input_dir, test_csv_path, output_path)
    
    if success:
        print(f"\n✓ Corrected submission file created: {output_path}")
        print("Format: Q_ID,Video_ID,Rank,option_X")
        print("Q_IDs now properly match video_ids from test dataset!")
        print("Ready for TREC25 submission!")
    else:
        print("✗ Failed to create submission file")


if __name__ == "__main__":
    main()
