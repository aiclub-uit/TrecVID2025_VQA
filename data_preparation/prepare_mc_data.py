#!/usr/bin/env python3
"""
Prepare Multiple Choice (MC) data for VQA inference.

This script converts the CSV dataset format to individual JSON files
that can be used by the MC inference script.

Usage:
    python prepare_mc_data.py \
        --csv_path /path/to/vqa.dataset.csv \
        --videos_dir /path/to/videos \
        --output_dir /path/to/prepared \
        --speech_dir /path/to/speech  # Optional
"""

import argparse
import json
import csv
import os
from datetime import datetime
from pathlib import Path


def load_speech_recognition_data(speech_dir_path):
    """Load speech recognition data from the whisper directory."""
    speech_data = {}
    
    if speech_dir_path and os.path.exists(speech_dir_path):
        for filename in os.listdir(speech_dir_path):
            if filename.endswith('.json'):
                video_id = filename.replace('.json', '')
                try:
                    with open(os.path.join(speech_dir_path, filename), 'r', encoding='utf-8') as f:
                        speech_data[video_id] = json.load(f)
                except Exception as e:
                    print(f"Error loading speech data for {video_id}: {e}")
    elif speech_dir_path:
        print(f"Speech recognition directory not found: {speech_dir_path}")
    
    return speech_data


def check_video_exists(video_path):
    """Check if video file exists."""
    return os.path.exists(video_path)


def create_mc_options(options):
    """Create MC options list maintaining original order."""
    mc_options = []
    for rank, option in enumerate(options, 1):
        mc_options.append({
            "rank": rank,
            "option": option
        })
    
    return mc_options


def prepare_mc_data(csv_path, videos_dir, output_dir, speech_dir=None, relative_video_path=None):
    """
    Prepare MC data from CSV to individual JSON files.
    
    Args:
        csv_path: Path to the CSV file with questions and choices
        videos_dir: Path to the directory containing video files
        output_dir: Path to output directory for JSON files
        speech_dir: Optional path to speech recognition results
        relative_video_path: Optional relative path prefix for video paths in JSON
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load speech recognition data if available
    speech_data = {}
    if speech_dir:
        print("Loading speech recognition data...")
        speech_data = load_speech_recognition_data(speech_dir)
        print(f"Loaded speech data for {len(speech_data)} videos")
    
    # Read and process CSV
    print(f"Processing CSV data from: {csv_path}")
    processed_count = 0
    skipped_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        for row in csv_reader:
            # Skip header or malformed rows
            if len(row) < 6:
                print(f"Skipping malformed row: {row}")
                skipped_count += 1
                continue
            
            # Handle different CSV formats
            if len(row) == 7:
                # Format: q_id, video_id, question, choice_1, choice_2, choice_3, choice_4
                q_id, video_id, question, choice_1, choice_2, choice_3, choice_4 = row
                correct_answer = None  # Test data - no answer
            elif len(row) == 8:
                # Format: q_id, video_id, question, choice_1, choice_2, choice_3, choice_4, correct_answer
                q_id, video_id, question, choice_1, choice_2, choice_3, choice_4, correct_answer = row
            else:
                # Try to parse first 7 columns
                q_id = row[0]
                video_id = row[1]
                question = row[2]
                choice_1, choice_2, choice_3, choice_4 = row[3], row[4], row[5], row[6] if len(row) > 6 else ""
                correct_answer = row[7] if len(row) > 7 else None
            
            try:
                q_id = int(q_id)
            except ValueError:
                # Skip header row or invalid q_id
                if q_id.lower() == 'q_id' or q_id.lower() == 'id':
                    continue
                print(f"Skipping row with invalid q_id: {q_id}")
                skipped_count += 1
                continue
            
            # Construct video path
            if relative_video_path:
                video_path = f"{relative_video_path}/{video_id}.mp4"
            else:
                video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            
            absolute_video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            
            # Check if video exists
            video_exists = check_video_exists(absolute_video_path)
            
            # Prepare choices list in original order
            choices = [choice_1, choice_2, choice_3, choice_4]
            choices = [c for c in choices if c]  # Filter empty choices
            
            # Create MC options maintaining original order
            mc_options = create_mc_options(choices)
            
            # Get speech recognition data
            if video_id in speech_data:
                speech_recognition = speech_data[video_id]
            else:
                # Default speech recognition structure if not found
                speech_recognition = {
                    "recognized_text": "",
                    "language": "en",
                    "model_name": "whisper-large-v3",
                    "device_used": "cuda",
                    "segments": [],
                    "confidence": None
                }
            
            # Create the data structure
            data = {
                "q_id": q_id,
                "video_id": video_id,
                "video_path": video_path,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "question": question,
                "choices": choices,
                "mc_options": mc_options,
                "total_options": len(choices),
                "timestamp": datetime.now().isoformat() + "Z",
                "video_exists": video_exists,
                "speech_recognition": speech_recognition
            }
            
            # Add correct answer if available (training data)
            if correct_answer:
                data["correct_answer"] = correct_answer
            
            # Write to JSON file
            output_file = os.path.join(output_dir, f"{video_id}.json")
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, indent=2, ensure_ascii=False)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} questions...")
    
    print(f"\nData preparation completed!")
    print(f"Processed: {processed_count} questions")
    print(f"Skipped: {skipped_count} rows")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Multiple Choice data for VQA inference"
    )
    parser.add_argument(
        "--csv_path", "-c",
        required=True,
        help="Path to CSV file with questions and choices"
    )
    parser.add_argument(
        "--videos_dir", "-v",
        required=True,
        help="Path to directory containing video files"
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Output directory for prepared JSON files"
    )
    parser.add_argument(
        "--speech_dir", "-s",
        default=None,
        help="Optional: Path to speech recognition results directory"
    )
    parser.add_argument(
        "--relative_video_path", "-r",
        default=None,
        help="Optional: Relative path prefix for video paths in output JSON"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    if not os.path.exists(args.videos_dir):
        print(f"Warning: Videos directory not found: {args.videos_dir}")
    
    prepare_mc_data(
        csv_path=args.csv_path,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        speech_dir=args.speech_dir,
        relative_video_path=args.relative_video_path
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
