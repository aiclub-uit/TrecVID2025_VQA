import csv
import json
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--videos_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Reading CSV from {args.csv_path}")
    print(f"Videos dir: {args.videos_dir}")
    print(f"Output dir: {args.output_dir}")

    processed = 0
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3: 
                continue
            
            # Format: ID, VideoID, Question
            q_id = row[0].strip()
            video_id = row[1].strip()
            question = row[2].strip()
            
            # Check video existence
            video_extensions = ['.mp4', '.mkv', '.webm', '.avi']
            final_video_path = None
            
            # Check absolute path
            for ext in video_extensions:
                p = os.path.join(args.videos_dir, f"{video_id}{ext}")
                if os.path.exists(p):
                    final_video_path = os.path.abspath(p)
                    break
            
            if not final_video_path:
                # Fallback to assumming mp4
                final_video_path = os.path.abspath(os.path.join(args.videos_dir, f"{video_id}.mp4"))
            
            data = {
                "q_id": q_id,
                "video_id": video_id,
                "video_path": final_video_path,
                "question": question,
                "video_exists": os.path.exists(final_video_path),
                "file_stem": video_id,
                "correct_answer": ""
            }
            
            output_file = os.path.join(args.output_dir, f"{video_id}.json")
            with open(output_file, 'w', encoding='utf-8') as out:
                json.dump(data, out, indent=2, ensure_ascii=False)
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} items")

    print(f"Finished. Processed {processed} items.")

if __name__ == "__main__":
    main()
