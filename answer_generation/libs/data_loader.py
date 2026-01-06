"""Data loading and preparation utilities for Video Question Answering."""

import json
import os
import cv2
from pathlib import Path
from typing import Dict, List


def load_prepared_data(prepared_folder: Path) -> List[Dict]:
    """Load all prepared JSON files from folder."""
    prepared_files = list(prepared_folder.glob("*.json"))
    data = []

    for file_path in prepared_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                item = json.load(f)
                item["file_stem"] = file_path.stem
                data.append(item)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return data


def get_video_info(video_path: str) -> Dict:
    """Get basic video information for reproducibility."""
    if not video_path or not os.path.exists(video_path):
        return {"error": "Video file not found"}

    try:
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        # Get file size
        file_size = os.path.getsize(video_path)

        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "resolution": f"{width}x{height}",
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
        }
    except Exception as e:
        return {"error": f"Could not get video info: {str(e)}"}
