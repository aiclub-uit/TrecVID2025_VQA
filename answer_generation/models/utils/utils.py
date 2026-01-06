"""
Shared utilities for VQA models.
"""

import cv2
import os
from PIL import Image
from typing import List, Optional, Union


def video_to_frames(
    video_path: str, num_frames: int = 16, return_type: str = "pil"
) -> Optional[List[Union[Image.Image, any]]]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        return_type: Format of returned frames ("pil" or "opencv")

    Returns:
        List of frames in the specified format, or None if extraction fails
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            if return_type == "pil":
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                frames.append(frame)

    cap.release()

    if not frames:
        return None

    return frames


def ensure_cache_dir(cache_dir: str) -> str:
    """
    Ensure cache directory exists and return absolute path.

    Args:
        cache_dir: Cache directory path

    Returns:
        Absolute path to cache directory
    """
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(cache_dir)
        print(f"Cache directory is not absolute, using absolute path: {cache_dir}")

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir


def detect_model_format(model_name: str) -> str:
    """
    Detect model format for prompt optimization.

    Args:
        model_name: Name of the model

    Returns:
        Model format string
    """
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        return "qwen"
    elif "aria" in model_name_lower:
        return "aria"
    elif "llava" in model_name_lower:
        return "llava"
    else:
        return "general"


def create_image_gallery(images, columns=3, spacing=20, bg_color=(200, 200, 200)):
    """
    Combine multiple images into a single larger image in a grid format.

    Parameters:
        image_paths (list of str): List of file paths to the images to display.
        columns (int): Number of columns in the gallery.
        spacing (int): Space (in pixels) between the images in the gallery.
        bg_color (tuple): Background color of the gallery (R, G, B).

    Returns:
        PIL.Image: A single combined image.
    """
    # Open all images and get their sizes
    img_width, img_height = images[0].size  # Assuming all images are of the same size

    # Calculate rows needed for the gallery
    rows = (len(images) + columns - 1) // columns

    # Calculate the size of the final gallery image
    gallery_width = columns * img_width + (columns - 1) * spacing
    gallery_height = rows * img_height + (rows - 1) * spacing

    # Create a new image with the calculated size and background color
    gallery_image = Image.new("RGB", (gallery_width, gallery_height), bg_color)

    # Paste each image into the gallery
    for index, img in enumerate(images):
        row = index // columns
        col = index % columns

        x = col * (img_width + spacing)
        y = row * (img_height + spacing)

        gallery_image.paste(img, (x, y))

    return gallery_image


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <video_path> [num_frames]")
        print("Example: python utils.py /path/to/video.mp4 32")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    frames = video_to_frames(video_path, num_frames)

    if frames:
        for i, frame in enumerate(frames):
            frame.save(f"frame_{i}.jpg")
        print(f"Extracted {len(frames)} frames.")
    else:
        print("Failed to extract frames from video.")
