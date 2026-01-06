#!/usr/bin/env python3
"""
Script to detect and re-encode AV1 and other problematic codec videos to H.264.
This will fix compatibility issues with video processing libraries.
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path


def detect_video_codec(video_path):
    """
    Detect the video codec of a file using ffprobe.
    Returns the codec name or None if detection fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        print(f"Error detecting codec for {video_path}: {e}")
        return None


def is_problematic_codec(codec_name):
    """
    Check if a codec is known to cause compatibility issues.
    """
    problematic_codecs = ["av01", "vp9", "vp8"]  # AV1, VP9, VP8 can cause issues
    return codec_name and codec_name.lower() in problematic_codecs


def reencode_video(input_path, output_path=None, backup=True):
    """
    Re-encode a video to H.264/AAC MP4 format.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video (defaults to input path with _h264 suffix)
        backup: Whether to backup the original file
    
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_h264.mp4"
    else:
        output_path = Path(output_path)
    
    # Create backup if requested
    backup_path = None
    if backup:
        backup_path = input_path.parent / f"{input_path.stem}_original{input_path.suffix}"
    
    # Re-encode command with multiple fallback strategies
    commands = [
        # First try: fast preset with good quality
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ],
        # Second try: baseline profile for maximum compatibility
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path)
        ],
        # Third try: very conservative settings
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "28",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-movflags", "+faststart",
            str(output_path)
        ]
    ]
    
    print(f"Re-encoding {input_path} to H.264...")
    
    success = False
    for i, cmd in enumerate(commands, 1):
        print(f"Attempt {i}/{len(commands)}")
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=300  # 5 minute timeout per attempt
            )
            
            if result.returncode == 0 and output_path.exists():
                print(f"Successfully re-encoded to {output_path}")
                success = True
                break
            else:
                print(f"Attempt {i} failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"Attempt {i} timed out")
        except Exception as e:
            print(f"Attempt {i} failed with exception: {e}")
    
    if success:
        # Create backup and replace original if requested
        if backup and backup_path:
            shutil.move(str(input_path), str(backup_path))
            print(f"Original backed up to {backup_path}")
        
        # Replace original with re-encoded version
        if backup:
            shutil.move(str(output_path), str(input_path))
            print(f"Replaced {input_path} with re-encoded version")
        
        return True
    else:
        print(f"Failed to re-encode {input_path}")
        # Clean up failed output file
        if output_path.exists():
            output_path.unlink()
        return False


def scan_directory(directory):
    """
    Scan a directory for video files with problematic codecs.
    """
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}
    problematic_videos = []
    
    directory = Path(directory)
    print(f"Scanning {directory} for video files...")
    
    for video_file in directory.rglob('*'):
        if video_file.suffix.lower() in video_extensions and video_file.is_file():
            print(f"Checking {video_file}...", end=' ')
            codec = detect_video_codec(video_file)
            if codec:
                print(f"codec: {codec}")
                if is_problematic_codec(codec):
                    problematic_videos.append((video_file, codec))
                    print(f"  -> PROBLEMATIC: {codec}")
            else:
                print("codec: unknown")
    
    return problematic_videos


def main():
    parser = argparse.ArgumentParser(
        description="Detect and re-encode videos with problematic codecs (AV1, VP9, VP8) to H.264"
    )
    parser.add_argument(
        "directory", 
        help="Directory to scan for problematic videos"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Only scan and report, don't re-encode"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true", 
        help="Don't backup original files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode all videos found, not just problematic ones"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        return 1
    
    # Scan for problematic videos
    problematic_videos = scan_directory(args.directory)
    
    if not problematic_videos:
        print("No videos with problematic codecs found!")
        return 0
    
    print(f"\nFound {len(problematic_videos)} videos with problematic codecs:")
    for video_path, codec in problematic_videos:
        print(f"  {video_path} ({codec})")
    
    if args.dry_run:
        print(f"\nDry run complete. Would re-encode {len(problematic_videos)} videos.")
        return 0
    
    # Re-encode problematic videos
    print(f"\nRe-encoding {len(problematic_videos)} videos...")
    success_count = 0
    
    for video_path, codec in problematic_videos:
        print(f"\n--- Processing {video_path} ---")
        if reencode_video(video_path, backup=not args.no_backup):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(problematic_videos)} videos successfully re-encoded")
    
    if success_count < len(problematic_videos):
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
