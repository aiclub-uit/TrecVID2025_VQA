import argparse
import csv
import json
import os
from pytubefix import YouTube
from pathlib import Path
import tempfile
import subprocess


def detect_video_codec(video_path):
    """
    Detect the video codec of a file using ffprobe.
    Returns the codec name or None if detection fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def is_problematic_codec(codec_name):
    """
    Check if a codec is known to cause compatibility issues.
    """
    # include both 'av1' and 'av01' as ffprobe may report either
    problematic_codecs = ["av1", "av01", "vp9", "vp8"]  # AV1, VP9, VP8 can cause issues
    return codec_name and codec_name.lower() in problematic_codecs


def save_video_metadata(yt, video_id, output_dir):
    """
    Saves video metadata (title, description, etc.) to a JSON file.
    """
    try:
        metadata = {
            "video_id": video_id,
            "title": yt.title,
            "description": yt.description,
            "author": yt.author,
            "length": yt.length,
            "views": yt.views,
            "publish_date": yt.publish_date.isoformat() if yt.publish_date else None,
            "keywords": yt.keywords,
            "thumbnail_url": yt.thumbnail_url,
        }

        metadata_filename = f"{video_id}.json"
        metadata_path = os.path.join(output_dir, metadata_filename)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Saved metadata to: {metadata_path}")
        return True

    except Exception as e:
        print(f"Error saving metadata for {video_id}: {e}")
        return False


def reencode_progressive_if_needed(video_path: Path):
    """Detect codec of a just-downloaded progressive file and re-encode to H.264 MP4 if problematic.
    Returns path to (possibly re-encoded) file.
    """
    codec = detect_video_codec(video_path)
    if not is_problematic_codec(codec):
        return video_path
    print(f"Progressive stream uses {codec} codec; re-encoding to H.264 for compatibility …")
    tmp_out = video_path.with_suffix(".h264_tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(tmp_out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"Re-encode failed, leaving original file. ffmpeg stderr:\n{proc.stderr}")
        return video_path
    # Replace original
    try:
        os.replace(tmp_out, video_path)
    except Exception as e:
        print(f"Could not replace original file after re-encode: {e}")
        return video_path
    print(f"Re-encoded and replaced original file: {video_path}")
    return video_path


def download_youtube_video(video_url, output_dir, filename, video_id):
    """
    Download the highest-available resolution by preferring adaptive streams
    (video-only + audio-only) and muxing them with ffmpeg. Falls back to
    progressive MP4 when appropriate.
    """
    try:
        print(f"Connecting to YouTube with URL: {video_url}")
        yt = YouTube(video_url)

        # Create subfolders for videos and metadata
        videos_dir = os.path.join(output_dir, "videos")
        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        # 1) Try best progressive first (fast, includes audio)
        prog = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
        )

        # 2) Find best adaptive video and audio (usually higher than progressive)
        best_video = (
            yt.streams.filter(adaptive=True, only_video=True, mime_type="video/mp4")
            .order_by("resolution")
            .desc()
            .first()
        ) or (
            yt.streams.filter(adaptive=True, only_video=True, mime_type="video/webm")
            .order_by("resolution")
            .desc()
            .first()
        )

        best_audio = (
            yt.streams.filter(adaptive=True, only_audio=True, mime_type="audio/mp4")
            .order_by("abr")
            .desc()
            .first()
        ) or (
            yt.streams.filter(adaptive=True, only_audio=True, mime_type="audio/webm")
            .order_by("abr")
            .desc()
            .first()
        )

        # Decide strategy: if adaptive video exists and is higher than progressive, use adaptive
        use_adaptive = bool(
            best_video and (
                not prog or best_video.resolution != getattr(prog, "resolution", None)
            )
        )

        if not use_adaptive and prog:
            print(f"Downloading progressive stream at {prog.resolution} …")
            out_path = os.path.join(
                videos_dir,
                filename if filename.endswith(".mp4") else f"{Path(filename).stem}.mp4",
            )
            prog.download(output_path=videos_dir, filename=os.path.basename(out_path))
            # New: re-encode if the progressive file uses an incompatible codec (AV1/VP9/VP8)
            reencode_progressive_if_needed(Path(out_path))
            print(f"Saved: {out_path}")
            save_video_metadata(yt, video_id, metadata_dir)
            return True

        if not best_video or not best_audio:
            # No adaptive combo available, but we may still have progressive
            if prog:
                print("Adaptive streams not available; falling back to progressive.")
                out_path = os.path.join(
                    videos_dir,
                    (
                        filename
                        if filename.endswith(".mp4")
                        else f"{Path(filename).stem}.mp4"
                    ),
                )
                prog.download(
                    output_path=videos_dir, filename=os.path.basename(out_path)
                )
                # New: re-encode if needed
                reencode_progressive_if_needed(Path(out_path))
                print(f"Saved: {out_path}")
                save_video_metadata(yt, video_id, metadata_dir)
                return True
            print(f"Error: No suitable streams found for {video_url}.")
            return False

        print(
            f"Downloading adaptive video {best_video.resolution} ({best_video.mime_type}) and best audio ({best_audio.mime_type}) …"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            v_path = Path(tmpdir) / f"video.{best_video.subtype}"
            a_path = Path(tmpdir) / f"audio.{best_audio.subtype}"
            best_video.download(output_path=tmpdir, filename=v_path.name)
            best_audio.download(output_path=tmpdir, filename=a_path.name)

            # Check if video has problematic codec
            video_codec = detect_video_codec(v_path)
            force_reencode = is_problematic_codec(video_codec)

            if force_reencode:
                print(
                    f"Detected {video_codec} codec - will re-encode to H.264 for compatibility"
                )

            # Choose container and encoding strategy:
            # - If both are mp4 and no problematic codec -> output mp4 (faststart)
            # - Otherwise -> output mkv to stream-copy without transcoding headaches
            # - If problematic codec detected -> force re-encode to H.264
            both_mp4 = best_video.subtype == "mp4" and best_audio.subtype == "mp4"
            ext = "mp4" if both_mp4 else "mkv"
            # Force MP4 output for problematic codecs to ensure compatibility
            if force_reencode:
                ext = "mp4"

            out_path = os.path.join(
                videos_dir,
                (
                    filename
                    if filename.endswith(f".{ext}")
                    else f"{Path(filename).stem}.{ext}"
                ),
            )

            # ffmpeg command
            if force_reencode:
                # Force re-encode with H.264 for problematic codecs
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(v_path),
                    "-i",
                    str(a_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    out_path,
                ]
            elif both_mp4:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(v_path),
                    "-i",
                    str(a_path),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    out_path,
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(v_path),
                    "-i",
                    str(a_path),
                    "-c",
                    "copy",
                    out_path,
                ]

            print("Merging with ffmpeg …")
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if proc.returncode != 0:
                print(
                    "ffmpeg failed; trying safe fallback (re-encode to H.264 + AAC in MP4)."
                )
                out_path = os.path.join(videos_dir, f"{Path(filename).stem}.mp4")
                cmd_fallback = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(v_path),
                    "-i",
                    str(a_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    out_path,
                ]
                proc2 = subprocess.run(
                    cmd_fallback,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if proc2.returncode != 0:
                    print(f"ffmpeg error:\n{proc.stderr}\n{proc2.stderr}")
                    print("Trying final fallback with more compatible settings...")
                    # Final fallback: force H.264 baseline profile for maximum compatibility
                    cmd_final = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(v_path),
                        "-i",
                        str(a_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-profile:v",
                        "baseline",
                        "-level",
                        "3.0",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "128k",
                        "-movflags",
                        "+faststart",
                        out_path,
                    ]
                    proc3 = subprocess.run(
                        cmd_final,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if proc3.returncode != 0:
                        print(
                            f"All ffmpeg attempts failed:\n{proc.stderr}\n{proc2.stderr}\n{proc3.stderr}"
                        )
                        return False

            print(f"Saved: {out_path}")

        # Save metadata
        save_video_metadata(yt, video_id, metadata_dir)
        return True

    except Exception as e:
        print(f"An unexpected error occurred for URL {video_url}: {e}")
        return False


def main():
    """
    Main function to parse arguments, read CSV, and download videos.
    """
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube based on a CSV file."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="videos",
        help="Output directory for downloaded videos (default: 'videos').",
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created directory: {args.output_dir}")

    # Read the CSV file
    try:
        with open(args.csv_file, "r", newline="", encoding="utf-8") as csvfile:
            # Using csv.reader and trimming whitespace for robustness
            csv_reader = csv.reader(row for row in csvfile if row)  # Skips empty rows
            for row in csv_reader:
                if len(row) > 1:
                    # The video ID is the second column in the CSV
                    video_id = row[1].strip()

                    # Construct the standard YouTube URL from the video ID.
                    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

                    output_video_filename = f"{video_id}.mp4"

                    # Proceed to download with the correct URL
                    download_youtube_video(
                        youtube_url, args.output_dir, output_video_filename, video_id
                    )

                print("-" * 20)  # Separator for clarity

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at: {args.csv_file}")
    except Exception as e:
        print(f"An error occurred while processing the CSV file: {e}")


if __name__ == "__main__":
    main()

# python src/download.py train/train.2025.csv train/metadata/ train/videos
