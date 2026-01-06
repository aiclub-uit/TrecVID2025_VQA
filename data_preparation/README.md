# Data Preparation

Scripts for downloading, processing, and preparing data for the VQA tasks.

## Scripts

### download.py
Download videos from YouTube based on a CSV file.

```bash
python download.py train.csv output_metadata/ output_videos/
```

**CSV Format:**
```
video_id,question,answer,...
```

**Features:**
- Downloads highest quality available
- Muxes separate audio/video streams
- Re-encodes problematic codecs (AV1, VP9) to H.264
- Saves video metadata as JSON

### reencode_av1_videos.py
Scan and re-encode videos with problematic codecs (AV1, VP9, VP8) to H.264.

```bash
# Scan only (dry run)
python reencode_av1_videos.py /path/to/videos --dry-run

# Scan and re-encode
python reencode_av1_videos.py /path/to/videos

# Re-encode without backup
python reencode_av1_videos.py /path/to/videos --no-backup
```

### prepare_mc_data.py
Prepare Multiple Choice data from CSV to individual JSON files.

```bash
python prepare_mc_data.py \
  --csv_path /path/to/vqa.dataset.csv \
  --videos_dir /path/to/videos \
  --output_dir /path/to/prepared \
  --speech_dir /path/to/speech  # Optional
```

### prepare_ag_csv.py
Prepare Answer Generation data from CSV to individual JSON files.

```bash
python prepare_ag_csv.py \
  --csv_path /path/to/vqa.dataset.csv \
  --videos_dir /path/to/videos \
  --output_dir /path/to/prepared
```

**Input CSV Format:**
```
q_id,video_id,question,choice1,choice2,choice3,choice4[,correct_answer]
```

**Output JSON Format:**
```json
{
  "q_id": 1,
  "video_id": "abc123",
  "video_path": "/path/to/video.mp4",
  "question": "What happens in the video?",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "Option A",
  "speech_recognition": {...}
}
```

## Data Directory Structure

After downloading and preparation:

```
data/
├── train/
│   ├── metadata/           # Video metadata JSON files
│   │   ├── video_id1.json
│   │   └── ...
│   └── videos/             # Video files
│       ├── video_id1.mp4
│       └── ...
├── test/
│   ├── metadata/
│   └── videos/
└── prepared/
    ├── MC_train/           # Prepared MC data (JSON per video)
    └── MC_test/
```

## Requirements

- `pytubefix` for YouTube downloads
- `ffmpeg` for video processing
- Standard Python libraries (json, csv, os, subprocess)

Install:
```bash
pip install pytubefix
# FFmpeg must be installed system-wide
sudo apt install ffmpeg  # Ubuntu
```
