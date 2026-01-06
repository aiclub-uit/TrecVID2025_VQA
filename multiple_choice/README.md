# Multiple Choice (MC) Task

This module handles Multiple Choice Video Question Answering, where the model ranks options from most to least likely correct.

## Features

- **Ranking-based Inference**: Rank all options (A, B, C, D) from most to least likely
- **Multiple Backend Support**: VLLM, Transformers, and llama.cpp HTTP server
- **Ensemble Voting**: Combine predictions using Borda, Plurality, Approval, or Condorcet methods
- **Audio Integration**: Include speech transcription for multimodal understanding

## Directory Structure

```
multiple_choice/
├── mc_infer.py           # Main inference script
├── eval.py               # Evaluation with ensemble voting
├── create_submission.py  # Submission file generator
├── models/
│   ├── __init__.py
│   ├── vllm.py           # VLLM backend
│   ├── transformers.py   # Transformers backend
│   └── llama_model.py    # llama.cpp HTTP server backend
├── libs/
│   ├── __init__.py
│   └── utils.py          # Utility functions
├── configs/
│   ├── internvl3_5_transformers.yaml
│   ├── internvl3_5_vllm.yaml
│   └── llama_cpp_server.yaml
└── example_data/
    └── sample.json       # Example data format
```

## Quick Start

### 1. Prepare Data

Create JSON files for each video sample:

```json
{
  "video_id": "example_video",
  "video_path": "/path/to/video.mp4",
  "question": "What is happening in the video?",
  "choices": [
    "A person is cooking",
    "A person is dancing",
    "A person is reading",
    "A person is sleeping"
  ],
  "correct_answer": "A person is cooking",
  "speech_recognition": {
    "recognized_text": "Optional audio transcription..."
  }
}
```

### 2. Run Inference

```bash
python mc_infer.py \
  --config configs/internvl3_5_transformers.yaml \
  --data_dir /path/to/prepared_data \
  --output /path/to/results
```

### 3. Evaluate Results

```bash
# Single folder evaluation
python eval.py /path/to/results --output evaluations/

# Ensemble evaluation with multiple models
python eval.py --ensemble \
  --ensemble-paths results/model1 results/model2 results/model3 \
  --output ensemble_eval/
```

### 4. Create Submission

```bash
python create_submission.py \
  --results results/ \
  --csv test_dataset.csv \
  --output submission.csv
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Percentage of correct first-place predictions |
| **Top-k Accuracy** | Percentage where correct answer is in top k positions |
| **MRR** | Mean Reciprocal Rank - average of 1/rank |
| **Mean Rank** | Average position of correct answer |

## Ensemble Voting Methods

- **Borda Count**: Points based on ranking position (most democratic)
- **Plurality**: First-place votes only (winner-takes-all)
- **Approval**: Top-2 positions as "approved"
- **Condorcet**: Pairwise comparisons between options

## Supported Models

| Model | Backend | Notes |
|-------|---------|-------|
| InternVL3.5 (8B/26B/38B) | Transformers, VLLM | Recommended |
| Qwen2.5-VL | Transformers | Good balance |
| LLaMA 3.2 Vision | llama.cpp | Resource-efficient |
