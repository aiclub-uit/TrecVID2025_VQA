# Multiple Choice Video Question Answering (MC-VQA)

A comprehensive framework for Multiple Choice Video Question Answering using Vision-Language Models (VLMs). This system supports multiple inference backends and includes tools for evaluation and submission generation.

## Features

- **Multiple Backend Support**: VLLM, Transformers, and llama.cpp HTTP server
- **Ranking-based Inference**: Rank all options from most to least likely
- **Audio Transcription Support**: Incorporate speech recognition for multimodal understanding
- **Comprehensive Evaluation**: Top-k accuracy, MRR, ensemble voting methods
- **Submission Generation**: Generate competition-ready submission files

## Supported Models

| Model | Backend | Notes |
|-------|---------|-------|
| InternVL3.5 (8B/26B/38B) | Transformers, VLLM | Recommended for best performance |
| Qwen2.5-VL | Transformers | Good balance of speed and accuracy |
| LLaMA 3.2 Vision | llama.cpp | For resource-constrained environments |
| Any llama.cpp compatible model | llama.cpp server | Via HTTP API |

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU inference)
- FFmpeg (for video processing)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/mc-video-qa.git
cd mc-video-qa

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt

# Optional: Install VLLM for faster inference
pip install vllm
```

## Quick Start

### 1. Prepare Your Data

Create JSON files for each video sample in a directory:

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

### 2. Configure the Model

Edit a configuration file in `configs/`:

```yaml
model_type: "transformers"  # or "vllm", "llamacpp"
model_name: "/path/to/your/model"

transformers_config:
  torch_dtype: "bfloat16"
  device_map: "auto"
  trust_remote_code: true

mc_config:
  max_frames: 16
  approach: "ranking"
```

### 3. Run Inference

```bash
python mc_infer.py \
  --config configs/internvl3_5_transformers.yaml \
  --data_dir /path/to/prepared_data \
  --output /path/to/results
```

### 4. Evaluate Results

```bash
# Single folder evaluation
python eval.py /path/to/results --output evaluations/

# Ensemble evaluation with multiple models
python eval.py --ensemble \
  --ensemble-paths results/model1 results/model2 results/model3 \
  --output ensemble_eval/
```

### 5. Generate Submission

```bash
python create_submission.py \
  --results results/ \
  --csv test_dataset.csv \
  --output submission.csv
```

## Configuration Guide

### Model Types

| `model_type` | Description | When to Use |
|--------------|-------------|-------------|
| `transformers` | Hugging Face Transformers | Standard inference, good flexibility |
| `vllm` | VLLM inference engine | High throughput, batch processing |
| `llamacpp` | llama.cpp HTTP server | Resource-constrained, GGUF models |

### Key Configuration Options

```yaml
# Model configuration
model_type: "transformers"
model_name: "/path/to/model"
model_format: "internvl"  # Optional: internvl, qwen, llama

# Transformers-specific
transformers_config:
  torch_dtype: "bfloat16"
  device_map: "auto"
  trust_remote_code: true
  max_num_tiles: 12  # For InternVL

# VLLM-specific
vllm_config:
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.9
  max_model_len: 32000

# Multiple choice settings
mc_config:
  max_frames: 16
  frame_sampling: "uniform"
  approach: "ranking"
  prompt_template: |
    Your custom prompt template...
```

## Data Format

### Input Data (JSON)

```json
{
  "video_id": "unique_video_identifier",
  "video_path": "/absolute/path/to/video.mp4",
  "question": "What does the person do at the end of the video?",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "Option A",
  "speech_recognition": {
    "recognized_text": "Audio transcription text..."
  }
}
```

### Output Result (JSON)

```json
{
  "video_id": "unique_video_identifier",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "generated_text": "B A D C",
  "evaluation": {
    "predicted_ranking": ["B", "A", "D", "C"],
    "correct_letter": "A",
    "top1_correct": false,
    "reciprocal_rank": 0.5
  },
  "inference_time": 2.5
}
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Top-1 Accuracy** | Percentage of correct first-place predictions |
| **Top-k Accuracy** | Percentage where correct answer is in top k positions |
| **MRR** | Mean Reciprocal Rank - average of 1/rank |
| **Mean Rank** | Average position of correct answer |

### Ensemble Voting Methods

- **Borda Count**: Points based on ranking position
- **Plurality**: First-place votes only
- **Approval**: Top-2 positions as "approved"
- **Condorcet**: Pairwise comparisons

## Project Structure

```
.
├── mc_infer.py              # Main inference script
├── eval.py                  # Evaluation and ensemble
├── create_submission.py     # Submission generator
├── requirements.txt         # Python dependencies
├── models/
│   ├── __init__.py
│   ├── vllm.py              # VLLM backend
│   ├── transformers.py      # Transformers backend
│   └── llama_model.py       # llama.cpp backend
├── libs/
│   ├── __init__.py
│   └── utils.py             # Utilities
├── configs/
│   └── *.yaml               # Configuration files
└── example_data/
    └── sample.json          # Example data format
```

## Tips for Best Results

1. **Frame Count**: Use 12-16 frames for most videos; increase for longer videos
2. **Model Size**: Larger models (38B) give better accuracy but require more VRAM
3. **Audio**: Include audio transcription when available for better understanding
4. **Ensemble**: Combine predictions from multiple models for improved accuracy

## Troubleshooting

### Common Issues

**Out of Memory**
```bash
# Use 8-bit quantization
transformers_config:
  load_in_8bit: true

# Or reduce frame count
mc_config:
  max_frames: 8
```

**VLLM Not Found**
```bash
pip install vllm
# Requires CUDA 11.8+
```

**Video Loading Fails**
```bash
# Install FFmpeg
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # Mac
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mc_video_qa,
  title = {Multiple Choice Video Question Answering},
  year = {2025},
  url = {https://github.com/your-username/mc-video-qa}
}
```

## Acknowledgments

- [InternVL](https://github.com/OpenGVLab/InternVL) for the vision-language model
- [VLLM](https://github.com/vllm-project/vllm) for high-throughput inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient CPU/GPU inference
