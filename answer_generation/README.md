# Answer Generation (AG) Module

This module handles open-ended Video Question Answering, generating natural language answers to questions about video content.

## Features

- **Multi-modal Understanding**: Combines visual and audio (speech transcription) information
- **Multiple Answer Generation**: Generates diverse answers for ensemble submission
- **Enhanced Prompting**: Video context generation for improved understanding
- **Flexible Backends**: Supports VLLM and Transformers

## Directory Structure

```
answer_generation/
├── infer.py              # Main inference script
├── eval.py               # Evaluation script
├── prepare.py            # Data preparation with speech recognition
├── create_submission.py  # Submission file generator
├── prompt_builder.py     # Prompt template definitions
├── models/
│   ├── __init__.py
│   ├── base_vqa.py       # Base model class
│   ├── enhanced_vqa.py   # Enhanced model with context
│   ├── vllm.py           # VLLM backend
│   ├── transformers.py   # Transformers backend
│   └── utils/
│       ├── utils.py
│       ├── prompt_handler.py
│       └── prompt_config_manager.py
├── libs/
│   ├── __init__.py
│   ├── evaluation_metrics.py
│   ├── data_loader.py
│   ├── result_saver.py
│   └── ...
└── configs/
    ├── videollama3.yaml
    └── qwen25_vl_vllm.yaml
```

## Quick Start

### 1. Prepare Data

```bash
python prepare.py \
  --metadata_folder /path/to/metadata \
  --videos_folder /path/to/videos \
  --output_folder prepared/AG_data \
  --enable_speech_recognition \
  --asr_model whisper \
  --whisper_model large-v3
```

### 2. Run Inference

```bash
python infer.py \
  --config configs/videollama3.yaml \
  --batch \
  --video_path /path/to/videos \
  --prepared_folder prepared/AG_data \
  --output results/my_experiment
```

### 3. Evaluate Results

```bash
python eval.py \
  --input results/my_experiment \
  --output evaluations/
```

### 4. Create Submission

```bash
python create_submission.py \
  --results_dirs results/my_experiment \
  --output submission.csv
```

## Data Format

### Input (Prepared JSON)

```json
{
  "video_id": "example_video",
  "video_path": "/path/to/video.mp4",
  "question": "What is the speaker explaining?",
  "answer": "Ground truth answer (for evaluation)",
  "speech_recognition": {
    "recognized_text": "Audio transcription...",
    "language": "en",
    "confidence": 0.95
  }
}
```

### Output (Result JSON)

```json
{
  "video_id": "example_video",
  "question": "What is the speaker explaining?",
  "predicted_answer": "The speaker is explaining...",
  "ground_truth_answer": "Ground truth answer",
  "metrics": {
    "bleu_score": 0.45,
    "rouge_l": 0.62,
    "bert_score": 0.78
  }
}
```

## Supported Models

| Model | Backend | Notes |
|-------|---------|-------|
| VideoLLaMA3 | Transformers | Best for video understanding |
| Qwen2.5-VL | VLLM, Transformers | Good balance of speed/quality |

## Configuration Options

### Enhanced Prompting

```yaml
enhanced_prompting:
  enabled: true
  video_context_generation: true   # Generate video description first
  use_audio_visual_enhanced: true  # Include audio transcription
```

### Multiple Answers

```yaml
multiple_answers:
  num_iterations: 10  # Generate 10 diverse answers
```

### Prompt Templates

- `visual_only`: Only video frames
- `audio_visual`: Video + audio transcription
- `enhanced_audio_visual`: Video + audio + generated context

## Speech Recognition

Supports two ASR backends:

1. **Whisper** (Recommended)
   ```yaml
   --asr_model whisper --whisper_model large-v3
   ```

2. **Seamless M4T**
   ```yaml
   --asr_model seamless-m4t
   ```
