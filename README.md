# TrecVID 2025 Video Question Answering

A comprehensive framework for Video Question Answering (VQA) supporting both **Multiple Choice (MC)** and **Answer Generation (AG)** tasks. Built using Vision-Language Models (VLMs) with multiple inference backends.

## Repository Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
│
├── multiple_choice/         # Multiple Choice (MC) Task
│   ├── README.md            # MC documentation
│   ├── mc_infer.py          # MC inference script
│   ├── eval.py              # MC evaluation and ensemble
│   ├── create_submission.py # MC submission generator
│   ├── models/              # VLLM, Transformers, llama.cpp backends
│   ├── libs/                # Utilities
│   ├── configs/             # Configuration examples
│   └── example_data/        # Example data format
│
└── answer_generation/       # Answer Generation (AG) Task
    ├── README.md            # AG documentation
    ├── infer.py             # AG inference script
    ├── eval.py              # AG evaluation
    ├── prepare.py           # Data preparation with ASR
    ├── create_submission.py # AG submission generator
    ├── prompt_builder.py    # Prompt templates
    ├── models/              # VLLM, Transformers backends
    ├── libs/                # Evaluation utilities
    └── configs/             # Configuration examples
```

## Features

### Multiple Choice (MC) Task
- **Ranking-based Inference**: Rank all options from most to least likely
- **Multiple Backend Support**: VLLM, Transformers, and llama.cpp HTTP server
- **Ensemble Voting**: Combine predictions using Borda, Plurality, Approval, or Condorcet methods

### Answer Generation (AG) Task
- **Open-ended Answers**: Generate natural language responses
- **Enhanced Prompting**: Video context generation for improved understanding
- **Multiple Answer Generation**: Generate diverse answers for ensemble submission
- **Speech Recognition**: Whisper and Seamless M4T ASR integration

### Common Features
- **Audio Transcription Support**: Multimodal understanding with speech
- **Comprehensive Evaluation**: Detailed metrics and analysis
- **Submission Generation**: Competition-ready output files

## Supported Models

| Model | Backend | Tasks |
|-------|---------|-------|
| InternVL3.5 (8B/26B/38B) | Transformers, VLLM | MC, AG |
| VideoLLaMA3 | Transformers | AG |
| Qwen2.5-VL | Transformers, VLLM | MC, AG |
| LLaMA 3.2 Vision | llama.cpp | MC |

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU inference)
- FFmpeg (for video processing)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/aiclub-uit/TrecVID2025_VQA.git
cd TrecVID2025_VQA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install core dependencies
pip install -r requirements.txt

# Optional: Install VLLM for faster inference
pip install vllm

# Optional: Install Whisper for speech recognition (AG task)
pip install openai-whisper
```

## Quick Start

### Multiple Choice Task

```bash
cd multiple_choice

# Run inference
python mc_infer.py \
  --config configs/internvl3_5_transformers.yaml \
  --data_dir /path/to/prepared_data \
  --output results/

# Evaluate results
python eval.py results/ --output evaluations/

# Create submission
python create_submission.py \
  --results results/ \
  --csv test_dataset.csv \
  --output submission.csv
```

See [multiple_choice/README.md](multiple_choice/README.md) for detailed documentation.

### Answer Generation Task

```bash
cd answer_generation

# Prepare data with speech recognition
python prepare.py \
  --metadata_folder /path/to/metadata \
  --videos_folder /path/to/videos \
  --output_folder prepared/ \
  --enable_speech_recognition

# Run inference
python infer.py \
  --config configs/videollama3.yaml \
  --batch \
  --video_path /path/to/videos \
  --prepared_folder prepared/ \
  --output results/

# Evaluate results
python eval.py --input results/ --output evaluations/

# Create submission
python create_submission.py \
  --results_dirs results/ \
  --output submission.csv
```

See [answer_generation/README.md](answer_generation/README.md) for detailed documentation.

## Tips for Best Results

1. **Frame Count**: Use 12-16 frames for MC; up to 180 for AG with VideoLLaMA3
2. **Model Size**: Larger models give better accuracy but require more VRAM
3. **Audio**: Include speech transcription when available
4. **Ensemble**: Combine predictions from multiple models for improved accuracy

## Troubleshooting

### Common Issues

**Out of Memory**
```yaml
# Use 8-bit quantization
transformers_config:
  load_in_8bit: true

# Or reduce frame count
mc_config:
  max_frames: 8
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
@software{trecvid2025_vqa,
  title = {TrecVID 2025 Video Question Answering},
  author = {AI Club UIT},
  year = {2025},
  url = {https://github.com/aiclub-uit/TrecVID2025_VQA}
}
```

## Acknowledgments

- [InternVL](https://github.com/OpenGVLab/InternVL) for vision-language models
- [VideoLLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) for video understanding
- [VLLM](https://github.com/vllm-project/vllm) for high-throughput inference
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
