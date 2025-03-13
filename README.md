# Whisper Terminal ASR

A lightweight, terminal-based Automatic Speech Recognition (ASR) application for on-premise, CPU-only deployment on macOS. This application is optimized for MacBooks, with special support for Apple Silicon through MLX integration.

## Features

- **Multiple Whisper Implementations**:
  - **OpenAI Whisper**: The original Whisper model
  - **Faster Whisper**: CTranslate2-based implementation for CPU optimization
  - **Insanely Fast Whisper**: Optimized with Flash Attention 2 and BetterTransformer
  - **MLX Whisper**: Optimized specifically for Apple Silicon (M1/M2/M3)
  - **Lightning Whisper MLX**: Ultra-fast implementation optimized for Apple Silicon, providing up to 10x faster inference than Whisper CPP and 4x faster than standard MLX implementations

- **Model Size Options**: 
  - All standard Whisper model sizes: tiny, base, small, medium, large (including large-v2, large-v3)
  - **Distilled Models** (with Lightning MLX): distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3
  - **Quantization Options** (with Lightning MLX): 4-bit and 8-bit quantization for even faster inference

- **Audio Input Methods**:
  - Transcribe existing audio files
  - Record from microphone with manual start/stop
  - Continuous streaming transcription in real-time

- **Terminal Interface**: Beautiful rich text interface with colors and progress indicators

- **Performance Benchmarking**: Compare different implementations and model sizes

## Requirements

- Python 3.8+
- macOS (optimized for Apple Silicon, but compatible with Intel)
- Microphone for recording functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper-term-asr.git
cd whisper-term-asr
```

2. Install the package with desired components:

```bash
# Basic installation
pip install -e .

# With all model implementations
pip install -e ".[all]"

# Apple Silicon users might want just Lightning MLX (recommended for fastest performance)
pip install -e ".[lightning-mlx]"

# For visualization support (benchmarking plots)
pip install -e ".[vis]"
```

## Usage

The application provides several commands:

### List Available Models

```bash
python -m whisper_term_asr list
```

### Show System Information

```bash
python -m whisper_term_asr info
```

### Transcribe an Audio File

```bash
# Using Faster Whisper (good for all systems)
python -m whisper_term_asr transcribe path/to/audio.wav --model-type faster --model-size base

# Using Lightning Whisper MLX (best for Apple Silicon)
python -m whisper_term_asr transcribe path/to/audio.wav --model-type lightning-mlx --model-size distil-medium.en --batch-size 12
```

Options:
- `--model-type`, `-m`: Model implementation (openai, faster, insanely-fast, mlx, lightning-mlx)
- `--model-size`, `-s`: Model size (tiny, base, small, medium, large, large-v2, large-v3, distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3)
- `--language`, `-l`: Language code (e.g., 'en', 'fr')
- `--device`, `-d`: Device to run on (cpu, cuda, mps)
- `--compute-type`, `-c`: Computation precision (float32, float16, int8)
- `--output`, `-o`: Output file for transcription
- `--batch-size`: Batch size for processing (primarily for Lightning MLX)
- `--quantization`: Quantization level for Lightning MLX models (4bit, 8bit)

### Transcribe from Microphone

```bash
# Manual recording mode (press Enter to start/stop)
python -m whisper_term_asr mic --model-type lightning-mlx --model-size distil-medium.en

# Continuous transcription mode
python -m whisper_term_asr mic --continuous --model-type lightning-mlx --model-size distil-small.en --quantization 8bit
```

Options:
- Same model options as above
- `--continuous`, `-C`: Enable continuous transcription
- `--save-audio`, `-S`: Save recorded audio files
- `--output-dir`, `-o`: Directory to save recordings

### Benchmark Models

```bash
python -m whisper_term_asr benchmark --models faster/base lightning-mlx/distil-medium.en mlx/base --audio-files path/to/audio1.wav path/to/audio2.wav --plot
```

Options:
- `--models`, `-m`: Models to benchmark (format: type/size)
- `--audio-files`, `-a`: Audio files to benchmark
- `--num-runs`, `-n`: Number of runs to average
- `--plot`, `-p`: Generate performance comparison plot
- `--plot-type`: Plot type (bar, box)
- `--batch-size`: Batch size for Lightning MLX models
- `--quantization`: Quantization level for Lightning MLX models

## Performance Recommendations

For the best performance on macOS:

1. **Apple Silicon (M1/M2/M3) Macs**:
   - The Lightning Whisper MLX implementation provides the best performance by far
   - Recommended configuration: `--model-type lightning-mlx --model-size distil-medium.en --batch-size 12`
   - For even faster performance with slight quality tradeoff: `--quantization 8bit`
   - Adjust batch size based on available RAM (higher is faster but uses more memory)

2. **Intel Macs**:
   - The Faster Whisper implementation generally performs best
   - Use `--model-type faster --compute-type float16`

3. **Memory-Constrained Systems**:
   - Use smaller models: tiny or base
   - Consider int8 quantization: `--compute-type int8` or `--quantization 8bit`
   - For Lightning MLX, reduce batch size: `--batch-size 4`

## Implementation Details

- The application uses a modular design to easily swap between model implementations
- Audio recording supports both manual and continuous streaming modes
- Rich terminal UI created with the `rich` library for better user experience
- Benchmarking tools to compare latency across implementations
- Lightning Whisper MLX uses batched decoding and distilled models for higher throughput

## License

MIT

## Acknowledgements

This project wraps several open-source implementations of OpenAI's Whisper model:
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Lightning Whisper MLX](https://github.com/irobot-ml/lightning-whisper-mlx) - Created by [Mustafa Aljadery](https://github.com/mustafaaljadery)
