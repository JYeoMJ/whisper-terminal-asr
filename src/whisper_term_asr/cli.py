"""
Command-line interface for the Whisper Terminal ASR application.
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import numpy as np

from .audio.recorder import AudioRecorder
from .models.factory import create_model, list_available_models
from .utils.benchmark import run_benchmark, plot_benchmark_results


# Initialize rich console
console = Console()


def display_title():
    """Display application title."""
    title = """
 __        ___     _                         _____                     _    ____ ____  
 \ \      / / |__ (_)___ _ __   ___ _ __    |_   _|__ _ __ _ __ ___   / \  / ___/ ___| 
  \ \ /\ / /| '_ \| / __| '_ \ / _ \ '__|_____| |/ _ \ '__| '_ ` _ \ / _ \| |   \___ \ 
   \ V  V / | | | | \__ \ |_) |  __/ | |_____| |  __/ |  | | | | | / ___ \ |___ ___) |
    \_/\_/  |_| |_|_|___/ .__/ \___|_|       |_|\___|_|  |_| |_| |_/_/   \_\____|____/ 
                        |_|                                                           
    """
    console.print(Panel(title, title="Whisper Terminal ASR", subtitle="v0.1.0", style="blue"))


def list_models_command():
    """List available models command."""
    models = list_available_models()
    
    # Create table
    table = Table(title="Available ASR Models")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Sizes", style="magenta")
    
    # Add rows
    for model_type, info in models.items():
        table.add_row(
            model_type,
            info["name"],
            info["description"],
            ", ".join(info["sizes"])
        )
    
    console.print(table)
    
    
def info_command():
    """Display system information."""
    # Create table
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="green")
    
    # Check for PyTorch
    try:
        import torch
        torch_version = torch.__version__
        torch_device = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch_device == "CPU" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = "MPS (Apple Silicon)"
        table.add_row("PyTorch", f"Version: {torch_version}, Device: {torch_device}")
    except ImportError:
        table.add_row("PyTorch", "Not installed")
    
    # Check for Faster Whisper
    try:
        import faster_whisper
        table.add_row("Faster Whisper", f"Version: {faster_whisper.__version__}")
    except (ImportError, AttributeError):
        table.add_row("Faster Whisper", "Not installed")
    
    # Check for MLX
    try:
        import mlx
        table.add_row("MLX", f"Version: {mlx.__version__}")
    except (ImportError, AttributeError):
        table.add_row("MLX", "Not installed")
    
    # Check for Lightning Whisper MLX
    try:
        import lightning_whisper_mlx
        table.add_row("Lightning Whisper MLX", f"Version: {lightning_whisper_mlx.__version__}")
    except (ImportError, AttributeError):
        table.add_row("Lightning Whisper MLX", "Not installed")
    
    # Check for Transformers
    try:
        import transformers
        table.add_row("Transformers", f"Version: {transformers.__version__}")
    except ImportError:
        table.add_row("Transformers", "Not installed")
    
    # Check for audio libraries
    try:
        import sounddevice as sd
        table.add_row("Audio Devices", f"Input: {sd.query_devices(kind='input')}")
    except ImportError:
        table.add_row("Audio Devices", "sounddevice not installed")
    
    console.print(table)


def transcribe_file_command(args):
    """Transcribe an audio file."""
    if not os.path.exists(args.audio_file):
        console.print(f"[red]Error: Audio file '{args.audio_file}' not found[/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        # Create model
        task = progress.add_task("[green]Loading model...", total=None)
        model = create_model(
            model_type=args.model_type,
            model_size=args.model_size,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            quantization=args.quantization
        )
        model.load_model()
        progress.update(task, completed=True)
        
        # Transcribe file
        task = progress.add_task(f"[green]Transcribing {os.path.basename(args.audio_file)}...", total=None)
        start_time = time.time()
        result = model.transcribe(args.audio_file, prompt=args.prompt)
        inference_time = time.time() - start_time
        progress.update(task, completed=True)
    
    # Display results
    console.print(Panel(
        result["text"],
        title=f"Transcription ({inference_time:.2f}s)",
        subtitle=f"Model: {args.model_type}/{args.model_size}",
        style="green"
    ))
    
    # Save result if output file specified
    if args.output:
        with open(args.output, "w") as f:
            if args.output.endswith(".json"):
                json.dump(result, f, indent=2)
            else:
                f.write(result["text"])
        console.print(f"[green]Results saved to {args.output}[/green]")


def transcribe_mic_command(args):
    """Transcribe from microphone."""
    # Create recorder
    recorder = AudioRecorder(
        sample_rate=16000,
        output_dir=args.output_dir or "recordings"
    )
    
    # Create model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Loading model...", total=None)
        model = create_model(
            model_type=args.model_type,
            model_size=args.model_size,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            quantization=args.quantization
        )
        model.load_model()
        progress.update(task, completed=True)
    
    # Continuous streaming mode
    if args.continuous:
        console.print("[yellow]Starting continuous transcription. Press Ctrl+C to stop...[/yellow]")
        
        # Set up callback for processing audio chunks
        def process_audio(audio_data, sample_rate):
            if len(audio_data) < sample_rate:  # Skip very short segments
                return
                
            result = model.transcribe(audio_data)
            if result["text"].strip():  # Only print if there's actually text
                console.print(f"[green]{result['text']}[/green]")
        
        try:
            # Start streaming with callback
            recorder.start_streaming(callback_fn=process_audio)
            
            # Keep running until interrupted
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            console.print("[yellow]Stopping transcription...[/yellow]")
            recorder.stop_streaming()
    
    # Manual recording mode
    else:
        console.print("[yellow]Press Enter to start recording, then Enter again to stop...[/yellow]")
        try:
            while True:
                input("Press Enter to start recording...")
                console.print("[red]Recording... Press Enter to stop[/red]")
                recorder.start_recording()
                input()
                audio_data = recorder.stop_recording(save=args.save_audio)
                
                if audio_data is not None:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task("[green]Transcribing...", total=None)
                        result = model.transcribe(audio_data, prompt=args.prompt)
                        progress.update(task, completed=True)
                    
                    # Display result
                    console.print(Panel(result["text"], title="Transcription", style="green"))
                
                if input("Continue? (y/n): ").lower() != 'y':
                    break
                    
        except KeyboardInterrupt:
            if recorder.recording:
                recorder.stop_recording(save=args.save_audio)
            console.print("[yellow]Exiting...[/yellow]")


def benchmark_command(args):
    """Run benchmark on models."""
    # Validate audio files
    audio_files = []
    for file_pattern in args.audio_files:
        # Handle glob patterns
        from glob import glob
        files = glob(file_pattern)
        if not files:
            console.print(f"[yellow]Warning: No files match pattern '{file_pattern}'[/yellow]")
        audio_files.extend(files)
    
    if not audio_files:
        console.print("[red]Error: No audio files found for benchmarking[/red]")
        return
    
    # Parse model specifications
    model_specs = []
    for model_spec in args.models:
        parts = model_spec.split("/")
        if len(parts) == 2:
            model_type, model_size = parts
            model_specs.append((model_type, model_size))
        else:
            console.print(f"[yellow]Warning: Invalid model spec '{model_spec}', should be 'type/size'[/yellow]")
    
    if not model_specs:
        console.print("[red]Error: No valid model specifications provided[/red]")
        return
    
    # Create models
    models = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for model_type, model_size in model_specs:
            task = progress.add_task(f"[green]Loading {model_type}/{model_size}...", total=None)
            try:
                model = create_model(
                    model_type=model_type,
                    model_size=model_size,
                    device=args.device,
                    compute_type=args.compute_type,
                    cache_dir=args.cache_dir,
                    batch_size=args.batch_size,
                    quantization=args.quantization
                )
                models.append(model)
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, description=f"[red]Failed to load {model_type}/{model_size}: {e}[/red]")
    
    if not models:
        console.print("[red]Error: No models could be loaded for benchmarking[/red]")
        return
    
    # Run benchmark
    results = run_benchmark(
        models=models,
        audio_paths=audio_files,
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )
    
    # Display results
    table = Table(title="Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Audio File", style="green")
    table.add_column("Avg Time (s)", style="magenta")
    table.add_column("Min Time (s)")
    table.add_column("Max Time (s)")
    
    for result in results["results"]:
        model_idx = result["model_idx"]
        audio_idx = result["audio_idx"]
        model_name = f"{results['models'][model_idx]['type']}/{results['models'][model_idx]['size']}"
        audio_name = os.path.basename(results["audio_files"][audio_idx]["path"])
        
        table.add_row(
            model_name,
            audio_name,
            f"{result['avg_inference_time']:.2f}",
            f"{result['min_inference_time']:.2f}",
            f"{result['max_inference_time']:.2f}"
        )
    
    console.print(table)
    
    # Plot results if requested
    if args.plot:
        try:
            plot_path = os.path.join(args.output_dir, "benchmark_plot.png") if args.output_dir else None
            plot_benchmark_results(results, output_path=plot_path, plot_type=args.plot_type)
        except Exception as e:
            console.print(f"[red]Error creating plot: {e}[/red]")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Whisper Terminal ASR - Lightweight terminal-based ASR application")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    
    # Common model arguments
    def add_model_arguments(parser):
        parser.add_argument("--model-type", "-m", default="faster", 
                           choices=["openai", "faster", "insanely-fast", "mlx", "lightning-mlx"],
                           help="Model implementation to use")
        
        # The model_size choices depend on the model_type, but we'll allow all possible values
        parser.add_argument("--model-size", "-s", default="base", 
                           choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", 
                                   "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3"],
                           help="Model size to use")
        
        parser.add_argument("--language", "-l", help="Language code (e.g., 'en', 'fr')")
        parser.add_argument("--device", "-d", help="Device to run on (cpu, cuda, mps)")
        parser.add_argument("--compute-type", "-c", default="float16", choices=["float32", "float16", "int8"],
                           help="Computation type/precision")
        parser.add_argument("--cache-dir", help="Directory to cache models")
        parser.add_argument("--prompt", "-p", help="Prompt to guide transcription")
        
        # Add Lightning MLX specific arguments
        parser.add_argument("--batch-size", type=int, default=12,
                           help="Batch size for processing (applies to Lightning MLX)")
        parser.add_argument("--quantization", choices=[None, "4bit", "8bit"],
                           help="Quantization level for Lightning MLX models")
    
    # Transcribe file command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    add_model_arguments(transcribe_parser)
    transcribe_parser.add_argument("audio_file", help="Audio file to transcribe")
    transcribe_parser.add_argument("--output", "-o", help="Output file for transcription")
    
    # Transcribe from microphone command
    mic_parser = subparsers.add_parser("mic", help="Transcribe from microphone")
    add_model_arguments(mic_parser)
    mic_parser.add_argument("--continuous", "-C", action="store_true", help="Continuous transcription mode")
    mic_parser.add_argument("--save-audio", "-S", action="store_true", help="Save recorded audio")
    mic_parser.add_argument("--output-dir", "-o", help="Directory to save recordings and transcriptions")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    benchmark_parser.add_argument("--models", "-m", nargs="+", default=["faster/base"], 
                                help="Models to benchmark (format: type/size)")
    benchmark_parser.add_argument("--audio-files", "-a", nargs="+", required=True,
                                help="Audio files to benchmark")
    benchmark_parser.add_argument("--device", "-d", help="Device to run on (cpu, cuda, mps)")
    benchmark_parser.add_argument("--compute-type", "-c", default="float16",
                                help="Computation type/precision")
    benchmark_parser.add_argument("--cache-dir", help="Directory to cache models")
    benchmark_parser.add_argument("--num-runs", "-n", type=int, default=3,
                                help="Number of runs to average over")
    benchmark_parser.add_argument("--output-dir", "-o", help="Directory to save results")
    benchmark_parser.add_argument("--plot", "-p", action="store_true", help="Plot benchmark results")
    benchmark_parser.add_argument("--plot-type", choices=["bar", "box"], default="bar",
                                help="Type of plot to create")
    benchmark_parser.add_argument("--batch-size", type=int, default=12,
                               help="Batch size for processing (applies to Lightning MLX)")
    benchmark_parser.add_argument("--quantization", choices=[None, "4bit", "8bit"],
                               help="Quantization level for Lightning MLX models")
    
    args = parser.parse_args()
    return args


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Display title
        display_title()
        
        # Execute command
        if args.command == "list":
            list_models_command()
        elif args.command == "info":
            info_command()
        elif args.command == "transcribe":
            transcribe_file_command(args)
        elif args.command == "mic":
            transcribe_mic_command(args)
        elif args.command == "benchmark":
            benchmark_command(args)
        else:
            # If no command specified, show help
            print("Please specify a command. Use --help for available commands.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if "--debug" in sys.argv:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
