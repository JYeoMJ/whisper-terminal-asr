"""
Benchmarking utilities for ASR models.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def run_benchmark(
    models: List[Any],
    audio_paths: List[str],
    output_dir: Optional[str] = None,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """
    Run benchmark on multiple models and audio files.
    
    Args:
        models: List of ASR model instances
        audio_paths: List of audio file paths to test
        output_dir: Directory to save results
        num_runs: Number of runs to average over
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_runs": num_runs,
        "models": [],
        "audio_files": [],
        "results": []
    }
    
    # Record audio file information
    for audio_path in audio_paths:
        file_info = {
            "path": audio_path,
            "size_bytes": os.path.getsize(audio_path),
            "duration_seconds": None  # Will be populated if possible
        }
        
        # Try to get audio duration if soundfile is available
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            file_info["duration_seconds"] = info.duration
            file_info["sample_rate"] = info.samplerate
            file_info["channels"] = info.channels
        except:
            pass
            
        results["audio_files"].append(file_info)
    
    # Run benchmarks for each model on each audio file
    for model in models:
        model_info = {
            "type": model.__class__.__name__,
            "size": model.model_size,
            "device": model.device,
            "compute_type": model.compute_type
        }
        results["models"].append(model_info)
        
        # Ensure model is loaded before benchmarking
        if model.model is None:
            model.load_model()
        
        for audio_idx, audio_path in enumerate(audio_paths):
            print(f"Benchmarking {model_info['type']} ({model_info['size']}) on {os.path.basename(audio_path)}...")
            
            # Run multiple benchmarks and average results
            load_times = []
            inference_times = []
            
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}...")
                
                # Don't need to reload model but record time anyway for consistency
                start_time = time.time()
                load_time = time.time() - start_time
                load_times.append(load_time)
                
                # Run transcription and measure time
                start_time = time.time()
                result = model.transcribe(audio_path)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Save only the text result from the first run
                if run == 0:
                    text_result = result.get("text", "")
            
            # Calculate statistics
            result_entry = {
                "model_idx": len(results["models"]) - 1,
                "audio_idx": audio_idx,
                "avg_load_time": sum(load_times) / num_runs,
                "avg_inference_time": sum(inference_times) / num_runs,
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times),
                "std_inference_time": np.std(inference_times) if len(inference_times) > 1 else 0,
                "text_sample": text_result[:200] + "..." if len(text_result) > 200 else text_result
            }
            
            results["results"].append(result_entry)
    
    # Save results to file if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_dir, f"benchmark-{timestamp}.json")
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark results saved to {output_path}")
    
    return results


def plot_benchmark_results(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    plot_type: str = "bar",
) -> None:
    """
    Plot benchmark results.
    
    Args:
        results: Benchmark results from run_benchmark
        output_path: Path to save plot image (if None, displays plot)
        plot_type: Type of plot ('bar' or 'box')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")
    
    # Extract model and audio information
    models = [f"{m['type']}_{m['size']}" for m in results["models"]]
    audio_files = [os.path.basename(a["path"]) for a in results["audio_files"]]
    
    # Group results by model and audio file
    grouped_results = {}
    for result in results["results"]:
        model_idx = result["model_idx"]
        audio_idx = result["audio_idx"]
        model_name = models[model_idx]
        audio_name = audio_files[audio_idx]
        
        if model_name not in grouped_results:
            grouped_results[model_name] = {}
        
        grouped_results[model_name][audio_name] = result["avg_inference_time"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up plot
    if plot_type == "bar":
        x = np.arange(len(audio_files))
        width = 0.8 / len(models)
        
        for i, model_name in enumerate(models):
            inference_times = [grouped_results[model_name].get(audio, 0) for audio in audio_files]
            ax.bar(x + i * width, inference_times, width, label=model_name)
        
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(audio_files, rotation=45, ha="right")
    
    elif plot_type == "box":
        data = []
        labels = []
        
        for model_name in models:
            model_times = [grouped_results[model_name].get(audio, 0) for audio in audio_files]
            data.append(model_times)
            labels.append(model_name)
        
        ax.boxplot(data, labels=labels)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Set labels and title
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_title("ASR Model Benchmark Comparison")
    ax.legend()
    
    # Adjust layout and save/display
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
