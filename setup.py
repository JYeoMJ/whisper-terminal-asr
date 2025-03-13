from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisper-term-asr",
    version="0.1.0",
    author="Developer",
    author_email="dev@example.com",
    description="A lightweight, terminal-based ASR application for on-premise, CPU-only testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper-term-asr",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "pyaudio>=0.2.11",
        "sounddevice>=0.4.4",
        "soundfile>=0.10.3",
        "click>=8.1.3",
        "tqdm>=4.64.0",
        "rich>=12.0.0",
    ],
    extras_require={
        "openai": ["openai-whisper>=20230314"],
        "faster": ["faster-whisper>=0.9.0"],
        "insanely-fast": ["transformers>=4.30.0", "optimum>=1.12.0", "accelerate>=0.23.0"],
        "mlx": ["mlx>=0.0.4", "mlx-lm>=0.0.3"],
        "lightning-mlx": ["lightning-whisper-mlx>=0.0.10", "mlx>=0.0.4"],
        "all": [
            "openai-whisper>=20230314",
            "faster-whisper>=0.9.0",
            "transformers>=4.30.0",
            "optimum>=1.12.0",
            "accelerate>=0.23.0",
            "mlx>=0.0.4", 
            "mlx-lm>=0.0.3",
            "lightning-whisper-mlx>=0.0.10"
        ],
        "vis": ["matplotlib>=3.5.0"],
    },
    entry_points={
        "console_scripts": [
            "whisper-term-asr=whisper_term_asr.cli:main",
        ],
    },
)
