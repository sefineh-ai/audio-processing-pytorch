# Audio Processing with PyTorch

[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

> **A modern, modular audio processing and deep learning toolkit built with PyTorch.**

---

## Overview
This project provides a robust foundation for building, training, and evaluating deep learning models for audio and speech processing tasks using PyTorch. It is designed for rapid prototyping, research, and production deployment, and leverages open-source datasets for reproducible results.

## Features
- Audio data loading and preprocessing (batch and single-file)
- Feature extraction (MFCC)
- Modular PyTorch model templates
- Training and evaluation scripts
- Integration with open-source datasets (UrbanSound8K)
- Extensible for speech, music, and general audio tasks

## Open Dataset: UrbanSound8K
This project demonstrates end-to-end audio classification using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset, a popular open-source collection of urban sounds with 10 classes.

### Downloading UrbanSound8K
Use the provided script to download and extract the dataset:
```bash
python src/download_urbansound8k.py
```

## Project Structure
```
audio-processing-pytorch/
├── data/                # Audio files, datasets
├── data_features/       # Extracted features (auto-generated)
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/                 # Source code (preprocessing, training, inference)
│   ├── preprocess.py
│   ├── train.py
│   ├── infer.py
│   └── download_urbansound8k.py
├── models/              # Saved models
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .gitignore
└── setup.py             # (optional) for packaging
```

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd audio-processing-pytorch
   ```
2. **Set up a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download UrbanSound8K:**
   ```bash
   python src/download_urbansound8k.py
   ```
4. **Preprocess the audio data:**
   ```bash
   python src/preprocess.py --input_dir UrbanSound8K/audio/ --output_dir data_features/
   ```
5. **Train the model:**
   ```bash
   python src/train.py --features_root data_features/ --epochs 10 --batch_size 16 --lr 0.001 --model_out models/simple_audio_classifier.pth
   ```

## Requirements
- Python 3.7+
- PyTorch >= 1.10
- librosa
- torchaudio
- numpy, pandas, matplotlib
- tqdm, requests

## License
MIT 