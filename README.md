# whisper-train

This repository contains scripts and resources for training automatic speech recognition (ASR) models with Oral History data using the Whisper model from OpenAI. The project is inspired by the Hugging Face documentation and blog posts on fine-tuning Whisper models. The workflows are optimized for the ASR4Memory project. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Installation

To set up the environment for this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/pkompiel/whisper_train.git
    cd whisper-train
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Preliminary remark

Remember to define input and output folders for each script

### Data Preparation

1. Anonymize subtile files if necessary (`.vtt`) using anonymized word segment tables (`.csv`) provided by the ASR4Memory pipelines.
    ```sh
    python anonymize_vtt.py
    ```
2. Create ASR training dataset by using the ffmpeg dataset creator to extract audio segments and prepare metadata. It needs an audio file (`.wav`) and a corresponding subtitle file (`.vtt`) as input
    ```sh
    python asr-dataset-creator-pydub.py
    ```

### Training

Use the dataset to train whisper model with the provided jupyter notebook

    ```
    whisper-train.ipynb
    ```

## Referencecs

- [Hugging Face Datasets Audio Dataset](https://huggingface.co/docs/datasets/audio_dataset)
- [Hugging Face Blog: Fine-Tune Whisper](https://huggingface.co/blog/fine-tune-whisper)