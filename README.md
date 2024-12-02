# asr-dataset-creator

This repository contains scripts and resources for preprocessing audiovisual Oral History data for training automatic speech recognition (ASR) models using Whisper from OpenAI. The workflows are optimized for the ASR4Memory project. 

## Installation

To set up the environment for this project, follow these steps:

1. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Create the configuration file:
    ```sh
    cp config.example.toml config.toml
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
    python asr-dataset-creator.py
    ```
3. Merge exiting datasets into one large dataset
    ```sh
    python asr-dataset-merger.py
    ```