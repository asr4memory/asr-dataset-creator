# asr-dataset-creator

This repository contains scripts and resources for preprocessing audiovisual Oral History data for training automatic speech recognition (ASR) models using Whisper from OpenAI. The workflows are optimized for the ASR4Memory project. For running the whole workflow, audio files (`.wav`) and corresponding subtitle files (`.vtt`) are required as input.

## Installation

To set up the environment for this project, follow these steps:

1. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Install Pytorch (https://pytorch.org/get-started/locally/) according to your system.
3. When using Mac with ARM install the MLX library:
    ```sh
    pip install mlx_lm
    ```
4. Create the configuration file:
    ```sh
    cp config.example.toml config.toml
    ```
5. Create data folder
    ```sh
    cp -r data_example data
    ```

## Usage

1. Convert subtitle files (`.vtt`) to text files (`.txt`)
    ```sh
    python vtt-to-txt.py
    ```
2. Extract entities (people and addresses) from the text files and save them in JSON files (`.json`). You can either use an CUDA or an MAC/ARM version of the script. The ner workflows script uses GliNER (https://github.com/urchade/GLiNER) for a rough pre extraction of entities (people and adresses). In a second step, the results are fed to a LLM prompt for refined results. Currently, the CUDA script uses Llama3.1 8B (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and the MAC/ARM script uses Llama 3.3 70B 4bit quantized (https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit). For better results, it is recommended to use the MAC/ARM script.
    ```sh
    python ner-workflow.py
    ```
    ```sh
    python ner-workflow-mac.py
    ```
3. Anonymize the subtile files if necessary (`.vtt`) using the JSON files (`.json`) provided by the ASR4Memory pipelines.
    ```sh
    python anonymize_vtt.py
    ```
4. Create ASR training dataset by using the ffmpeg dataset creator to extract audio segments and prepare metadata. It needs an audio file (`.wav`) and a corresponding subtitle file (`.vtt`) as input
    ```sh
    python asr-dataset-creator.py
    ```
5. Merge exiting datasets into one large dataset
    ```sh
    python asr-dataset-merger.py
    ```
6. Convert dataset to HDF5 format
    ```sh
    python hdf5-converter.py
    ```
7. Test the created datasets
    ```sh
    python load-data-set.py
    ```
