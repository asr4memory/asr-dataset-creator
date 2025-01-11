"""
Default configuration.
These settings are overridden by the settings in the config.toml file,
if present.
"""

CONST_DEFAULT_CONFIG = {
    "vtt_to_txt": {
        "input_directory": "",
        "output_directory": "",
    },
    "vtt_anonymization": {
        "input_directory": "",
        "output_directory": "",
        "vtt_replacements": [], 
    },
    "dataset_creator": {
        "input_directory": "",
        "output_directory": "",
        "sample_rate": "16000",
        "offset": 0.075,
    },
    "dataset_merger": {
        "input_directory": "",
        "output_directory": "",
    },
    "dataset_test": {
        "input_directory": "",
    },
}
