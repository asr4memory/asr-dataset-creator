"""
Default configuration.
These settings are overridden by the settings in the config.toml file,
if present.
"""

CONST_DEFAULT_CONFIG = {
    "logging": {
        "logging_directory": "",
    },    
    "vtt_to_txt": {
        "input_directory": "",
        "output_directory": "",
    },
    "ner_workflow": {
        "input_directory": "",
        "output_directory": "",
        "ner_batch_size": 15,
        "ner_threshold": 0.5,
        "llm_max_new_tokens": 10000, 
        "entity_linking_file": "",
        "entity_linking_threshold": 80,
    },
    "vtt_anonymization": {
        "input_directory_vtt": "",
        "input_directory_json": "",
        "output_directory": "",
        "vtt_replacements": [], 
    },
    "dataset_creator": {
        "input_directory_vtt": "",
        "input_directory_wav": "",
        "output_directory": "",
        "offset": 0.075,
    },
    "dataset_merger": {
        "input_directory": "",
        "output_directory": "",
    },
    "hdf5_converter": {
        "input_directory": "",
        "output_directory": "",
        "batch_size": 100,
        "threads": 4,
        "shard_size": 8192,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "random_seed": 1337,
    },
    "dataset_test": {
        "input_directory": "",
    },
}
