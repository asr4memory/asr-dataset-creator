import os
import toml
from default_config import CONST_DEFAULT_CONFIG

combined_config = {}

def initialize_config():
    "Merges configuration from config.toml with defaults."
    global combined_config

    config_file_path = os.path.join(os.getcwd(), "config.toml")

    # Load default configuration
    combined_config = CONST_DEFAULT_CONFIG.copy()

    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)

            vtt_anonymization_defaults = CONST_DEFAULT_CONFIG["vtt_anonymization"]
            vtt_anonymization_config = data.get("vtt_anonymization", {})
            combined_config["vtt_anonymization"] = merge_dicts(
                vtt_anonymization_defaults, vtt_anonymization_config
            )

            combined_config["dataset_creator"] = {
                **CONST_DEFAULT_CONFIG["dataset_creator"],
                **data.get("dataset_creator", {}),
            }
            combined_config["dataset_merger"] = {
                **CONST_DEFAULT_CONFIG["dataset_merger"],
                **data.get("dataset_merger", {}),
            }
            combined_config["dataset_test"] = {
                **CONST_DEFAULT_CONFIG["dataset_test"],
                **data.get("dataset_test", {}),
            }

    except FileNotFoundError:
        print("config.toml not found. Using default configuration.")

def merge_dicts(default, override):
    """
    Function to merge two dictionaries recursively.
    """
    result = default.copy()
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def get_config() -> dict:
    "Returns app configuration as a dictionary."
    return combined_config

initialize_config()