import pandas as pd
import re
from pathlib import Path
from app_config import get_config

# Load the configuration
config = get_config()["vtt_anonymization"]

INPUT_DIR = Path(config["input_directory"])
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def validate_input_files(INPUT_DIR):
    """
    Validates that exactly one VTT file and one CSV file exist in the input directory.
    """
    vtt_files = list(INPUT_DIR.glob("*.vtt"))
    csv_files = list(INPUT_DIR.glob("*.csv"))

    if len(vtt_files) != 1 or len(csv_files) != 1:
        raise ValueError("There must be exactly one VTT and one CSV file in the input directory")

    return vtt_files[0], csv_files[0]


def load_csv(file_path):
    """
    Loads the anonymized CSV file and processes it.
    """
    df = pd.read_csv(file_path, delimiter="\t", encoding="utf-8")
    df['CLEANED_WORD'] = df['WORD'].apply(lambda x: re.sub(r'[^\w\s-]', '', x) if isinstance(x, str) else x)
    words_to_anonymize = df[df['SCORE'] == 'x']['CLEANED_WORD'].tolist()
    return list(set(words_to_anonymize))  # Remove duplicates


def load_vtt(file_path):
    """
    Loads the VTT file content.
    """
    with open(file_path, "r", encoding="utf-8") as vtt_file:
        return vtt_file.read()


def apply_replacements(content, replacements):
    """
    Applies character replacements on the VTT content.
    """
    for item in replacements:
        pattern = item["pattern"]
        replacement = item["replacement"]
        content = re.sub(pattern, replacement, content)
    return content


def anonymize_words(content, words_to_anonymize):
    """
    Replaces specified words in the VTT content.
    """
    for word in words_to_anonymize:
        word_regex = r'\b' + re.escape(word) + r'\b'
        content = re.sub(word_regex, '', content)
    return content


def save_vtt(content, output_path):
    """
    Saves the modified VTT content to a file.
    """
    with open(output_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write(content)
    print(f"Anonymized VTT file saved at: {output_path}")


def main():
    """Main function to execute the VTT anonymization process."""
    # Validate input files
    input_vtt_file, input_csv_file = validate_input_files(INPUT_DIR)

    # Load CSV and extract words to anonymize
    words_to_anonymize = load_csv(input_csv_file)
    print(f"Words to anonymize: {words_to_anonymize}")

    # Load VTT file content
    vtt_content = load_vtt(input_vtt_file)

    # Apply replacements
    replacements = config.get("vtt_replacements", [])
    vtt_content = apply_replacements(vtt_content, replacements)

    # Anonymize words
    vtt_content = anonymize_words(vtt_content, words_to_anonymize)

    # Save the modified VTT content
    output_vtt_file = OUTPUT_DIR / f"{input_vtt_file.stem}_anonymized.vtt"
    save_vtt(vtt_content, output_vtt_file)


if __name__ == "__main__":
    main()