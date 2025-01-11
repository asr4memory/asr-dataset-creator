from pathlib import Path
import re
import string
from app_config import get_config


def clean_vtt_content(content, remove_punctuation=False):
    "Cleans up VTT strings so that WER can be detected."
    # Remove the VTT heading, segment numbers, time codes and notes and
    # comments in () and <>:
    result = re.sub(
        r"WEBVTT\n\n|\d+\n\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n|\(.*?\)|<.*?>",
        "",
        content,
    )
    # Corrected regular expression to remove the filler words "äh",
    # "ähs", "ähm", "hm", und "hmm" including the following comma,
    # semicolon, hyphen or period:
    result = re.sub(
        r"\b(äh|ähs|ähm|hm|hmm)\b\s*[,;:\-\.]?\s*", "", result, flags=re.IGNORECASE
    )
    # Remove underlines:
    result = re.sub(r"_", "", result)
    # Removing quotation marks:
    result = re.sub(r'[\'"]', "", result)
    # Remove all forms of blank lines:
    result = re.sub(r"^\s*$\n", "", result, flags=re.MULTILINE)
    # Removing all new lines
    # result = re.sub(r'\n', ' ', result)

    # Additional removal of all punctuation if requested:
    if remove_punctuation:
        # Remove all punctuation except . and : after numbers:
        punctuation_to_remove = string.punctuation.replace(".", "").replace(":", "")
        result = re.sub(
            r"(?<!\d)[{}]+".format(re.escape(punctuation_to_remove)), "", result
        )
        # Additional removal of punctuation that does not
        # follow numbers:
        result = re.sub(r"(?<=\D)[.:]+", "", result)

    return result

def run_on_directory():
    "Run through all VTT files in the specified directory."
    config = get_config()["vtt_to_txt"]
    input_path = Path(config["input_directory"])
    output_path = Path(config["output_directory"])

    for filepath in input_path.glob("*.vtt"):
        orig_stem = filepath.stem
        vtt_content = filepath.read_text(encoding="utf-8")

        # Write cleaned up output file (first version).
        cleaned_text = clean_vtt_content(vtt_content)
        new_filepath = output_path / f"{orig_stem}.cleared.txt"
        new_filepath.write_text(cleaned_text, encoding="utf-8")
        print(f"Cleaned text was saved in: {new_filepath}")

if __name__ == "__main__":
    run_on_directory()