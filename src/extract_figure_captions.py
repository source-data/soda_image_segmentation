# src/extract_figure_captions.py

import json
import argparse
from bs4 import BeautifulSoup
import os

def extract_captions(input_json, output_jsonl):
    """
    Extract figure captions from the annotated JSON file and save to a JSONL file.

    Args:
        input_json (str): Path to the input JSON file.
        output_jsonl (str): Path to the output JSONL file.
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    with open(output_jsonl, 'w') as f:
        for item in data:
            figure_id = item['data']['figure_id']
            raw_caption = item['data']['caption']
            plain_caption = BeautifulSoup(raw_caption, 'html.parser').get_text()

            output = {
                "figure_id": figure_id,
                "figure_caption": plain_caption
            }
            f.write(json.dumps(output) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Extract figure captions from annotated data")
    parser.add_argument('--input', type=str, required=True, help="Path to the input annotated_data.json file")
    parser.add_argument('--output', type=str, required=True, help="Path to the output JSONL file")

    args = parser.parse_args()

    extract_captions(args.input, args.output)

if __name__ == "__main__":
    main()
