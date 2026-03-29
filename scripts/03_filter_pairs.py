#!/usr/bin/env python3
"""Step 3: Filter response pairs by quality and contrast using LLM-as-judge."""

import argparse
import json
from pathlib import Path

import yaml

from src.filtering import filter_responses
from src.generation import load_responses


def main():
    parser = argparse.ArgumentParser(description="Filter response pairs")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    responses = load_responses(config["paths"]["responses_file"])
    print(f"Loaded {len(responses)} response records")

    scores_path = config["paths"].get("judge_scores_file", "data/judge_scores.jsonl")
    filtered = filter_responses(responses, config, scores_path)

    # Save filtered output
    output_path = Path(config["paths"]["filtered_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in filtered:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(filtered)} filtered records to {output_path}")
    print(f"Scores saved to {scores_path} (for inspection)")


if __name__ == "__main__":
    main()
