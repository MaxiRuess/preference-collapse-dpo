#!/usr/bin/env python3
"""Step 2: Generate response pairs (4 personas per prompt) + consensus pairs."""

import argparse

import yaml

from src.generation import generate_consensus_pairs, generate_response_pairs
from src.personas import get_all_personas
from src.prompts import load_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate response pairs")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--consensus-only", action="store_true", help="Only generate consensus pairs")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus pairs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prompts = load_prompts(config["paths"]["prompts_file"])
    print(f"Loaded {len(prompts)} prompts")

    if not args.consensus_only:
        personas = get_all_personas()
        print(f"Generating responses for {len(prompts)} prompts × {len(personas)} personas...")
        generate_response_pairs(
            prompts, personas, config, config["paths"]["responses_file"]
        )
        print(f"Responses saved to {config['paths']['responses_file']}")

    if not args.skip_consensus:
        n = config["generation"]["consensus_pair_count"]
        print(f"Generating {n} consensus pairs...")
        generate_consensus_pairs(
            prompts[:n], config, config["paths"]["consensus_file"]
        )
        print(f"Consensus pairs saved to {config['paths']['consensus_file']}")


if __name__ == "__main__":
    main()
