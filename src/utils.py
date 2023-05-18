"""Utility functions for the project."""
import argparse
import json
import os


def save_config(args: argparse.Namespace) -> None:
    """Save the config to a file."""
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)
