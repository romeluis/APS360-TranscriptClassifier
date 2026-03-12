"""
Shared data loading utilities for evaluation and visualization.

Loads transcript JSON files produced by the transcript generator.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_dataset(data_dir: str) -> list[dict]:
    """Load all transcript JSON files from an output directory.

    Args:
        data_dir: path to the transcript output directory
                  (e.g. transcript_generator/output/)

    Returns:
        list of dicts, each with keys: tokens, ner_tags, metadata
    """
    data_path = Path(data_dir)
    samples = []

    for json_file in sorted(data_path.rglob("transcript_*.json")):
        with open(json_file) as f:
            sample = json.load(f)
        if "tokens" in sample and "ner_tags" in sample:
            samples.append(sample)

    return samples


def group_by_template(samples: list[dict]) -> dict[str, list[dict]]:
    """Group samples by template_id from metadata."""
    groups = defaultdict(list)
    for s in samples:
        tid = s.get("metadata", {}).get("template_id", "unknown")
        groups[tid].append(s)
    return dict(groups)


def load_single(json_path: str) -> dict:
    """Load a single transcript JSON file.

    Args:
        json_path: path to one transcript_NNN.json file

    Returns:
        dict with keys: tokens, ner_tags, metadata
    """
    with open(json_path) as f:
        return json.load(f)
