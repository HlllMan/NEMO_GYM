#!/usr/bin/env python3
"""
Test script for standalone NemoGym score functions.

This script reads test samples from test_samples.jsonl,
retrieves extra_info from the parquet file using the saved index,
and passes data directly to score_fn to mimic verl reward manager behavior.

NOTE: Uses HuggingFace datasets (same as verl) instead of pandas.
      Handles both native dict and JSON-serialized extra_info (v5+ parquet).

Usage:
    python test_score_fns.py

Requirements:
    - test_samples.jsonl with model outputs and parquet indices
    - parquet file at the expected location
"""

import json
import sys
from pathlib import Path

import datasets

# Add mjnemogym to path
sys.path.insert(0, str(Path(__file__).parent))

from mjnemogym import score_fn_dict


PARQUET_PATH = Path(__file__).parent.parent.parent / "preproc_nemogym/test_output/test-512.parquet"
SAMPLES_PATH = Path(__file__).parent / "test_samples.jsonl"


def test_all_domains():
    """Test all domain score functions."""

    # Load parquet using HuggingFace datasets (same as verl)
    # HuggingFace datasets automatically converts numpy arrays to Python lists
    print(f"Loading parquet from: {PARQUET_PATH}")
    df = datasets.load_dataset("parquet", data_files=str(PARQUET_PATH))["train"]
    print(f"Dataset size: {len(df)}")

    # Load test samples
    print(f"\nLoading test samples from: {SAMPLES_PATH}")
    samples = []
    with open(SAMPLES_PATH) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} test samples")

    # Test each sample
    results = []
    for sample in samples:
        data_source = sample["data_source"]
        parquet_index = sample["parquet_index"]
        model_output = sample["model_output"]

        print(f"\n{'='*60}")
        print(f"Testing {data_source} (parquet index: {parquet_index})")
        print(f"{'='*60}")

        # Get extra_info from dataset
        row = df[parquet_index]
        extra_info = row["extra_info"]

        # Handle JSON-serialized extra_info (from v5+ parquet files)
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        print(f"Data source in parquet: {row['data_source']}")

        # Get score function
        if data_source not in score_fn_dict:
            print(f"ERROR: Unknown data_source: {data_source}")
            results.append((data_source, "ERROR", "Unknown data_source"))
            continue

        score_fn = score_fn_dict[data_source]

        # For workplace domain, need to handle response_output
        if data_source == "nemogym_workplace":
            # Parse model_output as response_output list
            try:
                response_output = json.loads(model_output)
                extra_info["response_output"] = response_output
                model_output = ""  # Clear since we use response_output
            except json.JSONDecodeError:
                pass

        # Call score function
        try:
            reward = score_fn(model_output, extra_info)
            print(f"Reward: {reward}")
            results.append((data_source, reward, "OK"))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((data_source, "ERROR", str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0
    for data_source, reward, status in results:
        if status == "OK" and reward > 0:
            print(f"PASS: {data_source} -> {reward}")
            passed += 1
        elif status == "OK":
            print(f"ZERO: {data_source} -> {reward}")
            failed += 1
        else:
            print(f"FAIL: {data_source} -> {status}")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed/zero")
    return passed, failed


if __name__ == "__main__":
    import multiprocessing
    # Fix macOS fork issue
    multiprocessing.set_start_method("fork", force=True)

    passed, failed = test_all_domains()
    sys.exit(0 if failed == 0 else 1)
