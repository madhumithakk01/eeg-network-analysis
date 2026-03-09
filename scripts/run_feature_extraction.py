#!/usr/bin/env python3
"""
Entry point for the feature extraction pipeline.

Processes EEG: preprocessing -> connectivity -> graph feature extraction.
Designed to run per-worker with a subset of patient IDs; reads from
Google Drive and writes to FEATURE_OUTPUT_PATH and INTERMEDIATE_OUTPUT_PATH.

Usage (planned): worker index and patient range via CLI or env.
Implementation not yet added.
"""

def main():
    print("Feature extraction pipeline: not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
