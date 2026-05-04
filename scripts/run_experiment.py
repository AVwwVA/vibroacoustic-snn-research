#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from vibro_snn_research.experiments import run_experiment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    result = run_experiment(args.config)
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
