#!/usr/bin/env python3
from __future__ import annotations
import argparse
from vibro_snn_research.analysis import run_analysis

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', default='outputs/analysis')
    args = parser.parse_args()
    outputs = run_analysis(args.manifest, args.output_dir)
    for name, path in outputs.items():
        print(f'{name}: {path}')
if __name__ == '__main__':
    main()
