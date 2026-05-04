#!/usr/bin/env python3
from __future__ import annotations
import argparse
from vibro_snn_research.reporting import aggregate_experiment_outputs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-root', default='outputs/experiments')
    parser.add_argument('--output-dir', default='outputs/reports')
    args = parser.parse_args()
    table_path, summary_path, pareto_path = aggregate_experiment_outputs(args.experiments_root, args.output_dir)
    print(f'table: {table_path}')
    print(f'summary: {summary_path}')
    print(f'pareto: {pareto_path}')
if __name__ == '__main__':
    main()
