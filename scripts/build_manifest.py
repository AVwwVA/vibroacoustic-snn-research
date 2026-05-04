#!/usr/bin/env python3
from __future__ import annotations
import argparse
from vibro_snn_research.manifest import build_manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip-path', default='University of Ottawa Ball-bearing Vibration and Acoustic Fault Data under Constant Load and Speed Conditions (UODS-VAFDC).zip')
    parser.add_argument('--extract-dir', default='data/uods_vafdc/raw')
    parser.add_argument('--manifest-path', default='outputs/manifests/ottawa_manifest.csv')
    args = parser.parse_args()
    frame = build_manifest(args.zip_path, args.extract_dir, args.manifest_path)
    print(f'Saved manifest with {len(frame)} records to {args.manifest_path}')
if __name__ == '__main__':
    main()
