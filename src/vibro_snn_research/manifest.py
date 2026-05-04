from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import zipfile
import pandas as pd
DATASET_DIRNAME = 'University of Ottawa Ball-bearing Vibration and Acoustic Fault Data under Constant Load and Speed Conditions (UODS-VAFDC)'
TRAIN_IDS = {1, 2, 3, 6, 7, 8, 16, 17, 18}
VAL_IDS = {4, 9, 19}
TEST_IDS = {5, 10, 20}
PRIMARY_HEALTHY_IDS = set(range(1, 11)) | {16, 17, 18, 19, 20}
FAMILY_MAP = {'H': 'healthy', 'I': 'inner_race', 'O': 'outer_race', 'B': 'ball', 'C': 'cage'}
HEALTH_MAP = {0: 'healthy', 1: 'fault_developing', 2: 'faulty'}
PATTERN = re.compile('(?P<letter>[HIOBC])_(?P<bearing>\\d+)_(?P<state>\\d)\\.mat$')

@dataclass(slots=True)
class RecordMetadata:
    record_id: str
    bearing_id: int
    fault_family: str
    health_state: int
    health_stage: str
    binary_label: int
    split: str
    subset: str
    primary_included: bool
    path: str

def parse_record_name(filename: str) -> RecordMetadata:
    match = PATTERN.search(filename)
    if not match:
        raise ValueError(f'Unsupported Ottawa record name: {filename}')
    letter = match.group('letter')
    bearing_id = int(match.group('bearing'))
    health_state = int(match.group('state'))
    fault_family = FAMILY_MAP[letter]
    health_stage = HEALTH_MAP[health_state]
    binary_label = 0 if letter == 'H' else 1
    if bearing_id in TRAIN_IDS:
        split = 'train'
    elif bearing_id in VAL_IDS:
        split = 'val'
    elif bearing_id in TEST_IDS:
        split = 'test'
    else:
        split = 'secondary'
    primary_included = letter != 'B' and bearing_id in PRIMARY_HEALTHY_IDS
    if letter in {'I', 'O', 'C'}:
        primary_included = True
    subset = 'primary' if primary_included and split != 'secondary' else 'secondary'
    return RecordMetadata(record_id=f'{letter}_{bearing_id}_{health_state}', bearing_id=bearing_id, fault_family=fault_family, health_state=health_state, health_stage=health_stage, binary_label=binary_label, split=split, subset=subset, primary_included=primary_included and split != 'secondary', path='')

def extract_dataset(zip_path: str | Path, output_dir: str | Path) -> list[Path]:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []
    with zipfile.ZipFile(zip_path, 'r') as archive:
        for member in archive.namelist():
            if not member.endswith('.mat'):
                continue
            relative_name = Path(member).name
            destination = output_dir / relative_name
            if not destination.exists():
                with archive.open(member) as source, destination.open('wb') as target:
                    shutil.copyfileobj(source, target)
            extracted_paths.append(destination.resolve())
    extracted_paths.sort()
    return extracted_paths

def build_manifest(zip_path: str | Path, output_dir: str | Path, manifest_path: str | Path | None=None) -> pd.DataFrame:
    raw_files = extract_dataset(zip_path, output_dir)
    records: list[dict[str, object]] = []
    for raw_path in raw_files:
        meta = parse_record_name(raw_path.name)
        records.append({'record_id': meta.record_id, 'bearing_id': meta.bearing_id, 'fault_family': meta.fault_family, 'health_state': meta.health_state, 'health_stage': meta.health_stage, 'binary_label': meta.binary_label, 'split': meta.split, 'subset': meta.subset, 'primary_included': int(meta.primary_included), 'path': str(raw_path)})
    frame = pd.DataFrame(records).sort_values(['subset', 'split', 'bearing_id', 'record_id']).reset_index(drop=True)
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(manifest_path, index=False)
    return frame
