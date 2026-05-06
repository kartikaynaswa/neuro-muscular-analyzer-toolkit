"""
Multi-Sensor Data Collection File Matching Utility.

Description:
    This script scans directories for EEG (OpenBCI), EMG (Myo), and Action/Label files.
    It extracts temporal boundaries (start/end Unix timestamps) and matches sensor 
    data to action sessions with configurable temporal tolerance.

Author: Kartikay Naswa (@kartikaynaswa)

Preconditions:
    - Directories contain .zip archives or extracted sensor data (.txt, .csv).
    - EEG files are .txt, follow OpenBCI header format and start with '%OpenBCI Raw EXG Data'
    - EMG files are csv, follow Myo data format and start with '# Recording started:'
    - Action files are csv and start with 'Label,Event,Unix Timestamp'
    - Assumes only 1 matching file of each type.

Postconditions:
    - Generates a CSV manifest mapping actions to the best available sensor files.
    - Reports gaps (start/end error) where sensor coverage is incomplete.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Maximum allowed gap (seconds) between Action and Sensor data
TOLERANCE = 2.9
OUTPUT_FILENAME = './data/synchronized/session_manifest_with_errors.csv'


def identify_file_type(file_path: Union[str, Path]) -> Optional[str]:
    """Identifies the sensor/data type by inspecting the file header.

    Returns:
        'EEG', 'EMG', 'Action', or None if the format is unrecognized.
    """
    try:
        with open(file_path, 'r', errors='ignore') as f:
            first_line = f.readline().strip()

            if first_line.startswith('%OpenBCI Raw EXG Data'):
                return 'EEG'
            if first_line.startswith('# Recording started:'):
                return 'EMG'
            if first_line.startswith('Label,Event,Unix Timestamp'):
                return 'Action'
    except (IOError, OSError) as e:
        logger.debug(f"Skipping file {file_path}: {e}")
    return None


def get_fast_boundaries(file_path: Union[str, Path], file_type: str) -> Optional[Tuple[float, float]]:
    """Extracts start/end timestamps using format-specific optimized parsing.

    Args:
        file_path: Path to the data file.
        file_type: Identified type ('EEG', 'EMG', or 'Action').

    Returns:
        Tuple of (start_timestamp, end_timestamp) or None on failure.
    """
    try:
        if file_type == 'EEG':
            # OpenBCI: Header line 5 (skip 4)
            df_first = pd.read_csv(file_path, skiprows=4, nrows=1)
            df_first.columns = df_first.columns.str.strip()
            start_ts = float(df_first['Timestamp'].iloc[0])
            ts_idx = list(df_first.columns).index('Timestamp')

            with open(file_path, 'rb') as f:
                # Jump back 2000 bytes or to the start of the file if smaller
                file_size = os.path.getsize(file_path)
                f.seek(max(0, file_size - 2000), os.SEEK_SET)
                lines = f.readlines()
                last_line = lines[-1].decode().strip().split(',')
                end_ts = float(last_line[ts_idx])
            return start_ts, end_ts

        if file_type == 'EMG':
            # Myo: Header line 15 (skip 14)
            df_first = pd.read_csv(file_path, skiprows=14, nrows=1)
            start_ts = df_first['timestamp'].iloc[0] / 1_000_000

            with open(file_path, 'rb') as f:
                # Jump back 2000 bytes or to the start of the file if smaller
                file_size = os.path.getsize(file_path)
                f.seek(max(0, file_size - 2000), os.SEEK_SET)
                lines = f.readlines()
                last_line = lines[-1].decode().strip().split(',')
                end_ts = float(last_line[0]) / 1_000_000
            return start_ts, end_ts

        if file_type == 'Action':
            df = pd.read_csv(file_path)
            return df['Unix Timestamp'].min(), df['Unix Timestamp'].max()

    except (pd.errors.EmptyDataError, ValueError, KeyError, IndexError) as e:
        logger.warning(f"Metadata extraction failed for {file_path}: {e}")
    return None


def unzip_recursive(directory: Union[str, Path]) -> None:
    """Finds and extracts ZIP archives recursively in-place."""
    directory = Path(directory)
    found_new = False
    for zip_path in directory.rglob("*.zip"):
        extract_to = zip_path.with_name(zip_path.name.replace('.zip', '_extracted'))
        if not extract_to.exists():
            logger.info(f"Extracting: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(extract_to)
                found_new = True
            except zipfile.BadZipFile:
                logger.error(f"Corrupt ZIP file detected: {zip_path}")

    if found_new:
        unzip_recursive(directory)


def main(starting_paths: List[str]) -> None:
    """Executes the file identification, indexing, and matching pipeline."""
    for path in starting_paths:
        unzip_recursive(path)

    all_files = [
        f for p in starting_paths
        for f in Path(p).rglob("*")
        if f.is_file()
    ]

    eeg_meta, emg_meta, action_meta = [], [], []

    logger.info("Indexing files and extracting timestamps...")
    for f in all_files:
        ftype = identify_file_type(f)
        if not ftype: continue

        bounds = get_fast_boundaries(f, ftype)
        if not bounds: continue

        meta = {'path': str(f), 'start': bounds[0], 'end': bounds[1], 'type': ftype}
        if ftype == 'EEG': eeg_meta.append(meta)
        elif ftype == 'EMG': emg_meta.append(meta)
        elif ftype == 'Action': action_meta.append(meta)

    results = []
    logger.info(f"Starting matching process for {len(action_meta)} action sessions...")

    for act in action_meta:
        # Relaxed Temporal Match
        matches_eeg = [
            e for e in eeg_meta
            if e['start'] <= act['start'] + TOLERANCE and e['end'] >= act['end'] - TOLERANCE
        ]
        matches_emg = [
            m for m in emg_meta
            if m['start'] <= act['start'] + TOLERANCE and m['end'] >= act['end'] - TOLERANCE
        ]

        # Select the first valid match for each sensor type
        eeg = matches_eeg[0] if matches_eeg else None
        emg = matches_emg[0] if matches_emg else None

        # Calculate metadata and error gaps
        results.append({
            'Session_Timestamp': act['start'],
            'Action_File': os.path.relpath(act['path']),
            'EEG_File': os.path.relpath(eeg['path']) if eeg else 'null',
            'EMG_File': os.path.relpath(emg['path']) if emg else 'null',
            'EEG_Start_Error': max(0, eeg['start'] - act['start']) if eeg else 0,
            'EEG_End_Error': max(0, act['end'] - eeg['end']) if eeg else 0,
            'EMG_Start_Error': max(0, emg['start'] - act['start']) if emg else 0,
            'EMG_End_Error': max(0, act['end'] - emg['end']) if emg else 0
        })

    df = pd.DataFrame(results).sort_values('Session_Timestamp')
    df.to_csv(OUTPUT_FILENAME, index=False)
    logger.info(f"Manifest exported to: {OUTPUT_FILENAME}")
    logger.info(f"Successfully processed {len(df)} sessions.")


if __name__ == "__main__":
    # multiple paths can be added in any order
    SEARCH_PATHS = [
        './data'
    ]
    main(SEARCH_PATHS)