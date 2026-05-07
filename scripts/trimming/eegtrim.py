"""
EEG Data Trimming and Filtering Utility.

Description:
    This script processes raw EEG data (OpenBCI format) by trimming it to 
    the boundaries of an experimental session, applying bandpass and notch 
    filters, and synchronizing it with event labels using a fill-forward method.

Author: Kartikay Naswa (@kartikaynaswa)

Preconditions:
    - Raw EEG data in TXT format (OpenBCI).
    - Action/Label CSV file with 'Unix Timestamp' and 'Label'.

Postconditions:
    - Generates a filtered EEG CSV file with synchronized labels.
    - Optionally generates a trimmed but unfiltered EEG CSV file.
"""

import argparse
import logging
import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_eeg(eeg_path, action_path, filtered_output_path, trimmed_output_path=None, ch_names=None, sfreq=250):
    """Trims, filters, and synchronizes EEG data."""
    ch_names = ch_names or ['C3', 'C4', 'F3', 'F4', 'Cz', 'P3', 'P4', 'Fz']

    # 1. LOAD ACTIONS AND BOUNDARIES
    # Identify the start and end of the experimental session
    df_actions = pd.read_csv(action_path)
    start_unix = df_actions['Unix Timestamp'].min()
    end_unix = df_actions['Unix Timestamp'].max()

    # 2. LOAD AND TRIM RAW EEG
    # Discard data outside the start_relax and SESSION_END window for efficiency
    df_eeg = pd.read_csv(eeg_path, comment='%')
    df_eeg.columns = df_eeg.columns.str.strip()
    df_eeg = df_eeg[(df_eeg['Timestamp'] >= start_unix) & (df_eeg['Timestamp'] <= end_unix)].copy()

    # 3. CONVERT TO MNE AND FILTER
    exg_cols = [f'EXG Channel {i}' for i in range(8)]
    eeg_values = df_eeg[exg_cols].values.T / 1e6  # Convert uV to Volts
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_values, info)

    # Apply 1-50Hz Bandpass and 50Hz Notch filters
    raw.filter(l_freq=1.0, h_freq=50.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)

    # 4. SYNCHRONIZATION (FILL-FORWARD LOGIC)
    # Align therapy-relevant gesture labels with the EEG time-series
    eeg_times = df_eeg['Timestamp'].values
    
    # Initialize with the first action (start_relax) to avoid "Unlabeled" gaps
    first_label = df_actions['Label'].iloc[0]
    labels = np.full(len(eeg_times), first_label, dtype='object')

    for i in range(len(df_actions)):
        row = df_actions.iloc[i]
        
        # Fill labels until the NEXT action begins or the session ends
        if i < len(df_actions) - 1:
            next_start_unix = df_actions.iloc[i+1]['Unix Timestamp']
            mask = (eeg_times >= row['Unix Timestamp']) & (eeg_times < next_start_unix)
        else:
            mask = (eeg_times >= row['Unix Timestamp'])
            
        labels[mask] = row['Label']

    # 5. EXPORT TRIMMED (UNFILTERED) IF REQUESTED
    if trimmed_output_path:
        unfiltered_vals = df_eeg[exg_cols].values  # Values remain in uV as loaded
        export_df_trimmed = pd.DataFrame(unfiltered_vals, columns=ch_names)
        export_df_trimmed.insert(0, 'Action_Label', labels)
        export_df_trimmed.insert(0, 'Unix_Timestamp', eeg_times)
        
        os.makedirs(os.path.dirname(trimmed_output_path), exist_ok=True)
        export_df_trimmed.to_csv(trimmed_output_path, index=False)
        logger.info(f"Successfully exported trimmed (unfiltered) to {trimmed_output_path}")

    # 6. PREPARE AND EXPORT FILTERED DATAFRAME
    # Convert back to microvolts for standard EEG analysis
    filtered_eeg_uv = raw.get_data().T * 1e6 
    export_df_filtered = pd.DataFrame(filtered_eeg_uv, columns=ch_names)
    export_df_filtered.insert(0, 'Action_Label', labels)
    export_df_filtered.insert(0, 'Unix_Timestamp', eeg_times)

    os.makedirs(os.path.dirname(filtered_output_path), exist_ok=True)
    export_df_filtered.to_csv(filtered_output_path, index=False)
    logger.info(f"Successfully exported filtered to {filtered_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim EEG data based on action timestamps.")
    parser.add_argument("--eeg", required=True, help="Path to raw EEG .txt file")
    parser.add_argument("--action", required=True, help="Path to action .csv file")
    parser.add_argument("--output", required=True, help="Path for the filtered output CSV")
    parser.add_argument("--trimmed_output", help="Optional path for the trimmed (unfiltered) output CSV")
    
    args = parser.parse_args()
    process_eeg(
        eeg_path=args.eeg,
        action_path=args.action,
        filtered_output_path=args.output,
        trimmed_output_path=args.trimmed_output
    )