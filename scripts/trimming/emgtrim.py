"""
EMG and IMU Data Trimming Utility.

Description:
    This script processes raw EMG and IMU data (Myo format) by trimming 
    it to an experimental window and synchronizing it with action labels. 
    It converts microsecond timestamps to Unix seconds for cross-sensor 
    alignment.

Author: Kartikay Naswa (@kartikaynaswa)

Preconditions:
    - Raw Myo CSV file containing 'timestamp', 'emg_*', and 'imu_*' columns.
    - Action/Label CSV file with 'Unix Timestamp' and 'Label'.

Postconditions:
    - Generates a trimmed EMG/IMU CSV file with synchronized action labels.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_emg(myo_path, action_path, output_path, emg_cols=None, imu_cols=None):
    """Trims and synchronizes EMG/IMU data."""
    emg_cols = emg_cols or [f'emg_{i}' for i in range(8)]
    imu_cols = imu_cols or ['quat_w', 'quat_x', 'quat_y', 'quat_z', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    # 1. LOAD ACTIONS AND BOUNDARIES
    df_actions = pd.read_csv(action_path)
    start_unix = df_actions['Unix Timestamp'].min()
    end_unix = df_actions['Unix Timestamp'].max()

    # 2. LOAD AND TRIM RAW EMG
    # The Myo file uses # for comments and microsecond timestamps
    df_myo = pd.read_csv(myo_path, comment='#')
    df_myo.columns = df_myo.columns.str.strip()
    
    # Convert Microseconds to Seconds to match Action/EEG timestamps [cite: 24]
    df_myo['Unix_Timestamp'] = df_myo['timestamp'] / 1e6
    
    # Trim to experimental window
    df_myo = df_myo[(df_myo['Unix_Timestamp'] >= start_unix) & 
                    (df_myo['Unix_Timestamp'] <= end_unix)].copy()

    # 3. SYNCHRONIZATION (FILL-FORWARD LOGIC) [cite: 426]
    myo_times = df_myo['Unix_Timestamp'].values
    
    # Initialize with the first action (start_relax) [cite: 432]
    first_label = df_actions['Label'].iloc[0]
    labels = np.full(len(myo_times), first_label, dtype='object')

    for i in range(len(df_actions)):
        row = df_actions.iloc[i]
        
        # Fill labels until the NEXT action begins
        if i < len(df_actions) - 1:
            next_start_unix = df_actions.iloc[i+1]['Unix Timestamp']
            mask = (myo_times >= row['Unix Timestamp']) & (myo_times < next_start_unix)
        else:
            mask = (myo_times >= row['Unix Timestamp'])
            
        labels[mask] = row['Label']

    # 4. PREPARE EXPORT DATAFRAME
    # Keep only EMG, IMU, Timestamp, and the new Label
    keep_cols = ['Unix_Timestamp'] + emg_cols + imu_cols
    export_df = df_myo[keep_cols].copy()
    export_df.insert(1, 'Action_Label', labels)

    # 5. EXPORT TO CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_df.to_csv(output_path, index=False)
    logger.info(f"Successfully exported {len(export_df)} EMG samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim EMG data based on action timestamps.")
    parser.add_argument("--myo", required=True, help="Path to raw Myo .csv file")
    parser.add_argument("--action", required=True, help="Path to action .csv file")
    parser.add_argument("--output", required=True, help="Path for the trimmed output CSV")

    args = parser.parse_args()
    process_emg(
        myo_path=args.myo,
        action_path=args.action,
        output_path=args.output
    )