# Sensor Data Trimming and Alignment Stage

This directory contains the scripts responsible for synchronizing, filtering, and trimming raw EEG and EMG data based on experimental session timestamps.

## Files Overview

- **`main_pipeline.py`**: The central orchestrator. It reads the session manifest, identifies the file paths for each session, and calls the specialized trimming functions for EEG and EMG data.
- **`eegtrim.py`**: Handles EEG data processing. It applies bandpass (1-50Hz) and notch (60Hz) filters using MNE-Python, trims the data to the session window, and synchronizes action labels using a fill-forward approach.
- **`emgtrim.py`**: Handles Myo EMG and IMU data. It converts microsecond timestamps to Unix seconds for alignment, trims the data, and maps action labels to the time-series.

## Requirements

Ensure you have the following Python packages installed:
```bash
pip install pandas numpy mne
```

## Inputs

The pipeline expects the following structure (as defined in the manifest):
- A manifest file at `data/synchronized/session_manifest_with_errors.csv`.
- Raw OpenBCI EEG files (.txt).
- Raw Myo EMG files (.csv).

## Outputs

Processed files are saved to `data/trimmed/` with the following naming conventions:
- `eeg_filtered_<session_id>.csv`: Bandpass and notch filtered EEG data with labels.
- `eeg_trimmed_<session_id>.csv`: Raw (unfiltered) EEG data trimmed to the session window.
- `emg_trimmed_<session_id>.csv`: Trimmed and synchronized EMG/IMU data.

## How to Run

To process all sessions listed in the manifest, run the main pipeline script from the project root:

```bash
python scripts/trimming/main_pipeline.py
```