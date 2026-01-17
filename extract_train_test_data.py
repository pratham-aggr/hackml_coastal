#!/usr/bin/env python3
"""
Simple script to extract train and test data from the .mat file.
This replicates the logic from ingestion.py for easy data extraction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.io import loadmat

# Station splits (as defined in ingestion.py)
TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 
                     'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']

def matlab2datetime(matlab_datenum):
    """Convert MATLAB datenum to Python datetime."""
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=float(matlab_datenum) % 1) - timedelta(days=366)

def extract_data(mat_file_path, output_dir=None):
    """
    Extract train and test data from .mat file.
    
    Parameters:
    -----------
    mat_file_path : str or Path
        Path to NEUSTG_19502020_12stations.mat file
    output_dir : str or Path, optional
        Directory to save CSV files. If None, data is returned without saving.
    
    Returns:
    --------
    tuple : (train_df, test_df, all_stations_df)
        DataFrames with columns: time, station_name, latitude, longitude, sea_level
    """
    mat_path = Path(mat_file_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    
    print(f"Loading data from {mat_path}...")
    d = loadmat(str(mat_path))
    
    # Extract data
    lat = d['lattg'].flatten()
    lon = d['lontg'].flatten()
    sea_level = d['sltg']  # Shape: (time, stations)
    station_names = [s[0] for s in d['sname'].flatten()]
    time = d['t'].flatten()
    time_dt = np.array([matlab2datetime(t) for t in time], dtype='datetime64[ns]')
    
    T, S = sea_level.shape
    print(f"Data shape: {T} time points × {S} stations")
    print(f"Stations: {station_names}")
    
    # Create hourly DataFrame for all stations
    df_hourly = pd.DataFrame({
        "time": np.tile(pd.to_datetime(time_dt), S),
        "station_name": np.repeat(station_names, T),
        "latitude": np.repeat(lat, T),
        "longitude": np.repeat(lon, T),
        "sea_level": sea_level.reshape(-1, order="F")
    })
    
    # Split into train and test
    train = df_hourly[df_hourly["station_name"].isin(TRAINING_STATIONS)].copy()
    test = df_hourly[df_hourly["station_name"].isin(TESTING_STATIONS)].copy()
    
    print(f"\nTrain stations ({len(TRAINING_STATIONS)}): {TRAINING_STATIONS}")
    print(f"Train data shape: {train.shape}")
    print(f"\nTest stations ({len(TESTING_STATIONS)}): {TESTING_STATIONS}")
    print(f"Test data shape: {test.shape}")
    
    # Save to CSV if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_csv = output_dir / "train_hourly.csv"
        test_csv = output_dir / "test_hourly.csv"
        
        train.to_csv(train_csv, index=False)
        test.to_csv(test_csv, index=False)
        
        print(f"\nSaved train data to: {train_csv}")
        print(f"Saved test data to: {test_csv}")
    
    return train, test, df_hourly

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract train and test data from .mat file")
    parser.add_argument("--mat_file", 
                       default="NEUSTG_19502020_12stations.mat",
                       help="Path to NEUSTG_19502020_12stations.mat file")
    parser.add_argument("--output_dir",
                       default="data",
                       help="Directory to save CSV files (default: ./data)")
    
    args = parser.parse_args()
    
    train_df, test_df, all_df = extract_data(args.mat_file, args.output_dir)
    
    print("\n=== Data Preview ===")
    print("\nTrain data head:")
    print(train_df.head())
    print("\nTest data head:")
    print(test_df.head())
