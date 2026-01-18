#!/usr/bin/env python3
"""Generate test_index.csv from test_hourly.csv"""
import pandas as pd
from pathlib import Path
from datetime import timedelta

HIST_DAYS = 7
FUTURE_DAYS = 14

test = pd.read_csv('data/test_hourly.csv')
daily = test.copy()
daily['date'] = pd.to_datetime(daily['time']).dt.floor('D')
daily = (daily.groupby(['station_name','date'])
         .agg(sea_level=('sea_level','mean'),
              sea_level_max=('sea_level','max'))
         .reset_index())

rows = []
for stn, g in daily.groupby('station_name'):
    g = g.sort_values('date').reset_index(drop=True)
    for anchor in g['date'].unique():
        hist_start = pd.to_datetime(anchor) - pd.Timedelta(days=HIST_DAYS)
        hist_end   = pd.to_datetime(anchor) - pd.Timedelta(days=1)
        future_end = pd.to_datetime(anchor) + pd.Timedelta(days=FUTURE_DAYS-1)
        if (g['date'].min() <= hist_start) and (g['date'].max() >= future_end):
            rows.append({
                'station_name': stn,
                'hist_start': hist_start.date().isoformat(),
                'hist_end':   hist_end.date().isoformat(),
                'future_start': pd.to_datetime(anchor).date().isoformat(),
                'future_end':   future_end.date().isoformat()
            })
test_index = pd.DataFrame(rows).reset_index().rename(columns={'index':'id'})
test_index.to_csv('data/test_index.csv', index=False)
print(f'Generated test_index.csv with {len(test_index)} rows')
