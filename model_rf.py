import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder


def matlab2datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)


def load_mat(path):
    data = loadmat(path)
    lat = data['lattg'].flatten()
    lon = data['lontg'].flatten()
    sea_level = data['sltg']
    station_names = [s[0] for s in data['sname'].flatten()]
    time = data['t'].flatten()
    time_dt = np.array([matlab2datetime(t) for t in time])
    return {
        'lat': lat,
        'lon': lon,
        'sea_level': sea_level,
        'station_names': station_names,
        'time_dt': time_dt
    }


def build_daily_df(matdata, selected_stations=None):
    lat = matdata['lat']
    lon = matdata['lon']
    station_names = matdata['station_names']
    time_dt = pd.to_datetime(matdata['time_dt'])
    sea_level = matdata['sea_level']

    if selected_stations is None:
        selected_idx = list(range(len(station_names)))
        selected_names = station_names
    else:
        selected_idx = [station_names.index(s) for s in selected_stations]
        selected_names = [station_names[i] for i in selected_idx]

    selected_lat = lat[selected_idx]
    selected_lon = lon[selected_idx]
    selected_sea_level = sea_level[:, selected_idx]

    # Build hourly DataFrame
    df_hourly = pd.DataFrame({
        'time': np.tile(time_dt, len(selected_names)),
        'station_name': np.repeat(selected_names, len(time_dt)),
        'latitude': np.repeat(selected_lat, len(time_dt)),
        'longitude': np.repeat(selected_lon, len(time_dt)),
        'sea_level': selected_sea_level.flatten()
    })

    # Compute flood threshold per station (mean + 1.5*std)
    threshold_df = df_hourly.groupby('station_name')['sea_level'].agg(['mean','std']).reset_index()
    threshold_df['flood_threshold'] = threshold_df['mean'] + 1.5 * threshold_df['std']
    df_hourly = df_hourly.merge(threshold_df[['station_name','flood_threshold']], on='station_name', how='left')

    # Daily aggregation
    df_daily = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')]).agg({
        'sea_level': 'mean',
        'latitude': 'first',
        'longitude': 'first',
        'flood_threshold': 'first'
    }).reset_index()

    hourly_max = df_hourly.groupby(['station_name', pd.Grouper(key='time', freq='D')])['sea_level'].max().reset_index()
    df_daily = df_daily.merge(hourly_max, on=['station_name','time'], suffixes=('','_max'))
    df_daily['flood'] = (df_daily['sea_level_max'] > df_daily['flood_threshold']).astype(int)

    # Feature engineering: rolling stats and lags
    df_daily['sea_level_3d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df_daily['sea_level_7d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_daily['sea_level_14d_mean'] = df_daily.groupby('station_name')['sea_level'].transform(lambda x: x.rolling(14, min_periods=1).mean())

    # Lags
    for lag in [1,2,3,7]:
        df_daily[f'sl_lag_{lag}'] = df_daily.groupby('station_name')['sea_level'].shift(lag)

    # Temporal features
    df_daily['dayofyear'] = df_daily['time'].dt.dayofyear
    df_daily['month'] = df_daily['time'].dt.month

    # Station label encoding for model-ready numeric column
    le = LabelEncoder()
    df_daily['station_id'] = le.fit_transform(df_daily['station_name'])

    # Forward-fill lag-induced NaNs per station
    df_daily = df_daily.groupby('station_name').apply(lambda g: g.fillna(method='bfill').fillna(method='ffill')).reset_index(drop=True)

    return df_daily


def build_train_test(df_daily, hist_days=7, future_days=14, features=None, hist_start=None):
    if features is None:
        features = ['sea_level', 'sea_level_3d_mean', 'sea_level_7d_mean', 'sea_level_14d_mean',
                    'sl_lag_1','sl_lag_2','sl_lag_3','sl_lag_7','dayofyear','month','station_id']

    X_train, y_train = [], []
    for stn, grp in df_daily.groupby('station_name'):
        grp = grp.sort_values('time').reset_index(drop=True)
        for i in range(len(grp) - hist_days - future_days):
            hist = grp.loc[i:i+hist_days-1, features].values.flatten()
            future = grp.loc[i+hist_days:i+hist_days+future_days-1, 'flood'].values
            if len(hist) == hist_days * len(features) and len(future) == future_days:
                X_train.append(hist)
                y_train.append(future)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train, y_train


def make_model(random_state=42, n_estimators=200, max_depth=12):
    base = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state, class_weight='balanced')
    model = MultiOutputClassifier(base, n_jobs=-1)
    return model


def save_model(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_model(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict(model, X):
    return model.predict(X)
