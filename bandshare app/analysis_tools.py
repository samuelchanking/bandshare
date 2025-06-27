# analysis_tools.py

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import streamlit as st
from typing import List, Dict

def convert_cumulative_to_daily(cumulative_data: List[Dict]) -> List[Dict]:
    """
    Converts a time-series of cumulative data to daily data.
    """
    if not cumulative_data or len(cumulative_data) < 2:
        return cumulative_data

    df = pd.DataFrame(cumulative_data)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    df['value'] = df['value'].diff().fillna(df['value'])
    df['value'] = df['value'].clip(lower=0)

    return df.to_dict('records')


@st.cache_data
def detect_anomalous_spikes(data_tuple, discretization_step=7, alpha=0.2, sensitivity=2):
    """
    Detects anomalous spikes from CUMULATIVE streaming data by analyzing the volatility
    of the rate of change.
    """
    def format_error(msg):
        return json.dumps({'anomalies': [], 'debug': None, 'error': msg})
    
    dates_str, cumulative_streams_list = data_tuple
    
    if not dates_str or not cumulative_streams_list:
        return format_error("Input data is empty.")

    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]
    streams = np.array(cumulative_streams_list)
    
    start_date = min(dates)
    days = np.array([(d - start_date).days for d in dates])

    if len(np.unique(days)) < 2:
        return format_error("Not enough unique days to perform interpolation.")
    interpolator = interp1d(days, streams, kind='linear', fill_value='extrapolate')

    max_days = max(days)
    if max_days < discretization_step:
        return format_error(f"Data range ({int(max_days)} days) is smaller than the discretization step ({discretization_step} days).")
    grid_days = np.arange(0, max_days + 1, discretization_step)
    s = interpolator(grid_days)

    Av = np.zeros_like(s)
    Av[0] = s[0]
    for i in range(1, len(s)):
        Av[i] = alpha * s[i] + (1 - alpha) * Av[i-1]

    # --- NEW CORRECTED LOGIC ---
    # 1. Calculate normalized finite differences (rate of change)
    s_diff = np.diff(s) / discretization_step
    Av_diff = np.diff(Av) / discretization_step
    
    # 2. Calculate the absolute difference between these rates
    abs_diff_dev = np.abs(s_diff - Av_diff)
    
    # 3. Calculate the smoothed standard deviation (EMA) of these differences
    smoothed_std = np.zeros_like(abs_diff_dev)
    if len(abs_diff_dev) > 0:
        smoothed_std[0] = abs_diff_dev[0]
        for i in range(1, len(abs_diff_dev)):
            smoothed_std[i] = alpha * abs_diff_dev[i] + (1 - alpha) * smoothed_std[i-1]

    # 4. Detect anomalous jumps
    anomalies = []
    for i in range(len(abs_diff_dev)):
        # Compare the actual difference in rates to a threshold based on the smoothed difference
        threshold = sensitivity * smoothed_std[i]
        if abs_diff_dev[i] > threshold:
            # The jump size corresponds to the original s array at this step
            jump_size = s[i+1] - s[i]
            anomaly_date = start_date + timedelta(days=int(grid_days[i+1]))
            anomalies.append({
                'date': anomaly_date.strftime('%Y-%m-%d'),
                'jump_size': float(jump_size)
            })
    # --- END OF NEW LOGIC ---

    debug_info = {
        'grid_days': grid_days.tolist(),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'discretized_cumulative': s.tolist(),
        'ema_cumulative': Av.tolist(),
        's_diffs': s_diff.tolist(),
        'Av_diffs': Av_diff.tolist(),
        'abs_diff_devs': abs_diff_dev.tolist(),
        'smoothed_stds': smoothed_std.tolist()
    }

    return json.dumps({'anomalies': anomalies, 'debug': debug_info, 'error': ''})
