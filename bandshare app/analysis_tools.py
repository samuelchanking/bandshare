# analysis_tools.py

import json
import numpy as np
import sys
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d, UnivariateSpline
import streamlit as st
import matplotlib.dates as mdates
from typing import List, Dict, Tuple, Any
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os # <-- ADDED IMPORT
import contextlib # <-- ADDED IMPORT
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.preprocessing import MinMaxScaler
import pywt
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.optimize import minimize

# Add this function to analysis_tools.py

def remove_sudden_flats_in_cumulative(series: pd.Series, window=3, max_run_length=2, growth_threshold=0.8):
    """
    Removes short flat runs (zero growth) in a cumulative time series if surrounded by mostly growing values.
    Replaces all but the first point in the flat run with NaN for later interpolation.
    Now time-aware: normalizes growth to rates (streams per day) using precise time deltas.
    
    Args:
        series (pd.Series): Cumulative time series with datetime index.
        window (int): Number of days before and after to check (total window = 2*window + run length).
        max_run_length (int): Maximum consecutive flats (duplicates) to consider for removal.
        growth_threshold (float): Fraction of window rates that must be positive to flag as anomalous (0-1).
    
    Returns:
        pd.Series: Series with short flat runs (except first point) replaced by NaN.
    """
    series = series.copy()
    # Identify groups of consecutive identical values
    shift_diff = (series != series.shift())
    group_key = shift_diff.cumsum()
    for key, group in series.groupby(group_key):
        run_length = len(group)
        if run_length > 1 and (run_length - 1) <= max_run_length:  # Flats are run_length - 1
            group_indices = group.index
            # Window around the group, excluding the group itself
            start_idx = group_indices[0] - pd.Timedelta(days=window)
            end_idx = group_indices[-1] + pd.Timedelta(days=window)
            window_series = series.loc[start_idx:end_idx].drop(group_indices, errors='ignore')
            
            if window_series.empty:
                continue
            
            # Compute time-normalized rates in the window
            diff_values = window_series.diff().dropna()
            delta_times = (window_series.index[1:] - window_series.index[:-1]).total_seconds() / 86400.0
            rates = diff_values / delta_times
            
            if rates.empty:
                continue
            
            # Fraction of positive growth in window (on rates)
            positive_growth_fraction = (rates > 0).sum() / len(rates)
            
            if positive_growth_fraction >= growth_threshold:
                # Set all but the first in the group to NaN
                series.loc[group_indices[1:]] = np.nan
    
    return series

def remove_sudden_zeros(series: pd.Series, window=3, max_run_length=2, non_zero_threshold=0.8):
    """
    Removes isolated zero values or short runs in a time series if surrounded by mostly non-zero values.
    Replaces them with NaN so the curve can connect over the missing dates.
    
    Args:
        series (pd.Series): Time series with datetime index.
        window (int): Number of days before and after to check (total window = 2*window + run length).
        max_run_length (int): Maximum consecutive zeros to consider for removal (e.g., 2 removes singles/doubles).
        non_zero_threshold (float): Fraction of window values that must be non-zero to flag as outlier (0-1).
    
    Returns:
        pd.Series: Series with sudden zeros/runs replaced by NaN.
    """
    series = series.copy()
    zero_mask = (series == 0)
    zero_groups = (zero_mask != zero_mask.shift()).cumsum()[zero_mask]
    
    for group in zero_groups.unique():
        group_indices = zero_groups[zero_groups == group].index
        run_length = len(group_indices)
        if run_length > max_run_length:
            continue  # Preserve longer runs (likely legitimate)
        
        # Define the full window around the run
        start_idx = group_indices[0] - pd.Timedelta(days=window)
        end_idx = group_indices[-1] + pd.Timedelta(days=window)
        
        # Get values in the window, excluding the run itself
        window_values = series.loc[start_idx:end_idx].drop(group_indices, errors='ignore')
        
        if window_values.empty:
            continue
        
        # Calculate the fraction of non-zero values in the window
        non_zero_fraction = (window_values != 0).sum() / len(window_values)
        
        # Remove if the window is mostly non-zero (meets threshold)
        if non_zero_fraction >= non_zero_threshold:
            series.loc[group_indices] = np.nan
    
    return series

def smooth_lumped_around_nans(series, threshold_multiplier=1.5, window=3, max_gap_length=2):
    """
    Detects and smooths 'lumped' high values adjacent to NaN gaps by distributing the lump across the gap.
    Optional: Now uses a time-weighted local average if points are not uniformly spaced.
    """
    series = series.copy()
    indices = series.index  # DatetimeIndex
    nan_mask = series.isna()
    nan_groups = (nan_mask != nan_mask.shift()).cumsum()[nan_mask]
    
    for group in nan_groups.unique():
        group_indices = nan_groups[nan_groups == group].index
        gap_length = len(group_indices)
        if gap_length > max_gap_length:
            continue
        
        # Get positions in the index list
        group_pos = [series.index.get_loc(idx) for idx in group_indices]
        before_pos = group_pos[0] - 1
        after_pos = group_pos[-1] + 1
        
        if before_pos < 0 or after_pos >= len(indices):
            continue
        
        before_idx = indices[before_pos]
        after_idx = indices[after_pos]
        
        val_before = series.loc[before_idx]
        val_after = series.loc[after_idx]
        
        if pd.isna(val_before) or pd.isna(val_after):
            continue
        
        # Define window for local average, excluding the gap, before, and after
        start_pos = group_pos[0] - window
        end_pos = group_pos[-1] + window
        window_indices = indices[max(start_pos, 0):min(end_pos + 1, len(indices))]
        exclude_indices = [before_idx, after_idx] + list(group_indices)
        window_values = series.loc[window_indices].drop(exclude_indices, errors='ignore')
        
        if window_values.empty or len(window_values) < 2:
            continue
        
        # Time-weighted local average (using inverse distance weights for simplicity)
        window_times = (window_values.index - before_idx).total_seconds() / 86400.0  # Days from reference
        weights = 1 / (np.abs(window_times) + 1e-6)  # Inverse distance, avoid div0
        local_avg = np.average(window_values, weights=weights)
        
        # Check if before or after is unusually high
        lump_pos = None
        lump_val = None
        if val_before > threshold_multiplier * local_avg:
            lump_pos = before_pos
            lump_val = val_before
            num_days = gap_length + 1
            dist_start_pos = before_pos
            dist_end_pos = group_pos[-1]
        elif val_after > threshold_multiplier * local_avg:
            lump_pos = after_pos
            lump_val = val_after
            num_days = gap_length + 1
            dist_start_pos = group_pos[0]
            dist_end_pos = after_pos
        
        if lump_pos is not None:
            distributed_val = lump_val / num_days
            dist_indices = indices[dist_start_pos: dist_end_pos + 1]
            series.loc[dist_indices] = distributed_val
    
    return series

def find_spikes_by_std_dev(series: pd.Series, threshold: float, window_size: int):
    """
    Finds local maxima and minima that are a significant number of standard
    deviations away from a rolling average, and also meet a relative deviation threshold of 80%.

    Args:
        series (pd.Series): Input time series data.
        threshold (float): The number of standard deviations for the threshold.
        window_size (int): The window size for the rolling average and standard deviation.

    Returns:
        pd.DataFrame: A DataFrame of detected spikes.
    """
    if series.empty or len(series) < window_size:
        return pd.DataFrame()

    # Calculate rolling average and standard deviation
    rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()

    # Find all local peaks and troughs
    maxima_indices, _ = find_peaks(series.values, distance=1)
    minima_indices, _ = find_peaks(-series.values, distance=1)
    all_extrema_indices = np.unique(np.concatenate((maxima_indices, minima_indices)))

    spikes = []
    for idx in all_extrema_indices:
        value = series.iloc[idx]
        mean_at_idx = rolling_mean.iloc[idx]
        std_at_idx = rolling_std.iloc[idx]

        # Check if the standard deviation is valid and the point exceeds the threshold
        if pd.notna(std_at_idx) and std_at_idx > 0:
            deviation = abs(value - mean_at_idx)
            if deviation > (threshold * std_at_idx):
                # Additional criterion: relative deviation >= 80% of the mean
                # Skip if mean is zero or negative to avoid invalid relative checks
                if mean_at_idx > 0 and (deviation / mean_at_idx) >= 0.2:
                    if idx in maxima_indices:
                        reason = "Maximum"
                    elif idx in minima_indices:
                        reason = "Minimum"
                    else:
                        reason = "Unknown"
                    
                    spikes.append({'date': series.index[idx], 'streams': value, 'type': reason})
            
    if not spikes:
        return pd.DataFrame()

    return pd.DataFrame(spikes).set_index('date').sort_index()


def detect_wavelet_spikes(data_tuple: Tuple, wavelet: str = 'db4', sensitivity: float = 3.0) -> str:
    """
    Detects spikes in daily time-series data using wavelet decomposition.
    This version adds the spike markers directly to the figure before serialization to fix UI errors.
    """
    def format_error(msg: str) -> str:
        return json.dumps({'anomalies': [], 'error': msg, 'plot_json': None})

    WAVELET_FAMILY_MAP = {
        'daubechies': 'db4', 'symlets': 'sym8', 'coiflets': 'coif5',
        'biorthogonal': 'bior3.3', 'reverse biorthogonal': 'rbio3.3',
        'discrete meyer': 'dmey', 'haar': 'haar'
    }

    try:
        wavelet_name_lower = wavelet.lower()
        if wavelet_name_lower in WAVELET_FAMILY_MAP:
            wavelet_to_use = WAVELET_FAMILY_MAP[wavelet_name_lower]
        elif wavelet_name_lower in pywt.wavelist(kind='discrete'):
            wavelet_to_use = wavelet_name_lower
        else:
            family_names = [f.lower() for f in pywt.families()]
            if wavelet_name_lower in family_names:
                 wavelist = pywt.wavelist(wavelet, 'discrete')
                 if wavelist:
                     wavelet_to_use = wavelist[0]
                 else:
                     raise ValueError(f"Could not find a default wavelet for family '{wavelet}'.")
            else:
                wavelet_to_use = wavelet

        dates_str, streams_list = data_tuple
        if not dates_str or not streams_list or len(dates_str) < 2:
            return format_error("Input data is empty or contains insufficient data points.")

        df = pd.DataFrame({
            'date': pd.to_datetime(list(dates_str)),
            'streams': list(streams_list)
        }).sort_values(by='date').set_index('date')

        streams = df['streams'].dropna()
        if len(streams) < 4:
            return format_error("Not enough data points for wavelet decomposition.")

        coeffs = pywt.wavedec(streams, wavelet_to_use)
        anomalies = []

        num_levels = len(coeffs)
        subplot_titles = [f'Original Signal (using {wavelet_to_use})'] + [f'Level {i} Details (cD{i})' for i in range(1, num_levels)][::-1]
        fig = make_subplots(
            rows=num_levels, cols=1, shared_xaxes=True,
            subplot_titles=subplot_titles, vertical_spacing=0.03
        )
        fig.add_trace(go.Scatter(x=streams.index, y=streams, mode='lines', name='Original Signal'), row=1, col=1)

        for i, detail_coeffs in enumerate(coeffs[1:]):
            level = i + 1
            median_abs_dev = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
            if median_abs_dev == 0:
                threshold = np.std(detail_coeffs) * sensitivity if np.std(detail_coeffs) > 0 else 1.0
            else:
                threshold = (median_abs_dev / 0.6745) * sensitivity

            coeff_timeline = pd.date_range(start=streams.index.min(), end=streams.index.max(), periods=len(detail_coeffs))
            fig.add_trace(go.Scatter(x=coeff_timeline, y=detail_coeffs, mode='lines', name=f'cD{level} Details'), row=level + 1, col=1)
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=level + 1, col=1)
            fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=level + 1, col=1)

            if level == 1:
                spike_indices = np.where(np.abs(detail_coeffs) > threshold)[0]
                for idx in spike_indices:
                    original_signal_index = int(np.floor(idx * (len(streams) / len(detail_coeffs))))
                    if 0 <= original_signal_index < len(streams):
                        anomaly_date = streams.index[original_signal_index]
                        value = streams.iloc[original_signal_index]
                        anomalies.append({
                            'date': anomaly_date.strftime('%Y-%m-%d'),
                            'streams': float(value),
                            'type': 'Spike (Wavelet)'
                        })

        # --- BUG FIX ---
        # Add spike markers to the plot here, before it's converted to JSON.
        # This ensures the plot is complete and avoids errors in the UI.
        if anomalies:
            anomalies_df_for_plot = pd.DataFrame(anomalies)
            anomalies_df_for_plot['date'] = pd.to_datetime(anomalies_df_for_plot['date'])
            fig.add_trace(go.Scatter(
                x=anomalies_df_for_plot['date'],
                y=anomalies_df_for_plot['streams'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Detected Spike'
            ), row=1, col=1)

        fig.update_layout(height=200 * num_levels, showlegend=False, title_text="Wavelet Decomposition Analysis")
        plot_json = fig.to_json()

        # Finalize anomalies list for the table
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies).sort_values('streams', ascending=False)
            anomalies_df = anomalies_df.drop_duplicates(subset='date', keep='first')
            anomalies = anomalies_df.to_dict('records')

        return json.dumps({'anomalies': anomalies, 'error': '', 'plot_json': plot_json})

    except Exception as e:
        return format_error(f"An unexpected error occurred during wavelet analysis: {e}. Please check the selected wavelet family.")


def handle_cumulative_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently removes duplicate consecutive values from a cumulative series
    by comparing the growth rate before and after the duplicate pair.
    It now includes extra logic to look for and remove an anomalous jump
    that occurs one day prior to the duplicate pair, if the growth around the
    duplicate pair itself is stable.
    """
    # We need at least 5 data points to apply the new look-behind logic.
    if len(df) < 5:
        # Fallback to the original simpler logic for shorter dataframes.
        return df.loc[df['value'].diff() != 0].copy()

    df = df.reset_index(drop=True)
    indices_to_drop = set()

    # Iterate through the dataframe to find duplicate pairs (p2, p3) within a 5-point window.
    for i in range(len(df) - 4):
        # Define indices for the 5-point window (p0, p1, p2, p3, p4).
        p0_idx, p1_idx, p2_idx, p3_idx, p4_idx = i, i + 1, i + 2, i + 3, i + 4
        
        # Skip if any index in our window is already marked for deletion.
        if any(idx in indices_to_drop for idx in [p0_idx, p1_idx, p2_idx, p3_idx, p4_idx]):
            continue

        val_p2, val_p3 = df.loc[p2_idx, 'value'], df.loc[p3_idx, 'value']

        # Check for a consecutive duplicate value.
        if val_p2 == val_p3:
            val_p0, val_p1, val_p4 = df.loc[p0_idx, 'value'], df.loc[p1_idx, 'value'], df.loc[p4_idx, 'value']
            
            # Calculate the growth (delta) for the three relevant periods.
            delta_pre_jump = val_p1 - val_p0
            delta_before = val_p2 - val_p1
            delta_after = val_p4 - val_p3

            # --- MODIFIED LOGIC ---
            
            # Condition 1: Check if growth rates around the duplicate pair are "similar".
            # This suggests the anomaly is not adjacent to the duplicates.
            are_similar = False
            if delta_before > 0 and delta_after > 0:
                ratio = delta_before / delta_after
                # Growth rates are considered similar if they are within a factor of 2 of each other.
                if 0.5 < ratio < 2.0:
                    are_similar = True

            # Condition 2: If growth is similar, check if the jump one day prior was abnormally large.
            is_pre_jump_large = False
            if are_similar:
                avg_normal_growth = (delta_before + delta_after) / 2
                # A pre-jump is large if it's more than double the average of the subsequent "normal" growth.
                if avg_normal_growth > 0 and delta_pre_jump > (1.8 * avg_normal_growth):
                    is_pre_jump_large = True
            
            if is_pre_jump_large:
                # **Special Case**: Anomaly detected at p1 (the day before the stable period).
                # As requested, remove p1 (the date with the anomalous jump) so it can be interpolated.
                indices_to_drop.add(p1_idx)
                
                # We must also resolve the original duplicate pair (p2, p3).
                # With p1 removed, the growth before the pair is now (p2 - p0), which is very large.
                # Following the standard logic, this means p2 is the erroneous part of the pair.
                indices_to_drop.add(p2_idx)
                
            else:
                # **Normal Case**: No prior anomaly found or growth was not similar.
                # Use the original logic: remove the duplicate that causes the larger disruption.
                if delta_before >= delta_after:
                    indices_to_drop.add(p2_idx)
                else:
                    indices_to_drop.add(p3_idx)

    # Perform the final cleanup.
    if indices_to_drop:
        df_cleaned = df.drop(list(indices_to_drop))
    else:
        df_cleaned = df.copy()
        
    # A final pass handles any remaining simple duplicates.
    df_final = df_cleaned.loc[df_cleaned['value'].diff() != 0].copy()
    
    return df_final.reset_index(drop=True)

# --- MODIFIED HELPER FUNCTION TO STANDARDIZE CLEANING ---
def clean_and_prepare_cumulative_data(
    history_data: List[Dict]
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Takes raw cumulative history, applies cleaning rules, and returns both the
    cleaned DataFrame and a list of dates that were removed for decreasing.
    Now includes timestamp adjustment for better data cleaning.
    """
    if not history_data:
        return pd.DataFrame(), pd.to_datetime([])

    parsed_history = []
    for entry in history_data:
        value = None
        if 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
            value = entry['plots'][0].get('value')
        elif 'value' in entry:
            value = entry.get('value')

        if entry.get('date') and value is not None:
            parsed_history.append({'date': entry['date'], 'value': value})

    if not parsed_history:
        return pd.DataFrame(), pd.to_datetime([])

    df = pd.DataFrame(parsed_history)

    # Apply timestamp adjustment
    dates_str = df['date'].tolist()
    values = df['value'].tolist()
    adjust_json = adjust_json
    adjust = json.loads(adjust_json)

    if adjust['error']:
        # Proceed with original data if adjustment fails
        pass
    else:
        adjusted = adjust['adjusted_data']
        df = pd.DataFrame({
            'date': [pd.to_datetime(item['timestamp']) for item in adjusted],
            'value': [item['cumulative_streams'] for item in adjusted]
        })

    df.sort_values(by='date', inplace=True)

    # Identify and store dates where the cumulative value erroneously decreased
    decreasing_mask = df['value'].diff() < 0
    decreasing_dates = df.loc[decreasing_mask, 'date']
    
    # Remove the rows with decreasing values
    df_processed = df[~decreasing_mask].reset_index(drop=True)
    
    # Apply the intelligent duplicate removal logic (adapted for datetime index)
    df_cleaned = handle_cumulative_duplicates(df_processed)

    return df_cleaned, decreasing_dates

    df = pd.DataFrame(parsed_history)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Remove duplicate dates, keeping the last recorded value for that day
    df_processed = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

    # Identify and store dates where the cumulative value erroneously decreased
    decreasing_mask = df_processed['value'].diff() < 0
    decreasing_dates = df_processed.loc[decreasing_mask, 'date']
    
    # Remove the rows with decreasing values
    df_processed = df_processed[~decreasing_mask].reset_index(drop=True)
    
    # Apply the intelligent duplicate removal logic
    df_cleaned = handle_cumulative_duplicates(df_processed)

    return df_cleaned, decreasing_dates


def convert_cumulative_to_daily(cumulative_data: List[Dict]) -> List[Dict]:
    """
    Converts a time-series of cumulative data to daily data.
    """
    if not cumulative_data or len(cumulative_data) < 2:
        return cumulative_data

    df = pd.DataFrame(cumulative_data)
    df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
    df.sort_values(by='date', inplace=True)

    # Calculate day-over-day difference and ensure no negative values
    df['value'] = df['value'].diff().fillna(df['value'])
    df['value'] = df['value'].clip(lower=0)

    return df.to_dict('records')

@st.cache_data
def detect_anomalous_spikes(data_tuple: Tuple, discretization_step: int = 7, sensitivity: float = 2.0, smoothing_window_size: int = 7) -> str:
    """
    Detects anomalous spikes from CUMULATIVE streaming data by analyzing the volatility
    of the rate of change. It now applies an initial smoothing step to the raw cumulative
    data before discretization to reduce noise.
    """
    def format_error(msg: str) -> str:
        return json.dumps({'anomalies': [], 'debug': None, 'error': msg})

    dates_str, cumulative_streams_list = data_tuple

    if not dates_str or not cumulative_streams_list or len(dates_str) < 2:
        return format_error("Input data is empty or contains insufficient data points (requires at least 2).")

    try:
        df = pd.DataFrame({
            'date': pd.to_datetime(list(dates_str)),
            'streams': list(cumulative_streams_list)
        }).sort_values(by='date')

        # This simple removal is okay here as the main cleaning is done prior
        df = df.loc[df['streams'].diff() != 0].copy()
        df.set_index('date', inplace=True)

        # Interpolate to get a value for every day, which is necessary for the analysis
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        daily_cumulative = df.reindex(full_date_range).interpolate(method='time')

        # Smooth the CUMULATIVE data first to reduce noise before discretization
        smoothed_cumulative = daily_cumulative['streams'].rolling(
            window=smoothing_window_size,
            center=True,
            min_periods=1
        ).mean()

        start_date = smoothed_cumulative.index.min()
        days = (smoothed_cumulative.index - start_date).days.to_numpy()
        streams = smoothed_cumulative.to_numpy()

    except Exception as e:
        return format_error(f"Failed during initial data preparation and smoothing: {e}")

    if len(np.unique(days)) < 2:
        return format_error("Not enough unique days to perform interpolation after initial smoothing.")

    # Create an interpolation function based on the smoothed cumulative data
    interpolator = interp1d(days, streams, kind='linear', fill_value='extrapolate')

    max_days = max(days)
    if max_days < discretization_step:
        return format_error(f"Data range ({int(max_days)} days) is smaller than the discretization step ({discretization_step} days).")

    # Create a discretized grid and get stream values at those points
    grid_days = np.arange(0, max_days + 1, discretization_step)
    s = interpolator(grid_days)

    # Smooth the discretized series to get the trend line 'Av'
    s_series = pd.Series(s)
    Av = s_series.rolling(window=smoothing_window_size, center=True, min_periods=1).mean().to_numpy()

    # Calculate the rate of change (daily average) for both series
    s_diff = np.diff(s) / discretization_step
    Av_diff = np.diff(Av) / discretization_step

    # Calculate the absolute deviation between the actual rate and the trend rate
    abs_diff_dev = np.abs(s_diff - Av_diff)

    # Smooth the deviation to create a dynamic threshold
    if len(abs_diff_dev) > 0:
        smoothed_std = pd.Series(abs_diff_dev).rolling(window=smoothing_window_size, center=True, min_periods=1).mean().to_numpy()
    else:
        smoothed_std = np.array([])

    anomalies = []
    for i in range(len(abs_diff_dev)):
        threshold = sensitivity * smoothed_std[i]
        # An anomaly is detected if the deviation exceeds the threshold
        if abs_diff_dev[i] > threshold and threshold > 0:

            spike_size_per_day = s_diff[i] - Av_diff[i]
            expected_rate_per_day = Av_diff[i]
            total_streams_on_day = s_diff[i]

            # Calculate relative significance to quantify the spike's importance
            if expected_rate_per_day > 1:
                relative_significance = spike_size_per_day / expected_rate_per_day
            else:
                relative_significance = float('inf') # Avoid division by zero

            anomaly_date = start_date + timedelta(days=int(grid_days[i+1]))
            anomalies.append({
                'date': anomaly_date.strftime('%Y-%m-%d'),
                'spike_size_streams_per_day': float(spike_size_per_day),
                'relative_significance': float(relative_significance),
                'total_streams_on_day': float(total_streams_on_day)
            })

    # Package debug information for plotting
    debug_info = {
        'grid_days': grid_days.tolist(),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'discretized_cumulative': s.tolist(),
        'boxcar_cumulative': Av.tolist(),
        's_diffs': s_diff.tolist(),
        'Av_diffs': Av_diff.tolist(),
        'abs_diff_devs': abs_diff_dev.tolist(),
        'smoothed_stds': smoothed_std.tolist()
    }

    return json.dumps({'anomalies': anomalies, 'debug': debug_info, 'error': ''})


def detect_additional_contribution(data, event_date, event_duration_days=30, smoothing_window_size=3):
    """
    Detects the effect of an event (e.g., playlist add) with a known duration using
    Prophet's holiday modeling. It automatically finds the optimal holiday prior scale
    and now returns the full forecast data for plotting.
    """
    try:
        # 1. Prepare data
        input_data = json.loads(data) if isinstance(data, str) else data
        df = pd.DataFrame({
            'ds': pd.to_datetime(input_data['dates']),
            'y': input_data['streams']
        })

        if smoothing_window_size > 1:
            df['y'] = df['y'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

        if df['ds'].duplicated().any():
            return json.dumps({'error': 'Duplicate dates detected in input data'})
        if len(df) < 14:
            return json.dumps({'error': 'Insufficient data points (minimum 14 days required)'})

        # 2. Define event holiday
        event_date_dt = datetime.strptime(event_date, '%Y-%m-%d')
        event_holiday = pd.DataFrame({
            'holiday': 'playlist_event',
            'ds': [event_date_dt + timedelta(days=i) for i in range(event_duration_days)],
            'lower_window': 0, 'upper_window': 0,
        })

        # 3. Find optimal scale
        best_scale = 0.01
        best_mse = float('inf')
        scales_to_test = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0]

        for scale in scales_to_test:
            with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
                model_temp = Prophet(
                    daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=True, holidays=event_holiday,
                    changepoint_prior_scale=0.05, holidays_prior_scale=scale
                )
                model_temp.fit(df)
            forecast_temp = model_temp.predict(df)
            mse = mean_squared_error(df['y'], forecast_temp['yhat'])
            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        # 4. Train final model
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            final_model = Prophet(
                daily_seasonality=True, weekly_seasonality=True,
                yearly_seasonality=True, holidays=event_holiday,
                changepoint_prior_scale=0.05, holidays_prior_scale=best_scale
            )
            final_model.fit(df)
        
        # 5. Get final forecast and holiday effect
        forecast = final_model.predict(df)
        holiday_effect = forecast[forecast['ds'].isin(event_holiday['ds'])][['ds', 'playlist_event']]
        if holiday_effect.empty:
            return json.dumps({'additional_contribution': {}, 'error': 'No holiday effect found for the specified event date'})

        avg_additional_streams = holiday_effect['playlist_event'].mean()
        if pd.isna(avg_additional_streams):
            avg_additional_streams = 0.0

        result_contribution = {
            'event_start_date': event_date_dt.strftime('%Y-%m-%d'),
            'event_duration_days': event_duration_days,
            'average_additional_streams_per_day': float(avg_additional_streams),
            'optimal_prior_scale': float(best_scale)
        }

        # --- NEW: Add forecast data to the output ---
        # Convert ds to string to ensure it's JSON serializable
        forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
        forecast_data = forecast.to_dict('records')
        
        return json.dumps({
            'additional_contribution': result_contribution, 
            'forecast_data': forecast_data, # Add forecast data here
            'error': ''
        })

    except Exception as e:
        return json.dumps({'error': f'Error processing data: {str(e)}'})

    
def detect_prophet_anomalies(data: Dict, interval_width: float = 0.95) -> str:
    """
    Detects anomalies in DAILY time-series data using Prophet's forecast intervals,
    determines the dominant seasonality, and returns INTERACTIVE PLOTS for visualization
    with a layout similar to other charts in the app.
    """
    try:
        # 1. Prepare DataFrame
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['dates']),
            'y': data['streams']
        })
        
        if df.empty or len(df) < 14:
             return json.dumps({
                'anomalies': [], 
                'error': "Insufficient data for anomaly detection (requires at least 14 days)."
            })

        # 2. Initialize, configure, and fit the Prophet model
        model = Prophet(interval_width=interval_width, daily_seasonality=True)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(df)
        
        # 3. Predict to get forecast bounds
        forecast = model.predict(df)
        
        # 4. Identify anomalies
        results_df = pd.concat([df.set_index('ds')['y'], forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]], axis=1).reset_index()
        results_df['anomaly'] = (results_df['y'] > results_df['yhat_upper']) | (results_df['y'] < results_df['yhat_lower'])
        anomalies = results_df[results_df['anomaly']].copy()

        anomalies_list = []
        if not anomalies.empty:
            anomalies['difference'] = anomalies['y'] - anomalies['yhat_upper']
            anomalies.loc[anomalies['y'] < anomalies['yhat_lower'], 'difference'] = anomalies['y'] - anomalies['yhat_lower']
            anomalies['ds'] = anomalies['ds'].dt.strftime('%Y-%m-%d')
            anomalies_list = anomalies[['ds', 'y', 'yhat', 'difference']].to_dict('records')

        # 5. Determine the best seasonality
        best_seasonality = "None"
        max_magnitude = -1
        seasonalities = {
            'yearly': forecast['yearly'] if 'yearly' in forecast else None,
            'weekly': forecast['weekly'] if 'weekly' in forecast else None,
            'daily': forecast['daily'] if 'daily' in forecast else None,
            'monthly': forecast['monthly'] if 'monthly' in forecast else None
        }
        for name, component in seasonalities.items():
            if component is not None:
                magnitude = np.mean(np.abs(component))
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    best_seasonality = name.capitalize()
        
        # 6. Generate and serialize INTERACTIVE plots
        fig_forecast = plot_plotly(model, forecast)

        # MODIFIED: Find the 'actuals' trace and change its color to white
        for trace in fig_forecast.data:
            # The actuals are plotted as black markers by default in prophet
            if trace.mode == 'markers' and trace.marker.color == 'black':
                trace.mode = 'lines+markers'
                trace.name = 'Actual Streams'
                trace.line = dict(color='white', width=1.5)
                trace.marker.color = 'white'
                trace.marker.size = 3
                break 
        
        if not anomalies.empty:
            fig_forecast.add_trace(go.Scatter(
                x=pd.to_datetime(anomalies['ds']), 
                y=anomalies['y'], 
                mode='markers', 
                marker=dict(color='red', size=8, symbol='x'), 
                name='Anomaly'
            ))

        fig_components = plot_components_plotly(model, forecast)
        
        fig_forecast.update_layout(
            yaxis_title="Daily Streams",
            hovermode='x unified'
        )
        fig_components.update_layout(
            hovermode='x unified'
        )

        # Serialize Plotly figures to JSON strings
        plots_json = {
            'forecast_plot': fig_forecast.to_json(),
            'components_plot': fig_components.to_json()
        }

        # 7. Package final results into JSON
        return json.dumps({
            'anomalies': anomalies_list,
            'best_seasonality': best_seasonality,
            'plots_json': plots_json,
            'error': ''
        })

    except Exception as e:
        return json.dumps({'anomalies': [], 'plots_json': None, 'error': f'An error occurred: {str(e)}'})



def load_and_process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    history = data.get('history', [])
    points = []
    for item in history:
        # Parse ISO with timezone
        dt_str = item['date']
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))  # Handle Z if present
        dt = dt.replace(tzinfo=None)  # Make naive (UTC assumed)
        val = item['value']
        points.append((dt, val))
    points.sort(key=lambda x: x[0])  # Sort by date
    
    # Remove consecutive duplicates in values
    if not points:
        return [], np.array([])
    filtered = [points[0]]
    for p in points[1:]:
        if p[1] != filtered[-1][1]:
            filtered.append(p)
    
    dates, cum_streams = zip(*filtered)
    return list(dates), np.array(cum_streams)


def robust_spline(t, y, s_smooth, iterations=2):
    w = np.ones(len(t))
    idx = np.argsort(t)
    t = t[idx]
    y = y[idx]
    for _ in range(iterations):
        spline = UnivariateSpline(t, y, w=w, s=s_smooth)
        r = y - spline(t)
        mad = np.median(np.abs(r))
        if mad == 0:
            break
        scale = mad / 0.67448975
        u = r / (scale * 4.685)
        mask = np.abs(u) < 1
        w = np.zeros_like(u)
        w[mask] = (1 - u[mask]**2)**2
    return spline

def adjust_and_plot_fast(input_path, output_path):
    dates, cum_streams = load_and_process_data(input_path)
    n = len(dates)
    final_spline = None
    min_time_diff = 1 / 24.0  # 1 hour in days
    epsilon = 1e-6
    if n < 3:
        h = np.full(n, 12.0)
        adjusted_times = [dates[i] + timedelta(hours=h[i]) for i in range(n)]
    else:
        ref_date = dates[0]
        t_orig = np.array([(d - ref_date).days for d in dates], dtype=float)
        midday_t = t_orig + 0.5
        t = midday_t.copy()
        
        # Initial min_t and max_t based on t_orig, enforcing min_time_diff
        min_t = np.zeros(n)
        max_t = np.zeros(n)
        min_t[0] = t_orig[0]
        max_t[0] = t_orig[0] + 1 - epsilon
        for i in range(1, n):
            min_t[i] = max(t_orig[i-1] + min_time_diff, t_orig[i])
            max_t[i] = t_orig[i] + 1 - epsilon
        
        # Compute second derivative for lambda step
        typical_second_deriv = 0.0
        if n >= 3:
            diff_t = np.maximum(np.diff(t_orig), min_time_diff)
            diff_t_prev = np.maximum(np.diff(t_orig[:-1]), min_time_diff)
            first_derivs = np.diff(cum_streams) / diff_t
            second_derivs = np.diff(first_derivs) / diff_t_prev
            typical_second_deriv = np.median(np.abs(second_derivs))
        
        # Compute initial s_smooth based on square-averaged first derivative
        if n >= 2:
            rates = np.diff(cum_streams) / np.maximum(np.diff(t_orig), min_time_diff)
            rms_rate = np.sqrt(np.mean(rates**2))
            max_deviation = rms_rate
            sigma = max_deviation / 2  # For most residuals < max_deviation
            s_smooth = n * sigma ** 2
        else:
            # Fallback for small datasets
            typical_rate = (cum_streams[-1] - cum_streams[0]) / (t_orig[-1] - t_orig[0]) if t_orig[-1] - t_orig[0] > 0 else 0.0
            rms_rate = typical_rate
            sigma = rms_rate / 2
            s_smooth = n * sigma ** 2
        
        # Iterate spline fitting and adjustment, 5 times
        num_iterations = 5
        for outer_iter in range(num_iterations):
            if outer_iter == num_iterations-1:
                # Update min_t and max_t based on current adjusted t for the last iteration
                min_t[0] = t[0]
                max_t[0] = t[0] + 1 - epsilon
                for i in range(1, n):
                    min_t[i] = max(t[i-1] + min_time_diff, t[i] - (1 - min_time_diff))
                    max_t[i] = t[i] + 1 - epsilon
            
            spline = robust_spline(t, cum_streams, s_smooth)
            final_spline = spline  # Store the last spline
            
            # Tune penalty coefficient lambda
            lam = 0.0
            step = typical_second_deriv ** 2 / 100.0 if typical_second_deriv > 0 else 1e-6
            max_tune_iter = 30
            for tune_iter in range(max_tune_iter):
                num_boundary = 0
                t_new = np.zeros(n)
                for i in range(n):
                    tts = np.linspace(min_t[i], max_t[i], 50)
                    cost = (spline(tts) - cum_streams[i])**2 + lam * ((tts - midday_t[i])/(max_t[i]-min_t[i]))**2
                    idx = np.argmin(cost)
                    t_new[i] = tts[idx]
                    if abs(t_new[i] - min_t[i]) < 1e-4 or abs(t_new[i] - max_t[i]) < 1e-4:
                        num_boundary += 1
                percentage = num_boundary / n
                if percentage <= 0.02:
                    t = t_new
                    break
                lam += step
                step *= 2  # Exponential increase
            
            # Update s_smooth based on current t for next iteration
            if n >= 2:
                diff_t = np.maximum(np.diff(t), min_time_diff)
                rates = np.diff(cum_streams) / diff_t
                rms_rate = np.sqrt(np.mean(rates**2))
                max_deviation = rms_rate
                sigma = max_deviation / 4
                s_smooth = n * sigma ** 2
            else:
                s_smooth *= 2  # Fallback increase
        
        # Very last smoothing step
        alpha = 0.8
        for _ in range(5):
            for i in range(1, n-1):
                d_left = cum_streams[i] - cum_streams[i-1]
                d_right = cum_streams[i+1] - cum_streams[i]
                a = t[i-1]
                b = t[i+1]
                # Analytical solution for t_i_optimal
                if d_left + d_right != 0 and b - a >= 2 * min_time_diff:
                    t_i_optimal = (d_left * b + d_right * a) / (d_left + d_right)
                else:
                    t_i_optimal = t[i]  # Fallback to current time
                # Compute new time
                t_new_i = (1 - alpha) * t[i] + alpha * t_i_optimal
                # Enforce bounds
                date_i = (ref_date + timedelta(seconds=t_new_i * 86400)).date()
                orig_date_i = (ref_date + timedelta(seconds=t_orig[i] * 86400)).date()
                date_i_minus_1 = (ref_date + timedelta(seconds=t_orig[i-1] * 86400)).date()
                date_i_plus_1 = (ref_date + timedelta(seconds=t_orig[i+1] * 86400)).date()
                if date_i == orig_date_i:
                    t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)
                elif date_i_minus_1 < date_i < date_i_plus_1:
                    t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)
                else:
                    # Clip to bounds [t[i-1] + min_time_diff, t[i+1] - min_time_diff]
                    t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)
        
        h = (t - t_orig) * 24
        adjusted_times = [ref_date + timedelta(seconds=t[i] * 86400) for i in range(n)]
    
    # Output adjusted JSON
    output_data = [
        {"timestamp": dt.isoformat(), "cumulative_streams": int(cum_streams[i])}
        for i, dt in enumerate(adjusted_times)
    ]
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Plots with date formatting and smaller markers
    ms = 3
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(dates, cum_streams, 'o-', label='Original (midnight)', markersize=ms)
    ax1.plot(adjusted_times, cum_streams, 'x-', label='Adjusted', markersize=ms)
    if final_spline is not None:
        # Plot the final spline
        t_min = min(t)
        t_max = max(t)
        t_spline = np.linspace(t_min, t_max, 1000)
        y_spline = final_spline(t_spline)
        spline_dates = [ref_date + timedelta(seconds=t_s * 86400) for t_s in t_spline]
        ax1.plot(spline_dates, y_spline, '--', label='Final Spline', color='green')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig1.autofmt_xdate()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Streams')
    ax1.legend()
    ax1.set_title('Cumulative Streams (Fast)')
    fig1.savefig('cumulative_fast.png')
    
    if n >= 2:
        # Derivative plot
        mid_times = [adjusted_times[i] + (adjusted_times[i+1] - adjusted_times[i]) / 2 for i in range(n - 1)]
        rates = []
        for i in range(n - 1):
            delta_t = (adjusted_times[i + 1] - adjusted_times[i]).total_seconds() / 86400.0
            delta_c = cum_streams[i + 1] - cum_streams[i]
            rate = delta_c / delta_t if delta_t > 0 else 0
            rates.append(rate)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(mid_times, rates, 'o-', markersize=ms)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig2.autofmt_xdate()
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Daily Rate (streams/day)')
        ax2.set_title('Adjusted Finite-Difference Derivative (Fast)')
        fig2.savefig('derivative_fast.png')
    
    # Adjustment plot
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(dates, h, 'o-', markersize=ms)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig3.autofmt_xdate()
    ax3.set_xlabel('Original Date')
    ax3.set_ylabel('Time Adjustment (hours)')
    ax3.set_title('Time Adjustments (Fast)')
    fig3.savefig('adjustment_fast.png')
    
    plt.show()  # Show all plots simultaneously
    
    print("Adjusted data saved to", output_path)
    print("Plots saved to cumulative_fast.png, derivative_fast.png, adjustment_fast.png")
    print("Interactive plots displayed on-screen.")

# Add this new function to the end of analysis_tools.py (before the if __name__ == "__main__" block)

def adjust_cumulative_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adjusts the timestamps of cumulative streaming history data using spline fitting and optimization,
    without generating plots or files. Returns the adjusted history in the same format as input.
    """
    # Extract and parse points
    points = []
    for item in history:
        dt_str = item['date']
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        dt = dt.replace(tzinfo=None)
        val = item['value']
        points.append((dt, val))
    points.sort(key=lambda x: x[0])

    # Remove consecutive duplicates in values
    if not points:
        return []
    filtered = [points[0]]
    for p in points[1:]:
        if p[1] != filtered[-1][1]:
            filtered.append(p)

    dates, cum_streams = zip(*filtered)
    dates = list(dates)
    cum_streams = np.array(cum_streams)
    n = len(dates)

    min_time_diff = 1 / 24.0  # 1 hour in days
    epsilon = 1e-6
    if n < 3:
        h = np.full(n, 12.0)
        adjusted_times = [dates[i] + timedelta(hours=h[i]) for i in range(n)]
        adjusted_data = [{"date": dt.isoformat() + 'Z', "value": int(cum_streams[i])} for i, dt in enumerate(adjusted_times)]
        return adjusted_data

    ref_date = dates[0]
    t_orig = np.array([(d - ref_date).days for d in dates], dtype=float)
    midday_t = t_orig + 0.5
    t = midday_t.copy()

    # Initial min_t and max_t based on t_orig, enforcing min_time_diff
    min_t = np.zeros(n)
    max_t = np.zeros(n)
    min_t[0] = t_orig[0]
    max_t[0] = t_orig[0] + 1 - epsilon
    for i in range(1, n):
        min_t[i] = max(t_orig[i-1] + min_time_diff, t_orig[i])
        max_t[i] = t_orig[i] + 1 - epsilon

    # Compute second derivative for lambda step
    typical_second_deriv = 0.0
    if n >= 3:
        diff_t = np.maximum(np.diff(t_orig), min_time_diff)
        diff_t_prev = np.maximum(np.diff(t_orig[:-1]), min_time_diff)
        first_derivs = np.diff(cum_streams) / diff_t
        second_derivs = np.diff(first_derivs) / diff_t_prev
        typical_second_deriv = np.median(np.abs(second_derivs))

    # Compute initial s_smooth based on square-averaged first derivative
    if n >= 2:
        rates = np.diff(cum_streams) / np.maximum(np.diff(t_orig), min_time_diff)
        rms_rate = np.sqrt(np.mean(rates**2))
        max_deviation = rms_rate
        sigma = max_deviation / 2  # For most residuals < max_deviation
        s_smooth = n * sigma ** 2
    else:
        # Fallback for small datasets
        typical_rate = (cum_streams[-1] - cum_streams[0]) / (t_orig[-1] - t_orig[0]) if t_orig[-1] - t_orig[0] > 0 else 0.0
        rms_rate = typical_rate
        sigma = rms_rate / 2
        s_smooth = n * sigma ** 2

    # Iterate spline fitting and adjustment, 5 times
    num_iterations = 5
    for outer_iter in range(num_iterations):
        if outer_iter == num_iterations-1:
            # Update min_t and max_t based on current adjusted t for the last iteration
            min_t[0] = t[0]
            max_t[0] = t[0] + 1 - epsilon
            for i in range(1, n):
                min_t[i] = max(t[i-1] + min_time_diff, t[i] - (1 - min_time_diff))
                max_t[i] = t[i] + 1 - epsilon

        spline = robust_spline(t, cum_streams, s_smooth)

        # Tune penalty coefficient lambda
        lam = 0.0
        step = typical_second_deriv ** 2 / 100.0 if typical_second_deriv > 0 else 1e-6
        max_tune_iter = 30
        for tune_iter in range(max_tune_iter):
            num_boundary = 0
            t_new = np.zeros(n)
            for i in range(n):
                tts = np.linspace(min_t[i], max_t[i], 50)
                cost = (spline(tts) - cum_streams[i])**2 + lam * ((tts - midday_t[i])/(max_t[i]-min_t[i]))**2
                idx = np.argmin(cost)
                t_new[i] = tts[idx]
                if abs(t_new[i] - min_t[i]) < 1e-4 or abs(t_new[i] - max_t[i]) < 1e-4:
                    num_boundary += 1
            percentage = num_boundary / n
            if percentage <= 0.02:
                t = t_new
                break
            lam += step
            step *= 2  # Exponential increase

        # Update s_smooth based on current t for next iteration
        if n >= 2:
            diff_t = np.maximum(np.diff(t), min_time_diff)
            rates = np.diff(cum_streams) / diff_t
            rms_rate = np.sqrt(np.mean(rates**2))
            max_deviation = rms_rate
            sigma = max_deviation / 4
            s_smooth = n * sigma ** 2
        else:
            s_smooth *= 2  # Fallback increase

    # Very last smoothing step
    alpha = 0.8
    for _ in range(5):
        for i in range(1, n-1):
            d_left = cum_streams[i] - cum_streams[i-1]
            d_right = cum_streams[i+1] - cum_streams[i]
            a = t[i-1]
            b = t[i+1]
            # Analytical solution for t_i_optimal
            if d_left + d_right != 0 and b - a >= 2 * min_time_diff:
                t_i_optimal = (d_left * b + d_right * a) / (d_left + d_right)
            else:
                t_i_optimal = t[i]  # Fallback to current time
            # Compute new time
            t_new_i = (1 - alpha) * t[i] + alpha * t_i_optimal
            # Enforce bounds
            date_i = (ref_date + timedelta(seconds=t_new_i * 86400)).date()
            orig_date_i = (ref_date + timedelta(seconds=t_orig[i] * 86400)).date()
            date_i_minus_1 = (ref_date + timedelta(seconds=t_orig[i-1] * 86400)).date()
            date_i_plus_1 = (ref_date + timedelta(seconds=t_orig[i+1] * 86400)).date()
            if date_i == orig_date_i:
                t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)
            elif date_i_minus_1 < date_i < date_i_plus_1:
                t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)
            else:
                # Clip to bounds [t[i-1] + min_time_diff, t[i+1] - min_time_diff]
                t[i] = min(max(t_new_i, t[i-1] + min_time_diff), t[i+1] - min_time_diff)

    adjusted_times = [ref_date + timedelta(seconds=t[i] * 86400) for i in range(n)]
    adjusted_data = [{"date": dt.isoformat(), "value": int(cum_streams[i])} for i, dt in enumerate(adjusted_times)]
    return adjusted_data


def analyze_song_streams(mid_times, rates, N=20, relative_threshold=1.6, absolute_threshold=50, epsilon=1e-6):
    """
    Detects spikes in daily streams using the logic from DataGuessing2 1.py, without plotting.
    Takes pre-computed mid_times (list of datetime) and rates (list of floats).
    Returns list of dicts with spike details, formatted to match app expectations.
    """
    m = len(rates)
    if m < 2:
        return []

    date_objs = mid_times  # Assuming mid_times are datetime objects
    s = rates

    # Compute global mean and std for z-score compatibility
    mean_s = np.mean(s)
    std_s = np.std(s) if np.std(s) > 0 else 1.0

    # Find local extrema
    extrema = []
    for j in range(m):
        if j == 0:
            if j + 1 < m:
                if s[j] > s[j + 1]:
                    extrema.append((j, 'max'))
                elif s[j] < s[j + 1]:
                    extrema.append((j, 'min'))
        elif j == m - 1:
            if s[j] > s[j - 1]:
                extrema.append((j, 'max'))
            elif s[j] < s[j - 1]:
                extrema.append((j, 'min'))
        else:
            is_max = s[j] > s[j - 1] and s[j] > s[j + 1]
            is_min = s[j] < s[j - 1] and s[j] < s[j + 1]
            if is_max:
                extrema.append((j, 'max'))
            elif is_min:
                extrema.append((j, 'min'))

    # Filter extrema based on thresholds
    kept_extrema = []
    for j, label in extrema:
        # Compute adjusted left_idx and right_idx
        left_idx = max(0, j - N)
        right_idx = min(m - 1, j + N)

        dt_window = (date_objs[right_idx] - date_objs[left_idx]).total_seconds() / 86400.0 if m > 1 else 1
        if dt_window <= 0:
            continue  # Skip invalid window

        # For daily rates, av is mean of s in window
        window_s = s[left_idx:right_idx+1]
        av = np.mean(window_s) if len(window_s) > 0 else 0

        # Check absolute threshold
        absolute_diff = abs(s[j] - av)
        if absolute_diff < absolute_threshold:
            continue

        # Check relative threshold
        if label == 'max':
            ratio = s[j] / av if av != 0 else float('inf')
            if ratio > relative_threshold:
                kept_extrema.append((j, label, av))
        elif label == 'min':
            denom = s[j] + epsilon
            ratio = av / denom if denom != 0 else float('inf')
            if ratio > relative_threshold:
                kept_extrema.append((j, label, av))

    # For each kept, compute i, f, area, and app-expected fields
    results = []
    for j, label, av in kept_extrema:
        dev_ext = s[j] - av
        sign_ext = get_sign(dev_ext)

        # Backwards to find i
        i = 0
        for k in range(j - 1, -1, -1):
            dev_k = s[k] - av
            if get_sign(dev_k) != sign_ext:
                i = k + 1
                break

        # Forwards to find f
        f = m - 1
        for k in range(j + 1, m):
            dev_k = s[k] - av
            if get_sign(dev_k) != sign_ext:
                f = k - 1
                break

        # Compute area
        area = 0.0
        for k in range(i, f):
            dt_span = (date_objs[k+1] - date_objs[k]).total_seconds() / 86400.0
            avg_dev = ((s[k] - av) + (s[k+1] - av)) / 2
            area += avg_dev * dt_span

        # Compute z-score for app compatibility
        z_score = (s[j] - mean_s) / std_s

        # Map to sign for app
        sign = 'Positive Spike' if label == 'max' else 'Negative Spike'

        results.append({
            'date': date_objs[j],  # datetime object (app uses res['date'].date())
            'streams': s[j],
            'z_score': z_score,
            'sign': sign,
            # Optional extras from DataGuessing (app ignores them)
            'start_time': date_objs[i].isoformat(),
            'end_time': date_objs[f].isoformat(),
            'area': area
        })

    return results

def get_sign(z_score, threshold=3.0):
    if z_score > threshold:
        return "Positive Spike"
    elif z_score < -threshold:
        return "Negative Spike"
    else:
        return "Normal"
