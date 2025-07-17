# analysis_tools.py

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import streamlit as st
from typing import List, Dict, Tuple
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import os # <-- ADDED IMPORT
import contextlib # <-- ADDED IMPORT
import io
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.preprocessing import MinMaxScaler
import pywt
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


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


def find_spikes_by_std_dev(series: pd.Series, threshold: float, window_size: int):
    """
    Finds local maxima and minima that are a significant number of standard
    deviations away from a rolling average.

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
            if abs(value - mean_at_idx) > (threshold * std_at_idx):
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
    """
    if not history_data:
        return pd.DataFrame(), pd.to_datetime([])

    parsed_history = []
    for entry in history_data:
        # This handles the two slightly different data structures for stream history
        value = None
        if 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
            value = entry['plots'][0].get('value')
        elif 'value' in entry: # Handles global audience data structure
            value = entry.get('value')

        if entry.get('date') and value is not None:
            parsed_history.append({'date': entry['date'], 'value': value})


    if not parsed_history:
        return pd.DataFrame(), pd.to_datetime([])

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
    df['date'] = pd.to_datetime(df['date'])
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
def detect_comparative_anomalies(target_df: pd.DataFrame, baseline_df: pd.DataFrame, sensitivity: float = 2.0, window: int = 7) -> dict:
    """
    Detects anomalies in a target country's streams by comparing its performance
    ratio against a baseline of other countries. Now returns debug data for plotting.
    """
    if target_df.empty or baseline_df.empty:
        return {"anomalies": [], "debug_data": None, "error": "Input dataframes cannot be empty."}

    target_df['date'] = pd.to_datetime(target_df['date'])
    target_df.set_index('date', inplace=True)
    baseline_df['date'] = pd.to_datetime(baseline_df['date'])
    baseline_df.set_index('date', inplace=True)

    df_merged = target_df.join(baseline_df, lsuffix='_target', rsuffix='_baseline', how='inner')
    
    df_merged.replace(0, np.nan, inplace=True)

    df_merged['ratio'] = df_merged['streams_target'] / df_merged['streams_baseline']
    
    ratio_series = df_merged['ratio'].dropna()
    if ratio_series.empty:
        return {"anomalies": [], "debug_data": None, "error": "No overlapping data to calculate ratio."}

    df_merged['rolling_mean'] = ratio_series.rolling(window=window, center=True, min_periods=1).mean()
    df_merged['rolling_std'] = ratio_series.rolling(window=window, center=True, min_periods=1).std()

    df_merged['upper_bound'] = df_merged['rolling_mean'] + sensitivity * df_merged['rolling_std']

    anomalies_df = df_merged[df_merged['ratio'] > df_merged['upper_bound']].copy()

    anomalies_list = []
    for date, row in anomalies_df.iterrows():
        anomalies_list.append({
            'date': date,
            'target_streams': row['streams_target'],
            'baseline_streams': row['streams_baseline'],
            'ratio': row['ratio'],
            'threshold': row['upper_bound']
        })
        
    debug_df = df_merged[['ratio', 'rolling_mean', 'upper_bound']].reset_index()
    debug_data = {
        'dates': debug_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'ratios': debug_df['ratio'].apply(lambda x: x if pd.notna(x) else -1).tolist(),
        'means': debug_df['rolling_mean'].apply(lambda x: x if pd.notna(x) else -1).tolist(),
        'bounds': debug_df['upper_bound'].apply(lambda x: x if pd.notna(x) else -1).tolist(),
    }

    return {"anomalies": anomalies_list, "debug_data": debug_data, "error": ""}

def find_trend_divergence_periods(data_dict: Dict[str, pd.DataFrame], baseline_countries: List[str], test_country: str, sensitivity: float = 2.0, window: int = 14) -> Dict:
    """
    Finds specific periods where a test country's trend diverges from a baseline trend.
    """
    # 1. Prepare Baseline Data
    baseline_dfs = [df for country, df in data_dict.items() if country in baseline_countries]
    if not baseline_dfs or test_country not in data_dict:
        return {"error": "Baseline countries or test country not found or have insufficient data."}

    # Aggregate baseline streams and ensure unique dates
    baseline_agg_df = pd.concat(baseline_dfs).groupby('date')['streams'].sum().reset_index()
    test_df = data_dict[test_country]

    # 2. Normalize Data
    scaler_base = MinMaxScaler()
    scaler_test = MinMaxScaler()
    
    baseline_agg_df['y_scaled'] = scaler_base.fit_transform(baseline_agg_df[['streams']])
    test_df['y_scaled'] = scaler_test.fit_transform(test_df[['streams']])
    
    baseline_agg_df['date'] = baseline_agg_df['date'].dt.tz_localize(None)
    test_df['date'] = test_df['date'].dt.tz_localize(None)

    # 3. Train Prophet Models to get trends
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        # Baseline model
        base_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        base_model.fit(baseline_agg_df.rename(columns={'date': 'ds', 'y_scaled': 'y'}))
        base_forecast = base_model.predict(base_model.history)
        
        # Test model
        test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        test_model.fit(test_df.rename(columns={'date': 'ds', 'y_scaled': 'y'}))
        test_forecast = test_model.predict(test_model.history)

    # 4. Analyze Trend Difference
    trends = pd.merge(
        base_forecast[['ds', 'trend']], 
        test_forecast[['ds', 'trend']], 
        on='ds', 
        suffixes=('_base', '_test')
    )
    trends['difference'] = trends['trend_test'] - trends['trend_base']

    # 5. Find Anomalous Periods in the difference
    trends['rolling_mean'] = trends['difference'].rolling(window=window, center=True, min_periods=1).mean()
    trends['rolling_std'] = trends['difference'].rolling(window=window, center=True, min_periods=1).std()
    
    trends['upper_bound'] = trends['rolling_mean'] + sensitivity * trends['rolling_std']
    trends['lower_bound'] = trends['rolling_mean'] - sensitivity * trends['rolling_std']
    
    trends['anomaly'] = (trends['difference'] > trends['upper_bound']) | (trends['difference'] < trends['lower_bound'])

    # Group consecutive anomalies into periods
    anomalous_periods = []
    in_anomaly = False
    start_date = None
    for i, row in trends.iterrows():
        if row['anomaly'] and not in_anomaly:
            in_anomaly = True
            start_date = row['ds']
        elif not row['anomaly'] and in_anomaly:
            in_anomaly = False
            anomalous_periods.append({'start': start_date, 'end': trends.loc[i-1, 'ds']})
    if in_anomaly: # Close the last period if it extends to the end
        anomalous_periods.append({'start': start_date, 'end': trends.iloc[-1]['ds']})

    return {
        "baseline_trend": base_forecast[['ds', 'trend']].to_dict('records'),
        "test_trend": test_forecast[['ds', 'trend']].to_dict('records'),
        "anomalous_periods": anomalous_periods,
        "error": ""
    }
    """
    Analyzes trends of multiple countries against a baseline using Prophet.
    Returns a dictionary with trend data for plotting and flags divergent countries.
    """
    if baseline_country not in data_dict:
        return {"error": "Baseline country not found in the provided data."}

    # Normalize all dataframes to compare trend shapes
    scalers = {}
    normalized_dfs = {}
    for country, df in data_dict.items():
        if df.empty or len(df) < 2: continue
        
        # MODIFIED: Remove timezone information to make the 'ds' column timezone-naive
        df['date'] = df['date'].dt.tz_localize(None)
        
        scaler = MinMaxScaler()
        # Reshape is needed for single-column scaling
        df['y_scaled'] = scaler.fit_transform(df[['streams']])
        normalized_dfs[country] = df.rename(columns={'date': 'ds', 'y_scaled': 'y'})
        scalers[country] = scaler
    
    if not normalized_dfs or baseline_country not in normalized_dfs:
        return {"error": "Not enough data to perform analysis or baseline country has insufficient data."}

    # Train baseline model
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        baseline_df = normalized_dfs[baseline_country]
        baseline_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        baseline_model.fit(baseline_df)
    baseline_forecast = baseline_model.predict(baseline_df[['ds']])
    baseline_trend = baseline_forecast[['ds', 'trend']].copy()

    # Analyze each test country against the baseline
    results = []
    trend_errors = []
    for country, df in normalized_dfs.items():
        if country == baseline_country:
            continue
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            test_model.fit(df)
        forecast = test_model.predict(df[['ds']])
        test_trend = forecast[['ds', 'trend']]
        
        # Merge trends and calculate error
        merged_trends = pd.merge(baseline_trend, test_trend, on='ds', suffixes=('_base', '_test'))
        if not merged_trends.empty:
            rmse = np.sqrt(mean_squared_error(merged_trends['trend_base'], merged_trends['trend_test']))
            trend_errors.append(rmse)
            results.append({
                "country": country,
                "trend_df": test_trend,
                "rmse": rmse
            })
    
    if not trend_errors:
        return {"error": "Could not compute trends for test countries."}

    # Flag divergent countries
    avg_rmse = np.mean(trend_errors)
    final_results = []
    for res in results:
        is_divergent = res['rmse'] > (avg_rmse * divergence_threshold)
        final_results.append({
            "country": res['country'],
            "trend": res['trend_df'].to_dict('records'),
            "is_divergent": is_divergent,
            "score": res['rmse'] / avg_rmse if avg_rmse > 0 else 0
        })

    return {
        "baseline_country": baseline_country,
        "baseline_trend": baseline_trend.to_dict('records'),
        "test_countries": final_results,
        "error": ""
    }


    """
    Analyzes trends of multiple countries against a baseline using Prophet.
    Returns a dictionary with trend data for plotting and flags divergent countries.
    """
    if baseline_country not in data_dict:
        return {"error": "Baseline country not found in the provided data."}

    # Normalize all dataframes to compare trend shapes
    scalers = {}
    normalized_dfs = {}
    for country, df in data_dict.items():
        if df.empty or len(df) < 2: continue
        scaler = MinMaxScaler()
        # Reshape is needed for single-column scaling
        df['y_scaled'] = scaler.fit_transform(df[['streams']])
        normalized_dfs[country] = df.rename(columns={'date': 'ds', 'y_scaled': 'y'})
        scalers[country] = scaler
    
    if not normalized_dfs:
        return {"error": "Not enough data to perform analysis."}

    # Train baseline model
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        baseline_df = normalized_dfs[baseline_country]
        baseline_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        baseline_model.fit(baseline_df)
    baseline_forecast = baseline_model.predict(baseline_df[['ds']])
    baseline_trend = baseline_forecast[['ds', 'trend']].copy()

    # Analyze each test country against the baseline
    results = []
    trend_errors = []
    for country, df in normalized_dfs.items():
        if country == baseline_country:
            continue
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            test_model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            test_model.fit(df)
        forecast = test_model.predict(df[['ds']])
        test_trend = forecast[['ds', 'trend']]
        
        # Merge trends and calculate error
        merged_trends = pd.merge(baseline_trend, test_trend, on='ds', suffixes=('_base', '_test'))
        if not merged_trends.empty:
            rmse = np.sqrt(mean_squared_error(merged_trends['trend_base'], merged_trends['trend_test']))
            trend_errors.append(rmse)
            results.append({
                "country": country,
                "trend_df": test_trend,
                "rmse": rmse
            })
    
    if not trend_errors:
        return {"error": "Could not compute trends for test countries."}

    # Flag divergent countries
    avg_rmse = np.mean(trend_errors)
    final_results = []
    for res in results:
        is_divergent = res['rmse'] > (avg_rmse * divergence_threshold)
        final_results.append({
            "country": res['country'],
            "trend": res['trend_df'].to_dict('records'),
            "is_divergent": is_divergent,
            "score": res['rmse'] / avg_rmse if avg_rmse > 0 else 0
        })

    return {
        "baseline_country": baseline_country,
        "baseline_trend": baseline_trend.to_dict('records'),
        "test_countries": final_results,
        "error": ""
    }
    
def run_automated_divergence_analysis(data_dict: Dict[str, pd.DataFrame], divergence_threshold: float = 1.5, outlier_threshold_multiplier: float = 1.5) -> Dict:
    """
    Performs a cross-validation style analysis to automatically find divergent countries and their anomalous periods.
    Now returns debug data for plotting the divergence scores.
    """
    if len(data_dict) < 2:
        return {"error": "At least two countries are required for this analysis."}

    # 1. Normalize all data and train a Prophet model for each country
    models = {}
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        for country, df in data_dict.items():
            if df.empty or len(df) < 14: continue
            df['date'] = df['date'].dt.tz_localize(None)
            scaler = MinMaxScaler()
            df['y_scaled'] = scaler.fit_transform(df[['streams']])
            
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df.rename(columns={'date': 'ds', 'y_scaled': 'y'}))
            forecast = model.predict(model.history)
            models[country] = forecast[['ds', 'trend']]

    if len(models) < 2:
        return {"error": "Could not train enough models to perform cross-comparison."}

    # 2. Cross-comparison to get divergence scores
    divergence_scores = {country: 0 for country in models.keys()}
    for baseline_country, baseline_trend in models.items():
        rmses = []
        for test_country, test_trend in models.items():
            if baseline_country == test_country: continue
            merged = pd.merge(baseline_trend, test_trend, on='ds')
            if not merged.empty:
                rmses.append(np.sqrt(mean_squared_error(merged['trend_x'], merged['trend_y'])))
        
        if not rmses: continue
        avg_rmse = np.mean(rmses)
        
        for test_country, test_trend in models.items():
             if baseline_country == test_country: continue
             merged = pd.merge(baseline_trend, test_trend, on='ds')
             if not merged.empty:
                rmse = np.sqrt(mean_squared_error(merged['trend_x'], merged['trend_y']))
                if rmse > avg_rmse * divergence_threshold:
                    divergence_scores[baseline_country] += 1
    
    # 3. Identify genuine and divergent countries
    avg_score = np.mean(list(divergence_scores.values()))
    score_threshold = avg_score * outlier_threshold_multiplier
    
    genuine_countries = [country for country, score in divergence_scores.items() if score <= score_threshold]
    divergent_countries = [country for country, score in divergence_scores.items() if score > score_threshold]

    if not genuine_countries:
        return {"error": "Could not identify a stable baseline of 'genuine' countries. Trends may be too chaotic."}

    # 4. Run final analysis for each divergent country against the genuine baseline
    final_analysis_results = []
    for d_country in divergent_countries:
        result = find_trend_divergence_periods(data_dict, genuine_countries, d_country)
        if not result.get("error"):
            result['country'] = d_country
            final_analysis_results.append(result)

    # 5. Prepare debug plot data
    debug_plot_data = {
        "divergence_scores": [{"country": c, "score": s} for c, s in divergence_scores.items()],
        "score_threshold": score_threshold
    }

    return {
        "genuine_countries": genuine_countries,
        "divergent_countries": divergent_countries,
        "analysis_results": final_analysis_results,
        "debug_plot_data": debug_plot_data, # ADDED
        "error": ""
    }

    """
    Performs a cross-validation style analysis to automatically find divergent countries and their anomalous periods.
    """
    if len(data_dict) < 2:
        return {"error": "At least two countries are required for this analysis."}

    # 1. Normalize all data and train a Prophet model for each country
    models = {}
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        for country, df in data_dict.items():
            if df.empty or len(df) < 14: continue
            df['date'] = df['date'].dt.tz_localize(None)
            scaler = MinMaxScaler()
            df['y_scaled'] = scaler.fit_transform(df[['streams']])
            
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df.rename(columns={'date': 'ds', 'y_scaled': 'y'}))
            forecast = model.predict(model.history)
            models[country] = forecast[['ds', 'trend']]

    if len(models) < 2:
        return {"error": "Could not train enough models to perform cross-comparison."}

    # 2. Cross-comparison to get divergence scores
    divergence_scores = {country: 0 for country in models.keys()}
    for baseline_country, baseline_trend in models.items():
        rmses = []
        for test_country, test_trend in models.items():
            if baseline_country == test_country: continue
            merged = pd.merge(baseline_trend, test_trend, on='ds')
            if not merged.empty:
                rmses.append(np.sqrt(mean_squared_error(merged['trend_x'], merged['trend_y'])))
        
        if not rmses: continue
        avg_rmse = np.mean(rmses)
        
        for test_country, test_trend in models.items():
             if baseline_country == test_country: continue
             merged = pd.merge(baseline_trend, test_trend, on='ds')
             if not merged.empty:
                rmse = np.sqrt(mean_squared_error(merged['trend_x'], merged['trend_y']))
                if rmse > avg_rmse * divergence_threshold:
                    divergence_scores[baseline_country] += 1
    
    # 3. Identify genuine and divergent countries
    avg_score = np.mean(list(divergence_scores.values()))
    score_threshold = avg_score * outlier_threshold_multiplier
    
    genuine_countries = [country for country, score in divergence_scores.items() if score <= score_threshold]
    divergent_countries = [country for country, score in divergence_scores.items() if score > score_threshold]

    if not genuine_countries:
        return {"error": "Could not identify a stable baseline of 'genuine' countries. Trends may be too chaotic."}

    # 4. Run final analysis for each divergent country against the genuine baseline
    final_analysis_results = []
    for d_country in divergent_countries:
        result = find_trend_divergence_periods(data_dict, genuine_countries, d_country)
        if not result.get("error"):
            result['country'] = d_country
            final_analysis_results.append(result)

    return {
        "genuine_countries": genuine_countries,
        "divergent_countries": divergent_countries,
        "analysis_results": final_analysis_results,
        "error": ""
    }
