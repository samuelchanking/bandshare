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


# --- REMOVED: The apply_spike_shifting_logic function has been deleted. ---


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
        if not anomalies.empty:
            fig_forecast.add_trace(go.Scatter(
                x=pd.to_datetime(anomalies['ds']), 
                y=anomalies['y'], 
                mode='markers', 
                marker=dict(color='red', size=8, symbol='x'), 
                name='Anomaly'
            ))

        fig_components = plot_components_plotly(model, forecast)
        
        # --- NEW: Update plot layouts for consistency ---
        fig_forecast.update_layout(
            yaxis_title="Daily Streams",
            hovermode='x unified'
        )
        fig_components.update_layout(
            hovermode='x unified'
        )
        # --------------------------------------------------

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

