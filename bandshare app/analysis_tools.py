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
        if 'date' in entry and 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
            value = entry['plots'][0].get('value')
            if value is not None:
                parsed_history.append({'date': entry['date'], 'value': value})

    if not parsed_history:
        return pd.DataFrame(), pd.to_datetime([])

    df = pd.DataFrame(parsed_history)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    df_processed = df.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)

    decreasing_mask = df_processed['value'].diff() < 0
    decreasing_dates = df_processed.loc[decreasing_mask, 'date']
    
    df_processed = df_processed[~decreasing_mask].reset_index(drop=True)
    
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

        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        daily_cumulative = df.reindex(full_date_range).interpolate(method='time')

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

    interpolator = interp1d(days, streams, kind='linear', fill_value='extrapolate')

    max_days = max(days)
    if max_days < discretization_step:
        return format_error(f"Data range ({int(max_days)} days) is smaller than the discretization step ({discretization_step} days).")

    grid_days = np.arange(0, max_days + 1, discretization_step)
    s = interpolator(grid_days)

    s_series = pd.Series(s)
    Av = s_series.rolling(window=smoothing_window_size, center=True, min_periods=1).mean().to_numpy()

    s_diff = np.diff(s) / discretization_step
    Av_diff = np.diff(Av) / discretization_step

    abs_diff_dev = np.abs(s_diff - Av_diff)

    if len(abs_diff_dev) > 0:
        smoothed_std = pd.Series(abs_diff_dev).rolling(window=smoothing_window_size, center=True, min_periods=1).mean().to_numpy()
    else:
        smoothed_std = np.array([])

    anomalies = []
    for i in range(len(abs_diff_dev)):
        threshold = sensitivity * smoothed_std[i]
        if abs_diff_dev[i] > threshold and threshold > 0:

            spike_size_per_day = s_diff[i] - Av_diff[i]
            expected_rate_per_day = Av_diff[i]
            total_streams_on_day = s_diff[i]

            if expected_rate_per_day > 1:
                relative_significance = spike_size_per_day / expected_rate_per_day
            else:
                relative_significance = float('inf')

            anomaly_date = start_date + timedelta(days=int(grid_days[i+1]))
            anomalies.append({
                'date': anomaly_date.strftime('%Y-%m-%d'),
                'spike_size_streams_per_day': float(spike_size_per_day),
                'relative_significance': float(relative_significance),
                'total_streams_on_day': float(total_streams_on_day)
            })

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


# --- Other functions (detect_additional_contribution, etc.) remain unchanged ---
# (Code for other functions is omitted for brevity but should be kept in your file)
def detect_additional_contribution(data, event_date, event_duration_days=30, is_cumulative=False, smoothing_window_size=7):
    """
    Detects the effect of an event with a known duration using Prophet's holiday modeling.
    It now automatically finds the optimal holiday prior scale.
    """
    try:
        input_data = json.loads(data) if isinstance(data, str) else data
        df = pd.DataFrame({
            'ds': pd.to_datetime(input_data['dates']),
            'y': input_data['streams']
        })

        if is_cumulative:
            df['y'] = df['y'].diff().fillna(df['y'].iloc[0])
            df = df.iloc[1:].copy()
        
        if smoothing_window_size > 1:
            df['y'] = df['y'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

        if df['ds'].duplicated().any():
            return json.dumps({'error': 'Duplicate dates detected in input data'})
        if len(df) < 14:
            return json.dumps({'error': 'Insufficient data points (minimum 14 days required)'})

        event_date_dt = datetime.strptime(event_date, '%Y-%m-%d')
        event_holiday = pd.DataFrame({
            'holiday': 'long_event',
            'ds': [event_date_dt + timedelta(days=i) for i in range(event_duration_days)],
            'lower_window': 0,
            'upper_window': 0,
        })

        best_scale = 0.01
        best_mse = float('inf')
        
        scales_to_test = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]

        for scale in scales_to_test:
            model_temp = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                holidays=event_holiday,
                changepoint_prior_scale=0.05,
                holidays_prior_scale=scale
            )
            model_temp.fit(df)
            forecast_temp = model_temp.predict(df)
            mse = mean_squared_error(df['y'], forecast_temp['yhat'])
            
            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        final_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            holidays=event_holiday,
            changepoint_prior_scale=0.05,
            holidays_prior_scale=best_scale
        )
        final_model.fit(df)
        
        future = final_model.make_future_dataframe(periods=event_duration_days)
        forecast = final_model.predict(future)

        holiday_effect = forecast[forecast['ds'].isin(event_holiday['ds'])][['ds', 'long_event']]
        if holiday_effect.empty:
            return json.dumps({'additional_contribution': {}, 'error': 'No holiday effect found for the specified event date'})

        avg_additional_streams = holiday_effect['long_event'].mean()
        if pd.isna(avg_additional_streams):
            avg_additional_streams = 0.0

        result = {
            'event_start_date': event_date_dt.strftime('%Y-%m-%d'),
            'event_duration_days': event_duration_days,
            'average_additional_streams_per_day': float(avg_additional_streams),
            'optimal_prior_scale': float(best_scale)
        }

        return json.dumps({'additional_contribution': result, 'error': ''})

    except Exception as e:
        return json.dumps({'error': f'Error processing data: {str(e)}'})

def detect_prophet_anomalies(data: Dict, is_cumulative: bool = False, interval_width: float = 0.95, smoothing_window_size: int = 1) -> str:
    """
    Detects anomalous spikes in time-series data using Prophet's forecast uncertainty intervals.
    """
    try:
        df = pd.DataFrame({
            'ds': pd.to_datetime(data['dates']),
            'y': data['streams']
        })

        if is_cumulative:
            df['y'] = df['y'].diff().fillna(df['y'].iloc[0])
            df = df.iloc[1:].copy()
        
        if smoothing_window_size > 1:
            df['y'] = df['y'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()
        
        if df.empty or len(df) < 2:
             return json.dumps({'anomalies': [], 'forecast_df': None, 'error': "Insufficient data for anomaly detection."})

        model = Prophet(interval_width=interval_width, daily_seasonality=True)
        model.fit(df)
        
        forecast = model.predict(df)
        
        results_df = pd.concat([df.set_index('ds')['y'], forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]], axis=1).reset_index()
        
        results_df['anomaly'] = (results_df['y'] > results_df['yhat_upper']) | (results_df['y'] < results_df['yhat_lower'])
        anomalies = results_df[results_df['anomaly']].copy()

        anomalies_list = []
        if not anomalies.empty:
            anomalies['difference'] = anomalies['y'] - anomalies['yhat_upper']
            anomalies.loc[anomalies['y'] < anomalies['yhat_lower'], 'difference'] = anomalies['y'] - anomalies['yhat_lower']
            anomalies['ds'] = anomalies['ds'].dt.strftime('%Y-%m-%d')
            anomalies_list = anomalies[['ds', 'y', 'yhat', 'difference']].to_dict('records')

        results_df['ds'] = results_df['ds'].dt.strftime('%Y-%m-%d')
        forecast_plot_data = results_df[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        return json.dumps({
            'anomalies': anomalies_list,
            'forecast_df': forecast_plot_data,
            'error': ''
        })

    except Exception as e:
        return json.dumps({'anomalies': [], 'forecast_df': None, 'error': f'An error occurred: {str(e)}'})

def detect_event_effect(data: Dict, start_date: str, end_date: str, is_cumulative: bool = False, baseline_period_days: int = 30, smoothing_window_size: int = 7) -> str:
    """
    Detects the effect of an event with a known start and end date by comparing the average
    of a smoothed time-series during the event to a baseline period.
    """
    try:
        df = pd.DataFrame({
            'date': pd.to_datetime(data['dates']),
            'value': data['streams']
        })
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        if is_cumulative:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(full_range).interpolate(method='time')
            df['value'] = df['value'].diff().fillna(0).clip(lower=0)

        if smoothing_window_size > 1:
            df['value'] = df['value'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        baseline_start_dt = start_date_dt - timedelta(days=baseline_period_days)

        baseline_df = df[(df.index >= baseline_start_dt) & (df.index < start_date_dt)]
        event_df = df[(df.index >= start_date_dt) & (df.index <= end_date_dt)]

        if baseline_df.empty:
            return json.dumps({'error': 'Not enough data to establish a baseline before the event start date.'})
        if event_df.empty:
            return json.dumps({'error': 'No data available within the specified event period.'})

        baseline_avg = baseline_df['value'].mean()
        event_avg = event_df['value'].mean()

        absolute_increase = event_avg - baseline_avg
        relative_increase = (absolute_increase / baseline_avg) if baseline_avg > 0 else float('inf')

        result = {
            'baseline_period_avg': float(baseline_avg),
            'event_period_avg': float(event_avg),
            'absolute_increase_per_day': float(absolute_increase),
            'relative_increase': float(relative_increase)
        }
        
        return json.dumps({'event_effect': result, 'error': ''})

    except Exception as e:
        return json.dumps({'error': f'An error occurred: {str(e)}'})

def detect_changepoint_effect(data: Dict, changepoint_date: str, is_cumulative: bool = False, window_days: int = 14, smoothing_window_size: int = 7) -> str:
    """
    Detects an effect around a single point in time by comparing a window of time
    before and after the changepoint.
    """
    try:
        df = pd.DataFrame({
            'date': pd.to_datetime(data['dates']),
            'value': data['streams']
        })
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        if is_cumulative:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(full_range).interpolate(method='time')
            df['value'] = df['value'].diff().fillna(0).clip(lower=0)

        if smoothing_window_size > 1:
            df['value'] = df['value'].rolling(window=smoothing_window_size, center=True, min_periods=1).mean()

        changepoint_dt = pd.to_datetime(changepoint_date)
        
        before_start_dt = changepoint_dt - timedelta(days=window_days)
        after_end_dt = changepoint_dt + timedelta(days=window_days)

        before_df = df[(df.index >= before_start_dt) & (df.index < changepoint_dt)]
        after_df = df[(df.index >= changepoint_dt) & (df.index <= after_end_dt)]

        if before_df.empty:
            return json.dumps({'error': f'Not enough data in the {window_days}-day window before the changepoint.'})
        if after_df.empty:
            return json.dumps({'error': f'No data available in the {window_days}-day window after the changepoint.'})

        before_avg = before_df['value'].mean()
        after_avg = after_df['value'].mean()

        absolute_increase = after_avg - before_avg
        relative_increase = (absolute_increase / before_avg) if before_avg > 0 else float('inf')

        result = {
            'before_period_avg': float(before_avg),
            'after_period_avg': float(after_avg),
            'absolute_increase_per_day': float(absolute_increase),
            'relative_increase': float(relative_increase)
        }
        
        return json.dumps({'changepoint_effect': result, 'error': ''})

    except Exception as e:
        return json.dumps({'error': f'An error occurred: {str(e)}'})
