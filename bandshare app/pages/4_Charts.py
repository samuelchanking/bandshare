# pages/4_Charts.py

import pandas as pd
import streamlit as st
import config
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import (
    get_audience_data, get_popularity_data,
    get_streaming_audience_from_db,
    get_local_streaming_history_from_db,
    get_song_audience_data,
    get_artist_events,
    get_all_songs_for_artist_from_db
)
from streamlit_ui import (
    display_top_countries_trend_chart,
    display_streaming_audience_chart,
    display_country_statistics_charts,
    display_single_timeseries_stats_chart,
    display_single_wavelet_analysis,
    display_artist_events,
)
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import plotly.io as pio
from analysis_tools import (
    detect_prophet_anomalies,
    detect_comparative_anomalies,
    find_trend_divergence_periods,
    run_automated_divergence_analysis,
    detect_wavelet_spikes,
    remove_sudden_zeros
)
import pywt
from scipy.signal import find_peaks

# --- Initialization ---
st.set_page_config(page_title="Artist Charts", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Helper Functions ---
def _get_optimal_y_range(dataframe, columns):
    """Calculates an optimal Y-axis range with 10% padding."""
    min_val, max_val = float('inf'), float('-inf')

    for col in columns:
        if col in dataframe.columns and not dataframe[col].dropna().empty:
            min_val = min(min_val, dataframe[col].min())
            max_val = max(max_val, dataframe[col].max())

    if pd.isna(min_val) or pd.isna(max_val) or min_val == float('inf'):
        return None

    if min_val == max_val:
        return [min_val - 1, max_val + 1]

    padding = (max_val - min_val) * 0.1
    if padding == 0: padding = 1

    return [min_val - padding, max_val + padding]

def update_song_audience_data(artist_uuid, start_date_filter, end_date_filter):
    songs = get_all_songs_for_artist_from_db(db_manager, artist_uuid)
    if not songs:
        st.warning("No songs found for this artist.")
        return
    with st.spinner(f"Updating audience data for {len(songs)} songs..."):
        full_song_data = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_song = {}
            for song in songs:
                song_uuid = song['uuid']
                query_filter = {'song_uuid': song_uuid, 'platform': 'spotify'}
                min_db_date, max_db_date = db_manager.get_timeseries_data_range('song_audience', query_filter)
                if min_db_date and start_date_filter < min_db_date.date():
                    future_to_song[executor.submit(api_client.get_song_streaming_audience, song_uuid, 'spotify', start_date_filter, min_db_date.date() - timedelta(days=1))] = song_uuid
                forward_start = max_db_date.date() + timedelta(days=1) if max_db_date else start_date_filter
                if forward_start <= end_date_filter:
                    future_to_song[executor.submit(api_client.get_song_streaming_audience, song_uuid, 'spotify', forward_start, end_date_filter)] = song_uuid
            for future in as_completed(future_to_song):
                song_uuid = future_to_song[future]
                try:
                    data = future.result()
                    if 'error' not in data and 'items' in data and data['items']:
                        if song_uuid not in full_song_data:
                            full_song_data[song_uuid] = {'items': [], 'platform': data.get('platform')}
                        full_song_data[song_uuid]['items'].extend(data['items'])
                except Exception as exc:
                    st.warning(f"Data fetch for song {song_uuid} failed: {exc}")
        if full_song_data:
            for song_uuid, data in full_song_data.items():
                db_manager.store_song_audience_data(song_uuid, data)
            st.success("Song audience data updated.")
            get_song_audience_data.clear()
            st.rerun()
        else:
            st.info("No new song data to update.")

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

def fetch_and_store_events(artist_uuid):
    """Fetches the complete list of events from the API and stores them in the database."""
    try:
        with st.spinner("Fetching complete event list from the API... This may take a moment for artists with many events."):
            # Fetch all events using the paginated client function
            raw_events = api_client.get_artist_events(artist_uuid)

            if 'error' in raw_events or not raw_events:
                st.warning("Could not fetch events or no events were found.")
                return

            # Store the raw, unenriched events. This will overwrite existing events.
            db_manager.store_artist_events(artist_uuid, raw_events)
            
            # Clear the cache to ensure the UI reloads the new data
            get_artist_events.clear()
            st.success(f"Successfully fetched and stored {len(raw_events)} events.")
            st.rerun()

    except Exception as e:
        st.error(f"An error occurred while fetching events: {e}")


def fetch_and_store_venue_festival_metadata(artist_uuid):
    """
    Scans all stored events, finds unique venue/festival UUIDs,
    and fetches and stores their metadata in the database.
    """
    try:
        with st.spinner("Scanning events for venue and festival information..."):
            # Get all events from the database
            events_from_db = get_artist_events(db_manager, artist_uuid)
            if not events_from_db:
                st.info("No events found in the database. Please fetch the event list first.")
                return

            # Create sets of unique UUIDs to avoid redundant API calls
            venue_uuids = set()
            festival_uuids = set()
            for event in events_from_db:
                if event.get('type') == 'festival' and event.get('festival', {}).get('uuid'):
                    festival_uuids.add(event['festival']['uuid'])
                elif event.get('venue', {}).get('uuid'):
                    venue_uuids.add(event['venue']['uuid'])

            if not venue_uuids and not festival_uuids:
                st.info("No venues or festivals were found in the stored event data.")
                return

            st.info(f"Found {len(venue_uuids)} unique venues and {len(festival_uuids)} unique festivals. Fetching details now...")
            progress_bar = st.progress(0, text="Fetching metadata...")
            stored_count = 0

            with ThreadPoolExecutor(max_workers=20) as executor:
                # Create a dictionary to map each future back to its type ('venue' or 'festival')
                future_to_type = {}
                for uuid in venue_uuids:
                    future = executor.submit(api_client.get_venue_metadata, uuid)
                    future_to_type[future] = 'venue'
                for uuid in festival_uuids:
                    future = executor.submit(api_client.get_festival_metadata, uuid)
                    future_to_type[future] = 'festival'

                total_futures = len(future_to_type)
                for i, future in enumerate(as_completed(future_to_type)):
                    metadata_type = future_to_type[future]
                    try:
                        metadata = future.result()
                        # If the API call was successful, store the entire metadata document
                        if metadata and 'error' not in metadata:
                            if metadata_type == 'venue':
                                db_manager.store_venue_metadata(metadata)
                            else: # festival
                                db_manager.store_festival_metadata(metadata)
                            stored_count += 1
                    except Exception as e:
                        st.warning(f"A metadata fetch failed: {e}")

                    progress_bar.progress((i + 1) / total_futures, text=f"Fetching metadata... ({i+1}/{total_futures})")

        if stored_count > 0:
            st.success(f"Successfully fetched and stored metadata for {stored_count} venues/festivals.")
        else:
            st.info("Completed. No new metadata was found or stored.")

    except Exception as e:
        st.error(f"An unexpected error occurred during the metadata fetch process: {e}")

def update_timeseries_data(artist_uuid, start_date_filter, end_date_filter):
    """Fetches and updates time-series data for the artist."""
    try:
        with st.spinner("Checking and updating artist time-series data..."):
            # Time-series data fetching setup
            ts_tasks = {
                'audience': (lambda start, end: api_client.get_artist_audience(artist_uuid, 'spotify', start, end), 'audience'),
                'popularity': (lambda start, end: api_client.get_artist_popularity(artist_uuid, 'spotify', start, end), 'popularity'),
                'streaming_audience': (lambda start, end: api_client.get_artist_streaming_audience(artist_uuid, 'spotify', start, end), 'streaming_audience'),
                'local_streaming_audience': (lambda start, end: api_client.get_local_streaming_audience(artist_uuid, 'spotify', start, end), 'local_streaming_history')
            }
            full_ts_data = {}
            data_was_updated = False

            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_name = {}
                for name, (func, coll_name) in ts_tasks.items():
                    query_filter = {'artist_uuid': artist_uuid, ('source' if name == 'popularity' else 'platform'): 'spotify'}
                    min_db_date, max_db_date = db_manager.get_timeseries_data_range(coll_name, query_filter)
                    if min_db_date and start_date_filter < min_db_date.date():
                        future_to_name[executor.submit(func, start_date_filter, min_db_date.date() - timedelta(days=1))] = name
                    forward_start_date = max_db_date.date() + timedelta(days=1) if max_db_date else start_date_filter
                    if forward_start_date <= end_date_filter:
                         future_to_name[executor.submit(func, forward_start_date, end_date_filter)] = name

                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        data = future.result()
                        if 'error' not in data and 'items' in data and data['items']:
                            platform_or_source = 'source' if name == 'popularity' else 'platform'
                            if name not in full_ts_data:
                                full_ts_data[name] = {'items': [], platform_or_source: data.get(platform_or_source)}
                            full_ts_data[name]['items'].extend(data['items'])
                    except Exception as exc:
                        st.warning(f'{name} data fetching generated an exception: {exc}')

            if full_ts_data:
                db_manager.store_timeseries_data(artist_uuid, full_ts_data)
                data_was_updated = True

            if data_was_updated:
                st.cache_data.clear()
                st.success("Time-series data updated successfully.")
                st.rerun()
            else:
                st.info("No new time-series data to update for the selected range.")
    except Exception as e:
        st.error(f"An unexpected error occurred during the update process: {e}")

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their charts.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Time-Series Charts for {artist_name}")

# --- Filters and Update Button ---
st.markdown("### Chart Data")
start_date_filter = st.date_input("Chart Start Date", date.today() - timedelta(days=1095))
end_date_filter = st.date_input("Chart End Date", date.today())

if st.button("Get/Update Artist Chart Data", use_container_width=True, type="primary"):
    update_timeseries_data(artist_uuid, start_date_filter, end_date_filter)

# --- Fetch Data ---
audience_data = get_audience_data(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
popularity_data = get_popularity_data(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
streaming_data = get_streaming_audience_from_db(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
local_streaming_data = get_local_streaming_history_from_db(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)

# --- Interactive Multi-Select Chart ---
st.markdown("---")
st.subheader("Interactive Artist Metrics")
st.caption("Select up to two metrics from the dropdown to compare them.")

def parse_and_prepare_df(raw_data, primary_value_key, new_column_name):
    """Parses various time-series data structures into a clean DataFrame."""
    if not raw_data: return None
    data_source = None
    if isinstance(raw_data, dict):
        data_source = raw_data.get('history', raw_data)
        if isinstance(data_source, dict):
            data_source = data_source.get('items', data_source)
    elif isinstance(raw_data, list):
        data_source = raw_data
    if not isinstance(data_source, list): return None
    parsed_data = []
    for entry in data_source:
        date_val = entry.get('date')
        value = entry.get(primary_value_key, entry.get('value'))
        if date_val and value is not None:
            parsed_data.append({'date': pd.to_datetime(date_val), new_column_name: value})
    if not parsed_data: return None
    df = pd.DataFrame(parsed_data).set_index('date')
    # NEW: Apply remove_sudden_zeros to clean the series
    df[new_column_name] = remove_sudden_zeros(df[new_column_name], window=3)
    return df

# Process and merge data
all_dfs = []
df_aud = parse_and_prepare_df(audience_data, 'followerCount', 'Audience')
if df_aud is not None: all_dfs.append(df_aud)

df_pop = parse_and_prepare_df(popularity_data, 'value', 'Popularity')
if df_pop is not None: all_dfs.append(df_pop)

df_stream = parse_and_prepare_df(streaming_data, 'value', 'Streaming Audience')
if df_stream is not None: all_dfs.append(df_stream)

all_metric_options = ['Audience', 'Popularity', 'Streaming Audience']
chart_options = st.multiselect(
    "Select metrics to display:",
    all_metric_options,
    default=all_metric_options
)

if all_dfs:
    merged_df = pd.concat(all_dfs, axis=1)
    for col in merged_df.columns:
        merged_df[col] = remove_sudden_zeros(merged_df[col], window=3)    
    if chart_options:
        available_options = [opt for opt in chart_options if opt in merged_df.columns and not merged_df[opt].dropna().empty]
        if available_options:
            fig = go.Figure()
            colors = { 'Audience': '#1f77b4', 'Popularity': '#ff7f0e', 'Streaming Audience': '#2ca02c' }
            is_multiple = len(available_options) > 1
            normalized_df = merged_df.copy()
            if is_multiple:
                for metric in available_options:
                    min_val = merged_df[metric].min()
                    max_val = merged_df[metric].max()
                    if max_val - min_val != 0:
                        normalized_df[metric] = (merged_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized_df[metric] = 0.5
            for metric in available_options:
                if is_multiple:
                    y_data = normalized_df[metric]
                    customdata = merged_df[metric].values
                    hovertemplate = f'<b>{metric}</b>: %{{customdata:,.0f}}<extra></extra>'
                else:
                    y_data = merged_df[metric]
                    customdata = None
                    hovertemplate = f'<b>{metric}</b>: %{{y:,.0f}}<extra></extra>'
                fig.add_trace(go.Scatter(
                    x=merged_df.index, 
                    y=y_data, 
                    mode='lines', 
                    name=metric, 
                    connectgaps=True, 
                    line=dict(color=colors.get(metric)),
                    customdata=customdata,
                    hovertemplate=hovertemplate
                ))
            fig.update_layout(title_text="Artist Performance Metrics", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            if is_multiple:
                fig.update_yaxes(visible=False, range=[-0.05, 1.05])
            else:
                y_range = _get_optimal_y_range(merged_df, available_options)
                fig.update_yaxes(title_text=available_options[0], range=y_range, visible=True)
            st.plotly_chart(fig, use_container_width=True)
            if is_multiple:
                st.caption("Trends are normalized to 0-1 for comparison. Hover to see original values.")
        else:
            st.info("Data for the selected metric(s) is not available for this period.")
    else:
        st.info("Select one or more metrics from the dropdown above to display the chart.")
else:
    st.info("No time-series data found for this artist.")

if len(available_options) == 2:
    st.markdown("---")
    st.subheader("Calculated Ratio Chart")
    metric1, metric2 = available_options[0], available_options[1]
    st.caption(f"This chart displays the result of **{metric1} / {metric2}**.")
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_series = merged_df[metric1].astype(float) / merged_df[metric2].astype(float)
    ratio_series.replace([np.inf, -np.inf], np.nan, inplace=True)
    if not ratio_series.dropna().empty:
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=ratio_series.index, y=ratio_series, mode='lines', name='Ratio', connectgaps=True))
        fig_ratio.update_layout(title=f"Ratio of {metric1} to {metric2}", yaxis_title="Ratio Value", hovermode="x unified")
        st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("Could not calculate a valid ratio for the selected metrics over this period.")
        
        
# --- Geographical Streaming Performance ---
st.markdown("---")
st.subheader("Geographical Streaming Performance")
if local_streaming_data:
    processed_data = []
    for daily_entry in local_streaming_data:
        date_val = daily_entry.get('date')
        for country_plot in daily_entry.get('countryPlots', []):
            processed_data.append({'date': date_val, 'country': country_plot.get('countryName'), 'streams': country_plot.get('value')})
    
    if processed_data:
        df = pd.DataFrame(processed_data)
        df['date'] = pd.to_datetime(df['date'])
        country_stats = df.groupby('country')['streams'].agg(['sum', 'mean']).reset_index().rename(columns={'sum': 'Total Streams', 'mean': 'Average Streams'})
        ranking_metric = st.selectbox("Rank Top 10 Countries by:", ("Total Streams", "Average Streams"))
        top_10_countries_df = country_stats.nlargest(10, ranking_metric)
        st.write(f"**Top 10 Countries by {ranking_metric}**")
        st.dataframe(top_10_countries_df, use_container_width=True, hide_index=True)

        df_to_plot = df[df['country'].isin(top_10_countries_df['country'].tolist())]

        display_top_countries_trend_chart(
            df_to_plot,
            anomalies=st.session_state.get("automated_analysis_main_chart_anomalies"),
            spikes=st.session_state.get("geo_spikes")
        )

        if 'geo_spikes' not in st.session_state:
            st.session_state.geo_spikes = None

        with st.expander("Find Streaming Spikes (Moving Average)"):
            st.markdown("This tool identifies significant streaming spikes using a moving average.")
            
            col1, col2 = st.columns(2)
            with col1:
                p_std_dev_threshold = st.slider("Standard Deviation Threshold", 0.5, 3.0, 1.0, 0.1, key="spike_std_dev", help="How many standard deviations from the rolling average a peak must be to be flagged.")
            with col2:
                p_window_size = st.slider("Rolling Window (days)", 3, 30, 7, 1, key="spike_window_size", help="The number of days to include in the moving average calculation.")

            if st.button("Find Spikes", key="find_geo_spikes", use_container_width=True, type="primary"):
                all_spikes = {}
                with st.spinner("Analyzing spikes for each country..."):
                    for country_name in top_10_countries_df['country'].tolist():
                        country_series = df_to_plot[df_to_plot['country'] == country_name].set_index('date')['streams'].sort_index()
                        spikes_df = find_spikes_by_std_dev(series=country_series, threshold=p_std_dev_threshold, window_size=p_window_size)
                        if not spikes_df.empty:
                            all_spikes[country_name] = spikes_df
                st.session_state.geo_spikes = all_spikes
                st.rerun()

        with st.expander("Advanced Spike Detection (Wavelet)"):
            st.markdown("This tool uses Wavelet Transforms to identify spikes. It's often more effective than moving averages for signals with changing trends as it doesn't rely on a fixed window.")
            
            col1, col2 = st.columns(2)
            with col1:
                p_wavelet_family = st.selectbox("Wavelet Family", pywt.families(short=False), index=pywt.families(short=False).index('Daubechies'), key="wavelet_family_geo")
            with col2:
                p_wavelet_sensitivity = st.slider("Detection Sensitivity", 1.0, 5.0, 3.0, 0.1, key="wavelet_sensitivity_geo", help="Higher value means fewer, more significant spikes will be detected.")

            if 'wavelet_plots_geo' not in st.session_state:
                st.session_state.wavelet_plots_geo = {}

            if st.button("Run Wavelet Analysis", key="find_wavelet_spikes_geo", use_container_width=True, type="primary"):
                all_spikes_wavelet = {}
                wavelet_plots = {}
                with st.spinner("Analyzing spikes using wavelets for each country..."):
                    for country_name in top_10_countries_df['country'].tolist():
                        country_series = df_to_plot[df_to_plot['country'] == country_name].set_index('date')['streams'].sort_index()
                        dates_for_analysis = country_series.index.strftime('%Y-%m-%d').tolist()
                        streams_for_analysis = country_series.values.tolist()
                        
                        data_tuple = (tuple(dates_for_analysis), tuple(streams_for_analysis))
                        result_json = detect_wavelet_spikes(data_tuple=data_tuple, wavelet=p_wavelet_family, sensitivity=p_wavelet_sensitivity)
                        results = json.loads(result_json)
                        
                        if not results.get("error"):
                            if results.get("anomalies"):
                                spikes_df = pd.DataFrame(results['anomalies']).set_index(pd.to_datetime(pd.DataFrame(results['anomalies'])['date']))
                                all_spikes_wavelet[country_name] = spikes_df
                            if results.get("plot_json"):
                                wavelet_plots[country_name] = results['plot_json']

                st.session_state.geo_spikes = all_spikes_wavelet
                st.session_state.wavelet_plots_geo = wavelet_plots
                st.rerun()

            if st.session_state.get("wavelet_plots_geo"):
                st.markdown("---")
                st.subheader("Wavelet Analysis Results")

                detected_spikes_dict = st.session_state.get("geo_spikes", {})
                if not detected_spikes_dict:
                    st.success("Analysis complete. No significant spikes were found with the current settings.")

                # Loop through each country's plot and display it using the new function
                for country, plot_json in st.session_state.wavelet_plots_geo.items():
                    spikes_for_country = detected_spikes_dict.get(country)
                    display_single_wavelet_analysis(
                        plot_json=plot_json,
                        spikes_df=spikes_for_country,
                        chart_key=f"wavelet_chart_geo_{country}",
                        title=f"Decomposition for {country}"
                    )

                if st.button("Clear Wavelet Results", key="clear_wavelet_geo", use_container_width=True):
                    st.session_state.wavelet_plots_geo = {}
                    st.session_state.geo_spikes = None
                    st.rerun()


        st.markdown("---")
        with st.expander("Automated Divergence Discovery"):
            if 'top_10_countries_df' in locals() and not top_10_countries_df.empty:
                analysis_state_key = "automated_divergence_results"

                if st.button("Run Automated Analysis", key="run_auto_analysis", type="primary", use_container_width=True):
                    with st.spinner("Performing automated cross-validation... This will take some time."):
                        data_dict = {
                            country_name: df[df['country'] == country_name].groupby('date')['streams'].sum().reset_index()
                            for country_name in top_10_countries_df['country'].tolist()
                        }
                        results = run_automated_divergence_analysis(data_dict)
                        st.session_state[analysis_state_key] = results

                        main_chart_anomalies = []
                        if not results.get("error"):
                            for analysis in results.get("analysis_results", []):
                                for period in analysis.get("anomalous_periods", []):
                                    main_chart_anomalies.append({
                                        'start': period['start'],
                                        'end': period['end'],
                                        'country': analysis['country']
                                    })
                        st.session_state["automated_analysis_main_chart_anomalies"] = main_chart_anomalies
                        st.rerun()

                if analysis_state_key in st.session_state:
                    results = st.session_state[analysis_state_key]
                    if results.get("error"):
                        st.error(f"Analysis failed: {results['error']}")
                    else:
                        st.subheader("Automated Analysis Results")
                        st.write(f"**Genuine Countries (Stable Baseline):** {', '.join(results['genuine_countries'])}")
                        st.write(f"**Divergent Countries (Outliers):** {', '.join(results['divergent_countries'])}")

                        if results.get("debug_plot_data") and st.checkbox("Show Divergence Score Plot", key="show_auto_debug"):
                            debug_data = results["debug_plot_data"]
                            scores_df = pd.DataFrame(debug_data["divergence_scores"])

                            fig_scores = px.bar(
                                scores_df, x='country', y='score', title="Divergence Scores by Country",
                                labels={'score': 'Divergence Score (Lower is more "Genuine")', 'country': 'Country'}, color='country'
                            )
                            fig_scores.add_hline(
                                y=debug_data["score_threshold"], line_dash="dash", line_color="red",
                                annotation_text="Outlier Threshold", annotation_position="bottom right"
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)

                        st.markdown("---")

                        if not results['analysis_results']:
                            st.info("No divergent countries were identified to analyze further.")

                        for analysis in results['analysis_results']:
                            st.subheader(f"Divergence Details for: {analysis['country']}")
                            anomalous_periods = analysis.get("anomalous_periods", [])
                            if anomalous_periods:
                                st.write(f"Found {len(anomalous_periods)} period(s) of significant trend divergence.")
                            else:
                                st.write("No significant periods of trend divergence were found.")

                            fig_trends = go.Figure()
                            baseline_trend_df = pd.DataFrame(analysis['baseline_trend'])
                            fig_trends.add_trace(go.Scatter(x=baseline_trend_df['ds'], y=baseline_trend_df['trend'], mode='lines', name=f"Genuine Baseline Trend", line=dict(color='cyan', width=4, dash='dash')))
                            test_trend_df = pd.DataFrame(analysis['test_trend'])
                            fig_trends.add_trace(go.Scatter(x=test_trend_df['ds'], y=test_trend_df['trend'], mode='lines', name=f"Trend for {analysis['country']}", line=dict(color='white', width=2)))
                            for period in anomalous_periods:
                                fig_trends.add_vrect(x0=period['start'], x1=period['end'], fillcolor="rgba(255, 0, 0, 0.2)", layer="below", line_width=0)
                            fig_trends.update_layout(title=f"Normalized Trend Comparison: {analysis['country']}", yaxis_title="Normalized Trend Value", hovermode="x unified")
                            st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("Data for top countries is needed to run this analysis.")
    else:
        st.info("Although local streaming data exists, it could not be processed.")
else:
    st.info("No local streaming data available to analyze for this period.")


# --- Streaming Audience Analysis ---
st.markdown("---")
st.subheader("Streaming Audience Analysis")
if streaming_data:
    processed_sa_data = []
    for entry in streaming_data:
        date_val, value = entry.get('date'), entry.get('value')
        if date_val and value is not None:
            processed_sa_data.append({'date': pd.to_datetime(date_val), 'streams': value})

    if processed_sa_data:
        df_sa = pd.DataFrame(processed_sa_data).set_index('date').sort_index()

        if 'streaming_audience_spikes' not in st.session_state:
            st.session_state.streaming_audience_spikes = None

        # Fetch event data to pass to the chart
        events_data = get_artist_events(db_manager, artist_uuid)

        display_streaming_audience_chart(
            db_manager,
            streaming_data, 
            spikes=st.session_state.streaming_audience_spikes,
            events=events_data
        )

        with st.expander("Find Streaming Audience Spikes (Moving Average)"):
            st.markdown("""
            This tool identifies significant streaming spikes using a moving average on the overall streaming audience data.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                p_sa_std_dev_threshold = st.slider(
                    "Standard Deviation Threshold", 0.5, 3.0, 1.0, 0.1,
                    key="sa_spike_std_dev",
                    help="How many standard deviations from the rolling average a peak must be to be flagged."
                )
            with col2:
                p_sa_window_size = st.slider(
                    "Rolling Window (days)", 3, 30, 7, 1,
                    key="sa_spike_window_size",
                    help="The number of days to include in the moving average calculation."
                )

            if st.button("Find Spikes", key="find_sa_spikes", use_container_width=True, type="primary"):
                spikes_df = find_spikes_by_std_dev(
                    series=df_sa['streams'],
                    threshold=p_sa_std_dev_threshold,
                    window_size=p_sa_window_size
                )
                st.session_state.streaming_audience_spikes = spikes_df
                st.rerun()

            if st.session_state.get("streaming_audience_spikes") is not None:
                st.markdown("---")
                st.subheader("Spike Detection Results")
                detected_spikes_sa = st.session_state.streaming_audience_spikes
                
                if detected_spikes_sa.empty:
                    st.success("Analysis complete. No significant spikes were found with the current settings.")
                else:
                    st.success(f"Found {len(detected_spikes_sa)} spikes.")
                    st.dataframe(detected_spikes_sa.style.format({"streams": "{:,.0f}"}), use_container_width=True)

                st.markdown("---")
                st.subheader("Statistical Analysis Chart")
                display_single_timeseries_stats_chart(
                    df=df_sa, 
                    std_threshold=p_sa_std_dev_threshold, 
                    window_size=p_sa_window_size
                )

                if st.button("Clear Spike Results", key="clear_sa_spikes", use_container_width=True):
                    st.session_state.streaming_audience_spikes = None
                    st.rerun()

        with st.expander("Advanced Streaming Audience Spike Detection (Wavelet)"):
            st.markdown("This tool uses Wavelet Transforms to identify spikes in the overall streaming audience data.")
            
            col1, col2 = st.columns(2)
            with col1:
                p_sa_wavelet_family = st.selectbox("Wavelet Family", pywt.families(short=False), index=pywt.families(short=False).index('Daubechies'), key="wavelet_family_sa")
            with col2:
                p_sa_wavelet_sensitivity = st.slider("Detection Sensitivity", 1.0, 5.0, 3.0, 0.1, key="wavelet_sensitivity_sa", help="Higher value means fewer, more significant spikes will be detected.")

            # Initialize session state keys if they don't exist
            if 'wavelet_plot_sa' not in st.session_state:
                st.session_state.wavelet_plot_sa = None
            if 'wavelet_analysis_error_sa' not in st.session_state:
                st.session_state.wavelet_analysis_error_sa = None


            if st.button("Run Wavelet Analysis", key="find_wavelet_spikes_sa", use_container_width=True, type="primary"):
                with st.spinner("Running wavelet analysis..."):
                    dates_for_analysis = df_sa.index.strftime('%Y-%m-%d').tolist()
                    streams_for_analysis = df_sa['streams'].tolist()
                    data_tuple = (tuple(dates_for_analysis), tuple(streams_for_analysis))
                    
                    result_json = detect_wavelet_spikes(data_tuple=data_tuple, wavelet=p_sa_wavelet_family, sensitivity=p_sa_wavelet_sensitivity)
                    results = json.loads(result_json)
                    
                    # Store results and errors in session state
                    st.session_state.wavelet_analysis_error_sa = results.get('error')
                    st.session_state.wavelet_plot_sa = results.get('plot_json')
                    
                    if not results.get("error") and results.get("anomalies"):
                        anomalies_df = pd.DataFrame(results['anomalies'])
                        if not anomalies_df.empty:
                            anomalies_df['date'] = pd.to_datetime(anomalies_df['date'])
                            st.session_state.streaming_audience_spikes = anomalies_df.set_index('date')
                        else:
                            st.session_state.streaming_audience_spikes = pd.DataFrame()
                    else:
                        st.session_state.streaming_audience_spikes = pd.DataFrame()
                
                st.rerun()

            # --- MODIFIED DISPLAY LOGIC ---
            # Check for and display any errors first
            if st.session_state.get("wavelet_analysis_error_sa"):
                st.error(f"Analysis Failed: {st.session_state.wavelet_analysis_error_sa}")
            
            # If no error, check for a plot to display
            elif st.session_state.get("wavelet_plot_sa"):
                st.markdown("---")
                st.subheader("Wavelet Analysis Results")
                
                display_single_wavelet_analysis(
                    plot_json=st.session_state.wavelet_plot_sa,
                    spikes_df=st.session_state.get("streaming_audience_spikes"),
                    chart_key="wavelet_chart_sa",
                    title="Decomposition Plot"
                )
            
            # Show the clear button if there are any results (plot or error) to clear
            if st.session_state.get("wavelet_plot_sa") or st.session_state.get("wavelet_analysis_error_sa"):
                if st.button("Clear Wavelet Results", key="clear_wavelet_sa", use_container_width=True):
                    st.session_state.wavelet_plot_sa = None
                    st.session_state.streaming_audience_spikes = None
                    st.session_state.wavelet_analysis_error_sa = None
                    st.rerun()


    else:
        st.info("Could not process streaming audience data.")
else:
    st.info("No streaming audience data available for this period.")

st.markdown("---")
st.subheader("Daily Streaming Composition by Song")
st.caption("This stacked bar chart shows the daily artist streams broken down by individual song contributions (top 10 songs + 'Other').")

if st.button("Update Song Audience Data", use_container_width=True, type="primary"):
    update_song_audience_data(artist_uuid, start_date_filter, end_date_filter)

# Fetch all songs
songs = get_all_songs_for_artist_from_db(db_manager, artist_uuid)

# Fetch and process song audience data
song_daily_data = {}
for song in songs:
    song_uuid = song['uuid']
    song_aud = get_song_audience_data(db_manager, song_uuid, "spotify", start_date_filter, end_date_filter)
    if song_aud:
        # Parse cumulative data
        parsed_cum = parse_and_prepare_df(song_aud, 'value', song['name'])
        if parsed_cum is not None:
            # Compute daily streams
            daily_streams = parsed_cum.diff().clip(lower=0).fillna(0)
            song_daily_data[song['name']] = daily_streams

if song_daily_data:
    # Concatenate all daily data into one DataFrame
    all_daily_df = pd.concat(song_daily_data.values(), axis=1).fillna(0)

    # Limit to top 10 songs by total streams
    total_per_song = all_daily_df.sum()
    top_songs = total_per_song.nlargest(10).index
    other = all_daily_df.drop(columns=top_songs, errors='ignore').sum(axis=1)
    plot_df = all_daily_df[top_songs]
    plot_df['Other'] = other

    # Create stacked bar chart
    fig = px.bar(
        plot_df.reset_index(),
        x='date',
        y=plot_df.columns,
        title="",
        labels={'value': 'Daily Streams', 'date': 'Date', 'variable': 'Song'}
    )
    fig.update_layout(
        barmode='stack',
        xaxis_title="Date",
        yaxis_title="Daily Streams",
        yaxis_tickformat=",.0f",
        hovermode="x unified",
        legend_title_text='Songs'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No song audience data available to display the composition chart.")

st.markdown("---")
st.subheader("Event Management")
st.caption("First, fetch the raw event list. Then, fetch the details for all associated venues and festivals.")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch/Refresh Event List", use_container_width=True, type="secondary", help="Retrieves the full event history from the API. This will overwrite any existing events in the database."):
            fetch_and_store_events(artist_uuid)
    with col2:
        if st.button(
            "Fetch Venue/Festival Details", 
            use_container_width=True, 
            type="primary", 
            help="Scans stored events and fetches details for all associated venues and festivals, storing them in the database."
        ):
            # --- MODIFIED: Show fetched metadata inline ---
            fetched_metadata = []

            def fetch_and_display_venue_festival_metadata(artist_uuid):
                try:
                    with st.spinner("Scanning events for venue and festival information..."):
                        events_from_db = get_artist_events(db_manager, artist_uuid)
                        if not events_from_db:
                            st.info("No events found in the database. Please fetch the event list first.")
                            return []

                        venue_uuids = set()
                        festival_uuids = set()
                        for event in events_from_db:
                            if event.get('type') == 'festival' and event.get('festival', {}).get('uuid'):
                                festival_uuids.add(event['festival']['uuid'])
                            elif event.get('venue', {}).get('uuid'):
                                venue_uuids.add(event['venue']['uuid'])

                        if not venue_uuids and not festival_uuids:
                            st.info("No venues or festivals were found in the stored event data.")
                            return []

                        st.info(f"Found {len(venue_uuids)} unique venues and {len(festival_uuids)} unique festivals. Fetching details now...")

                        progress_bar = st.progress(0, text="Fetching metadata...")
                        stored_count = 0
                        results_to_display = []  # <--- A list to collect all results

                        with ThreadPoolExecutor(max_workers=20) as executor:
                            future_to_type = {}
                            for uuid in venue_uuids:
                                future = executor.submit(api_client.get_venue_metadata, uuid)
                                future_to_type[future] = ('venue', uuid)
                            for uuid in festival_uuids:
                                future = executor.submit(api_client.get_festival_metadata, uuid)
                                future_to_type[future] = ('festival', uuid)
                            total_futures = len(future_to_type)
                            for i, future in enumerate(as_completed(future_to_type)):
                                metadata_type, uuid = future_to_type[future]
                                try:
                                    metadata = future.result()
                                    if metadata and 'error' not in metadata:
                                        results_to_display.append(dict(metadata_type=metadata_type, uuid=uuid, data=metadata))
                                        # Also store to DB immediately, to not break current logic
                                        if metadata_type == 'venue':
                                            db_manager.store_venue_metadata(metadata)
                                        else:
                                            db_manager.store_festival_metadata(metadata)
                                        stored_count += 1
                                except Exception as e:
                                    st.warning(f"A metadata fetch failed: {e}")
                                progress_bar.progress((i + 1) / total_futures, text=f"Fetching metadata... ({i+1}/{total_futures})")

                        if stored_count > 0:
                            st.success(f"Successfully fetched and stored metadata for {stored_count} venues/festivals.")
                        else:
                            st.info("Completed. No new metadata was found or stored.")

                        return results_to_display

                except Exception as e:
                    st.error(f"An unexpected error occurred during the metadata fetch process: {e}")
                    return []

            # --- Run the wrapped function, return the list
            fetched_metadata = fetch_and_display_venue_festival_metadata(artist_uuid)

            # --- Show a preview table or raw JSON in Streamlit ---
            if fetched_metadata:
                st.markdown("### Preview of fetched venue/festival metadata (not from DB)")
                # Just show first N items for readability
                for item in fetched_metadata[:10]:
                    st.markdown(f"**Type**: {item['metadata_type']}, **UUID**: `{item['uuid']}`")
                    st.json(item['data'])
                if len(fetched_metadata) > 10:
                    st.info(f"Showing first 10 of {len(fetched_metadata)} total. Data is fetched directly from API -- not DB.")
            else:
                st.info("No data was fetched from the API. Check API client or event data.")

# Display the events table from the database
st.markdown("---")
with st.spinner("Loading event data from database..."):
    events_data = get_artist_events(db_manager, artist_uuid)
display_artist_events(db_manager, events_data)
