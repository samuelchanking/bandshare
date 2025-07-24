# streamlit_ui.py

# --- ADD THESE IMPORTS TO THE TOP OF THE FILE ---
import json
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy.interpolate import splrep, BSpline
from analysis_tools import (
    detect_anomalous_spikes, convert_cumulative_to_daily,
    detect_additional_contribution,
    detect_prophet_anomalies,
    clean_and_prepare_cumulative_data,
    detect_wavelet_spikes,
    adjust_and_plot_fast
)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from streamlit_caching import (
    get_song_details, get_full_song_data_from_db,
    get_song_audience_data, get_playlist_audience_data, get_playlists_for_song,
    get_tracks_for_playlist, get_playlist_placements_for_songs,
    get_global_song_audience_data,
    get_song_popularity_data,
)
from datetime import datetime, date, timedelta
import plotly.express as px
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed
from client_setup import initialize_clients
import config
from itertools import zip_longest
import base64
import plotly.io as pio
import textwrap

# --- Helper to get api_client and db_manager ---
@st.cache_resource
def get_clients():
    api_client, db_manager = initialize_clients(config)
    return api_client, db_manager


def _get_optimal_y_range(dataframe, columns):
    """Calculates an optimal Y-axis range with padding for a given set of data columns."""
    min_val = float('inf')
    max_val = float('-inf')

    for col in columns:
        if col in dataframe.columns and not dataframe[col].dropna().empty:
            min_val = min(min_val, dataframe[col].min())
            max_val = max(max_val, dataframe[col].max())

    if min_val == float('inf') or max_val == float('-inf'):
        return None

    if pd.isna(min_val) or pd.isna(max_val):
        return None

    if min_val == max_val:
        return [min_val - 1, max_val + 1]

    padding = (max_val - min_val) * 0.1
    if padding == 0: padding = 1

    return [min_val - padding, max_val + padding]


def display_all_songs_audience_chart(songs_with_data):
    """
    Displays a multi-line chart comparing the audience for multiple songs.
    """
    st.subheader("All Songs Audience Comparison")
    if not songs_with_data:
        st.info("No song audience data is available to display. Click the update button to fetch it.")
        return

    all_dfs = []
    for song_name, audience_data in songs_with_data.items():
        if audience_data:
            parsed_data = []
            history = audience_data
            if isinstance(audience_data, dict):
                history = audience_data.get('history', audience_data.get('items', audience_data))

            if not isinstance(history, list):
                st.warning(f"Could not parse audience data for '{song_name}'. Expected a list.")
                continue

            for entry in history:
                date_val = entry.get('date')
                if not date_val: continue

                value = None
                if 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
                    value = entry['plots'][0].get('value')
                elif 'value' in entry:
                    value = entry.get('value')

                if value is not None:
                    parsed_data.append({'date': pd.to_datetime(date_val), 'value': value})

            if parsed_data:
                song_df = pd.DataFrame(parsed_data).set_index('date')
                song_df.rename(columns={'value': song_name}, inplace=True)
                all_dfs.append(song_df)

    if not all_dfs:
        st.warning("Could not parse any valid data points to plot.")
        return

    combined_df = pd.concat(all_dfs, axis=1)
    combined_df.sort_index(inplace=True)

    fig = go.Figure()
    for col in combined_df.columns:
        fig.add_trace(go.Scatter(
            x=combined_df.index,
            y=combined_df[col],
            mode='lines',
            name=col,
            connectgaps=False
        ))

    y_range = _get_optimal_y_range(combined_df, combined_df.columns)
    fig.update_layout(
        title="Song Audience Over Time",
        yaxis_range=y_range,
        yaxis_tickformat=",.0f",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def display_demographics(local_audience):
    """Displays local audience data in raw JSON format."""
    st.subheader("Demographics")
    with st.expander("Local Audience (Instagram)"):
        if local_audience:
            if '_id' in local_audience:
                del local_audience['_id']
            st.json(local_audience)
        else:
            st.info("No local audience data available.")


def display_artist_metadata(metadata):
    """
    Displays artist's metadata in a structured layout.
    """
    if not metadata:
        st.warning("No artist metadata available.")
        return

    metadata_obj = metadata.get('object', metadata)

    st.header(f"Artist Overview: {metadata_obj.get('name', 'Unknown Artist')}")
    st.markdown("---")

    col1, col2 = st.columns([1, 4])
    with col1:
        if image_url := metadata_obj.get("imageUrl"):
            st.image(image_url, width=150)

    with col2:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Country", metadata_obj.get("countryCode", "N/A"))
        m_col2.metric("Type", metadata_obj.get("type", "N/A").capitalize())
        m_col3.metric("Career Stage", metadata_obj.get("careerStage", "N/A").replace("_", " ").title())
        st.write("")

        genres = metadata_obj.get('genres', [])
        if genres:
            genre_tags = []
            for genre_info in genres:
                root_genre = genre_info.get('root', '').capitalize()
                sub_genres = [g.capitalize() for g in genre_info.get('sub', [])]
                if root_genre:
                    genre_tags.append(root_genre)
                genre_tags.extend(sub_genres)
            st.write(f"**Genres:** {', '.join(genre_tags)}")

    biography = metadata_obj.get('biography')
    if biography:
        st.markdown("---")
        st.subheader("Biography")
        st.markdown(biography)

    st.markdown("---")


def display_full_song_streaming_chart(db_manager, history_data: list, entry_points: list, chart_key: str, song_release_date: str = None):
    if not history_data:
        st.info("No streaming data available for this song.")
        return

    df_cumulative, decreasing_dates = clean_and_prepare_cumulative_data(history_data)

    if df_cumulative.empty:
        st.warning("Could not parse valid data points after cleaning.")
        return

    interpolated_df = pd.DataFrame()
    if not df_cumulative.empty and len(df_cumulative) > 1:
        df_for_interp = df_cumulative.set_index('date')
        # Modified: Use midnight timestamps for reindexing
        full_date_range = pd.date_range(
            start=df_for_interp.index.min().floor('D'), 
            end=df_for_interp.index.max().floor('D'), 
            freq='D'
        )
        interpolated_df = df_for_interp.reindex(full_date_range).interpolate(method='time')
        
        # No need for 'interpolated' flag since all points are now at midnights and potentially interpolated
        interpolated_df['hover_text'] = ''  # Simplified, no (interpolated) note
        
        # Remove any points that were originally decreasing values (but now with adjusted times, map approximately)
        if not decreasing_dates.empty:
            for dec_date in decreasing_dates:
                closest_midnight = dec_date.floor('D')
                if closest_midnight in interpolated_df.index:
                    interpolated_df.loc[closest_midnight, 'value'] = np.nan
            interpolated_df['value'] = interpolated_df['value'].interpolate(method='time')  # Re-interpolate if needed

    df_daily = pd.DataFrame()
    if not interpolated_df.empty:
        daily_values = interpolated_df['value'].diff().clip(lower=0)
        df_daily = pd.DataFrame({'date': daily_values.index, 'value': daily_values.values})
        if not df_daily.empty:
            df_daily = df_daily.iloc[1:].copy() if len(df_daily) > 1 else pd.DataFrame()

    # The rest of the function (chart plotting) remains mostly unchanged, but update cumulative chart hover without customdata if needed
    st.subheader("Raw Data: Total Accumulated Streams")
    st.caption("This chart shows the cumulative stream data interpolated at midnight timestamps after adjustment.")
    fig_cumulative = go.Figure()

    fig_cumulative.add_trace(go.Scatter(
        x=interpolated_df.index, y=interpolated_df['value'],
        mode='lines', name='Total Streams',
        connectgaps=True,
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Streams</b>: %{y:,.0f}<extra></extra>'
    ))

    fig_cumulative.update_layout(showlegend=False, yaxis_title="Cumulative Streams")
    st.plotly_chart(fig_cumulative, use_container_width=True, key=f"cumulative_chart_{chart_key}")
    
        
    st.markdown("---")
    st.subheader("Calculated Data: Daily Streams")
    st.caption("This chart shows the calculated day-over-day change from the cleaned data.")

    if df_daily.empty:
        st.info("Not enough data to calculate daily stream changes.")
        return

    if len(df_daily) > 1:
        df_daily = df_daily.iloc[1:].copy()
        
    

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['value'], mode='lines', name='Daily Streams',
        connectgaps=False, 
        line=dict(color='#1f77b4'),
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily Streams</b>: %{y:,.0f}<extra></extra>'
    ))
    
    sorted_entry_points = sorted(entry_points, key=lambda x: x.get('entryDate', ''))
    last_entry_date = None
    y_shift_offset = 15
    y_max_daily = df_daily['value'].max() if not df_daily.empty else 0
    for entry in sorted_entry_points:
        try:
            entry_date_dt = datetime.fromisoformat(entry['entryDate'].replace('Z', '+00:00'))
            entry_date_val = entry_date_dt.date()
            if last_entry_date and (entry_date_val - last_entry_date) < timedelta(days=30): y_shift_offset += 35
            else: y_shift_offset = 15
            fig_daily.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            playlist_name = entry.get('name', 'N/A')
            playlist_uuid = entry.get('uuid')
            entry_subscribers = entry.get('entrySubscribers')
            if entry_subscribers is None and playlist_uuid: entry_subscribers = db_manager.get_timeseries_value_for_date('playlist_audience', {'playlist_uuid': playlist_uuid}, entry_date_dt)
            subscribers_formatted = f"{entry_subscribers:,}" if entry_subscribers is not None else "N/A"
            annotation_text = f"{playlist_name}<br>{subscribers_formatted} subs"
            hover_text = f"Added to '{playlist_name}' ({subscribers_formatted} subs) on {entry_date_val.strftime('%Y-%m-%d')}"
            fig_daily.add_annotation(x=entry_date_val, y=y_max_daily, text=annotation_text, showarrow=False, yshift=y_shift_offset, font=dict(color="white"), bgcolor="rgba(255, 0, 0, 0.6)", borderpad=4, hovertext=hover_text)
            last_entry_date = entry_date_val
        except (ValueError, KeyError) as e:
            print(f"Could not process playlist annotation. Error: {e}")
            continue
    if song_release_date:
        try:
            release_date_dt = pd.to_datetime(song_release_date)
            if not df_daily.empty and 'date' in df_daily.columns:
                chart_start_date = df_daily['date'].min()
                chart_end_date = df_daily['date'].max()
                if not pd.isna(chart_start_date) and not pd.isna(chart_end_date) and chart_start_date <= release_date_dt <= chart_end_date:
                    fig_daily.add_vline(x=release_date_dt, line_width=2, line_dash="solid", line_color="green")
                    fig_daily.add_annotation(x=release_date_dt, y=y_max_daily, text="Song Release", showarrow=True, arrowhead=1, ax=-25, ay=-60, font=dict(color="green"), bgcolor="rgba(0, 200, 0, 0.6)")
        except (ValueError, TypeError) as e:
            st.warning(f"Could not plot release date marker due to an issue with the date format: {e}")
    
    y_range = _get_optimal_y_range(df_daily, ['value'])
    fig_daily.update_layout(yaxis_range=y_range, showlegend=False, hovermode="x unified", yaxis_tickformat=",.0f")
    st.plotly_chart(fig_daily, use_container_width=True, key=f"daily_chart_{chart_key}")
    st.markdown("---")
    st.subheader("Dual Axis View: Daily (Left) and Cumulative (Right) Streams")
    st.caption("This chart combines daily and cumulative streams on a dual y-axis for comparison.")

    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

    # Add daily streams to primary y-axis (left)
    fig_dual.add_trace(
        go.Scatter(
            x=df_daily['date'], 
            y=df_daily['value'], 
            mode='lines', 
            name='Daily Streams',
            connectgaps=False, 
            line=dict(color='#1f77b4'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily Streams</b>: %{y:,.0f}<extra></extra>'
        ), 
        secondary_y=False
    )

    # Add cumulative streams to secondary y-axis (right)
    fig_dual.add_trace(
        go.Scatter(
            x=interpolated_df.index, 
            y=interpolated_df['value'],
            mode='lines', 
            name='Cumulative Streams',
            connectgaps=True,
            line=dict(color='#ff7f0e'),
            customdata=interpolated_df['hover_text'],
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Cumulative Streams</b>: %{y:,.0f}%{customdata}<extra></extra>'
        ), 
        secondary_y=True
    )

    # Add playlist entry annotations (copied from fig_daily)
    y_max_daily = df_daily['value'].max() if not df_daily.empty else 0
    sorted_entry_points = sorted(entry_points, key=lambda x: x.get('entryDate', ''))
    last_entry_date = None
    y_shift_offset = 15
    for entry in sorted_entry_points:
        try:
            entry_date_dt = datetime.fromisoformat(entry['entryDate'].replace('Z', '+00:00'))
            entry_date_val = entry_date_dt.date()
            if last_entry_date and (entry_date_val - last_entry_date) < timedelta(days=30): y_shift_offset += 35
            else: y_shift_offset = 15
            fig_dual.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            playlist_name = entry.get('name', 'N/A')
            playlist_uuid = entry.get('uuid')
            entry_subscribers = entry.get('entrySubscribers')
            if entry_subscribers is None and playlist_uuid: entry_subscribers = db_manager.get_timeseries_value_for_date('playlist_audience', {'playlist_uuid': playlist_uuid}, entry_date_dt)
            subscribers_formatted = f"{entry_subscribers:,}" if entry_subscribers is not None else "N/A"
            annotation_text = f"{playlist_name}<br>{subscribers_formatted} subs"
            hover_text = f"Added to '{playlist_name}' ({subscribers_formatted} subs) on {entry_date_val.strftime('%Y-%m-%d')}"
            fig_dual.add_annotation(x=entry_date_val, y=y_max_daily, text=annotation_text, showarrow=False, yshift=y_shift_offset, font=dict(color="white"), bgcolor="rgba(255, 0, 0, 0.6)", borderpad=4, hovertext=hover_text)
            last_entry_date = entry_date_val
        except (ValueError, KeyError) as e:
            print(f"Could not process playlist annotation. Error: {e}")
            continue

    # Add song release date marker (copied from fig_daily)
    if song_release_date:
        try:
            release_date_dt = pd.to_datetime(song_release_date)
            if not df_daily.empty and 'date' in df_daily.columns:
                chart_start_date = df_daily['date'].min()
                chart_end_date = df_daily['date'].max()
                if not pd.isna(chart_start_date) and not pd.isna(chart_end_date) and chart_start_date <= release_date_dt <= chart_end_date:
                    fig_dual.add_vline(x=release_date_dt, line_width=2, line_dash="solid", line_color="green")
                    fig_dual.add_annotation(x=release_date_dt, y=y_max_daily, text="Song Release", showarrow=True, arrowhead=1, ax=-25, ay=-60, font=dict(color="green"), bgcolor="rgba(0, 200, 0, 0.6)")
        except (ValueError, TypeError) as e:
            st.warning(f"Could not plot release date marker due to an issue with the date format: {e}")

    # Set optimal y-ranges
    y_range_daily = _get_optimal_y_range(df_daily, ['value'])
    y_range_cum = _get_optimal_y_range(interpolated_df, ['value'])

    # Update layout
    fig_dual.update_layout(
        yaxis_range=y_range_daily,
        yaxis2_range=y_range_cum,
        yaxis_title="Daily Streams",
        yaxis2_title="Cumulative Streams",
        yaxis_tickformat=",.0f",
        yaxis2_tickformat=",.0f",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig_dual, use_container_width=True, key=f"dual_chart_{chart_key}")

    analysis_session_key = f"analysis_results_{chart_key}"
    with st.expander("🔬 Analyze Streaming Spikes"):
        st.markdown("""
        This tool detects anomalous spikes by analyzing the rate of change of the cumulative stream count. It uses a moving average to smooth the data and identify significant deviations.
        - **Discretization Step:** The window (in days) for analyzing the rate of change.
        - **Smoothing Window Size:** The number of data points to include in the moving average for smoothing the trend line. A larger window creates a smoother line.
        - **Sensitivity:** How far from the smoothed average the rate of change must be to be flagged as a spike.
        """)
        param_cols = st.columns(3)
        p_step = param_cols[0].number_input("Discretization Step (days)", min_value=1, max_value=30, value=7, step=1, key=f"step_{chart_key}")
        p_smooth = param_cols[1].number_input("Smoothing Window Size", min_value=2, max_value=30, value=7, step=1, key=f"smooth_{chart_key}")
        p_sens = param_cols[2].number_input("Sensitivity", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key=f"sens_{chart_key}")
        button_cols = st.columns(2)
        if button_cols[0].button("Run Analysis", key=f"analyze_{chart_key}", use_container_width=True, type="primary"):
            dates_for_analysis = df_cumulative['date'].dt.strftime('%Y-%m-%d').tolist()
            streams_for_analysis = df_cumulative['value'].tolist()
            data_tuple = (tuple(dates_for_analysis), tuple(streams_for_analysis))
            with st.spinner("Analyzing data for anomalies..."):
                result_json = detect_anomalous_spikes(data_tuple, discretization_step=p_step, smoothing_window_size=p_smooth, sensitivity=p_sens)
                st.session_state[analysis_session_key] = json.loads(result_json)
                st.rerun()
        if button_cols[1].button("Clear Analysis", key=f"clear_{chart_key}", use_container_width=True):
            if analysis_session_key in st.session_state:
                del st.session_state[analysis_session_key]
            st.rerun()
        if st.session_state.get(analysis_session_key):
            results = st.session_state[analysis_session_key]
            if 'error' in results and results['error']: st.error(f"Analysis failed: {results['error']}")
            elif not results['anomalies']: st.success("No significant anomalies detected.")
            else:
                st.success(f"Detected {len(results['anomalies'])} potential anomalies.")
                anomaly_df = pd.DataFrame(results['anomalies'])
                anomaly_df['spike_size_streams_per_day_formatted'] = anomaly_df['spike_size_streams_per_day'].apply(lambda x: f"+{int(x):,}")
                def format_rel_sig(x):
                    if x == float('inf') or x > 10000: return "N/A (from low base)"
                    return f"{x:+.1%}"
                anomaly_df['relative_significance_formatted'] = anomaly_df['relative_significance'].apply(format_rel_sig)
                anomaly_df.rename(columns={'date': 'Date', 'spike_size_streams_per_day_formatted': 'Spike Size (Anomalous Streams/Day)', 'relative_significance_formatted': 'Relative Significance'}, inplace=True)
                st.dataframe(anomaly_df[['Date', 'Spike Size (Anomalous Streams/Day)', 'Relative Significance']], use_container_width=True, hide_index=True)
            if results.get('debug') and st.checkbox("Show debug plots", key=f"debug_cb_{chart_key}"):
                with st.container(border=True):
                    st.subheader("Debug Information")
                    try:
                        debug_data = results['debug']
                        start_date_obj = datetime.strptime(debug_data['start_date'], '%Y-%m-%d')
                        grid_dates = [start_date_obj + timedelta(days=d) for d in debug_data['grid_days']]
                        st.markdown("##### Plot 1: Rate of Change Comparison")
                        st.write("This plot shows the rate of change of the discretized data (`s_diff`) versus the rate of the smoothed average (`Av_diff`).")
                        fig_debug1 = go.Figure()
                        fig_debug1.add_trace(go.Scatter(x=grid_dates[1:], y=debug_data['s_diffs'], mode='lines', name='Discretized Rate (s_diff)'))
                        fig_debug1.add_trace(go.Scatter(x=grid_dates[1:], y=debug_data['Av_diffs'], mode='lines', name='EMA Rate (Av_diff)', line=dict(dash='dash')))
                        fig_debug1.update_layout(title_text='Rate of Change: Discretized vs. Smoothed', yaxis_title="Streams per Day", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_debug1, use_container_width=True)
                        st.markdown("##### Plot 2: Anomaly Detection")
                        st.write("An anomaly is flagged when the actual difference in rates (solid blue) crosses the dynamic threshold (red dots).")
                        fig_debug2 = go.Figure()
                        fig_debug2.add_trace(go.Scatter(x=grid_dates[1:], y=debug_data['abs_diff_devs'], mode='lines', name='Absolute Difference in Rates'))
                        threshold_values = [p_sens * std for std in debug_data['smoothed_stds']]
                        fig_debug2.add_trace(go.Scatter(x=grid_dates[1:], y=threshold_values, mode='lines', name='Detection Threshold', line=dict(color='red', dash='dot')))
                        fig_debug2.update_layout(title_text='Detection Logic', yaxis_title="Difference Score / Threshold", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_debug2, use_container_width=True)
                    except (KeyError, IndexError, TypeError) as e:
                        st.error(f"Failed to generate debug plots. Data might be missing or malformed. Error: {e}")

def display_typed_playlists(playlists: List[Dict[str, Any]], title: str):
    st.subheader(title)
    if not playlists:
        st.info("No playlists to display. Try fetching the data first.")
        return
    for playlist_chunk in grouper(playlists, 5):
        cols = st.columns(5)
        for i, playlist in enumerate(playlist_chunk):
            if playlist:
                with cols[i]:
                    playlist_name = playlist.get('name', 'N/A')
                    image_url = playlist.get('imageUrl')
                    subscribers = playlist.get('latestSubscriberCount', 0)
                    playlist_uuid = playlist.get('uuid')
                    if image_url: st.image(image_url, use_container_width=True)
                    else: st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)
                    st.caption(f"**{playlist_name}**")
                    st.write(f"{subscribers:,} Subscribers")
                    if st.button("Details", key=f"btn_global_{playlist_uuid}", use_container_width=True):
                        st.session_state.selected_playlist_uuid = playlist_uuid
                        st.rerun()

# --- NEW FUNCTION ---
def display_spike_timeline(spikes_df: pd.DataFrame, start_date: date, end_date: date):
    """
    Displays a timeline of detected spikes, grouped by category.
    """
    if spikes_df.empty:
        st.info("No spikes were detected to display on the timeline.")
        return

    # Ensure date column is in datetime format
    spikes_df['date'] = pd.to_datetime(spikes_df['date'])

    # FIXED: Check if 'category' column exists, otherwise fallback to 'metric'
    grouping_col = 'category' if 'category' in spikes_df.columns else 'metric'
    
    # Use the selected column for grouping on the y-axis
    groups = spikes_df[grouping_col].unique()
    group_levels = {group: i for i, group in enumerate(groups)}
    spikes_df['level'] = spikes_df[grouping_col].map(group_levels)

    fig = go.Figure()

    # Create a line for each group level
    for i, group in enumerate(groups):
        fig.add_shape(
            type='line',
            x0=start_date, y0=i,
            x1=end_date, y1=i,
            line=dict(color='grey', width=1, dash='dot')
        )

    # Add markers for each spike
    fig.add_trace(go.Scatter(
        x=spikes_df['date'],
        y=spikes_df['level'],
        mode='markers',
        marker=dict(
            size=15,
            symbol='x',
            color='red'
        ),
        hoverinfo='text',
        hovertext=spikes_df.apply(
            lambda row: f"<b>Date:</b> {row['date'].strftime('%Y-%m-%d')}<br>"
                        f"<b>Metric:</b> {row['metric']}<br>"
                        f"<b>Value:</b> {row['value']:,.0f}",
            axis=1
        ),
        name='Spikes'
    ))

    fig.update_layout(
        title="Spike Event Timeline",
        xaxis_title="Date",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(groups))),
            ticktext=groups,
            showgrid=False
        ),
        showlegend=False,
        height=max(400, len(groups) * 60) # Adjust height based on number of groups
    )

    st.plotly_chart(fig, use_container_width=True)


# --- MODIFIED FUNCTION ---
def display_global_playlist_tracks(db_manager, playlist_uuid: str, playlist_name: str):
    """
    Displays the full tracklist for a global playlist, with a per-song analysis panel
    and a new feature to plot overall playlist impact vs. peak position.
    """
    st.header(f"Tracklist & Analysis for: {playlist_name}")
    if st.button("⬅️ Back to all Global Playlists"):
        # Clear all session state results for this view to avoid stale data
        st.session_state.selected_playlist_uuid = None
        keys_to_clear = [key for key in st.session_state if key.startswith(('global_', 'impact_report_'))]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()

    st.code(f"Playlist UUID: {playlist_uuid}")
    st.markdown("---")
    
    with st.container(border=True):
        st.subheader("⚖️ Event Weight vs. Peak Position Analysis")
        st.caption("This report calculates the optimal event weight for each song and plots it against peak position. Run the report with different smoothing windows to compare results.")
        
        report_key = f"impact_report_dict_{playlist_uuid}"
        
        # Initialize the dictionary in session_state if it doesn't exist
        if report_key not in st.session_state:
            st.session_state[report_key] = {}

        cols = st.columns([2,1,1])
        with cols[0]:
            smoothing_for_report = st.number_input(
                "Smoothing Window (days)", min_value=1, max_value=30, value=1, step=1, 
                key=f"report_smoothing_{playlist_uuid}", 
                help="Applies a rolling average to daily data before analysis. Use 1 for no smoothing."
            )
        with cols[1]:
            st.write("") # Spacer
            if st.button("Generate Impact Report", key=f"report_btn_{playlist_uuid}", type="primary", use_container_width=True):
                with st.spinner(f"Generating report for smoothing window: {smoothing_for_report} days..."):
                    full_tracklist = list(db_manager.collections['global_song'].find({'playlist_uuid': playlist_uuid}).sort('position', 1))
                    report_data = []
                    progress_bar = st.progress(0, text="Analyzing songs...")

                    for i, track in enumerate(full_tracklist):
                        song_info = track.get('song', {})
                        song_uuid, song_name = song_info.get('uuid'), song_info.get('name', 'N/A')
                        
                        progress_text = f"Analyzing... ({i+1}/{len(full_tracklist)}) - {song_name}"
                        progress_bar.progress((i + 1) / len(full_tracklist), text=progress_text)

                        if not song_uuid: continue

                        entry_details = db_manager.collections['global_song_playlists'].find_one({'playlist.uuid': playlist_uuid, 'song_uuid': song_uuid})
                        if not entry_details or not entry_details.get('entryDate') or not entry_details.get('peakPosition'):
                            continue

                        entry_date_obj = datetime.fromisoformat(entry_details['entryDate'].replace('Z', '+00:00')).date()
                        peak_position = entry_details['peakPosition']

                        start_date = entry_date_obj - timedelta(days=90)
                        end_date = entry_date_obj + timedelta(days=90)
                        audience_data = get_global_song_audience_data(db_manager, song_uuid, start_date, end_date)
                        if not audience_data: continue

                        df_cumulative, _ = clean_and_prepare_cumulative_data(audience_data)
                        if df_cumulative.empty or len(df_cumulative) < 2: continue

                        df_for_interp = df_cumulative.set_index('date')
                        full_date_range = pd.date_range(start=df_for_interp.index.min(), end=df_for_interp.index.max(), freq='D')
                        interpolated_df = df_for_interp.reindex(full_date_range).interpolate(method='time')
                        daily_values = interpolated_df['value'].diff().clip(lower=0)
                        df_daily = pd.DataFrame({'date': daily_values.index, 'value': daily_values.values}).iloc[1:]
                        if df_daily.empty: continue

                        latest_date = df_daily['date'].max().date()
                        duration_days = (latest_date - entry_date_obj).days + 1
                        if duration_days <= 1: continue
                        
                        prophet_input_data = {'dates': df_daily['date'].dt.strftime('%Y-%m-%d').tolist(), 'streams': df_daily['value'].tolist()}
                        result_json = detect_additional_contribution(
                            data=prophet_input_data, 
                            event_date=entry_date_obj.strftime('%Y-%m-%d'), 
                            event_duration_days=duration_days, 
                            smoothing_window_size=smoothing_for_report
                        )
                        results = json.loads(result_json)

                        if not results.get('error') and 'additional_contribution' in results:
                            event_weight = results['additional_contribution'].get('optimal_prior_scale', 0)
                            report_data.append({
                                'song_name': song_name,
                                'peak_position': peak_position,
                                'event_weight': event_weight
                            })

                    # Store the generated report in the dictionary with the smoothing value as the key
                    st.session_state[report_key][smoothing_for_report] = report_data
                    progress_bar.progress(1.0, text="Report generation complete!")
                    st.rerun()
        with cols[2]:
            st.write("") # Spacer
            if st.button("Clear All Reports", use_container_width=True):
                st.session_state[report_key] = {}
                st.rerun()

        # Display all the reports that have been generated and stored
        if st.session_state.get(report_key):
            # Sort reports by smoothing window size for consistent display order
            sorted_reports = sorted(st.session_state[report_key].items())
            
            for smoothing_val, report_data in sorted_reports:
                st.markdown("---")
                st.subheader(f"Analysis Results (Smoothing Window: {smoothing_val} days)")
                if not report_data:
                    st.info("Could not generate a report for this smoothing value.")
                    continue

                df_report = pd.DataFrame(report_data).dropna()
                
                if df_report.empty:
                     st.warning("Analysis ran, but no songs had sufficient data to be included in the plot.")
                else:
                    fig = px.scatter(
                        df_report,
                        x='peak_position',
                        y='event_weight',
                        trendline="ols",
                        trendline_color_override="red",
                        hover_name='song_name',
                        title=f'Event Weight vs. Peak Position for "{playlist_name}"',
                        labels={
                            "peak_position": "Peak Position in Playlist",
                            "event_weight": "Optimal Event Weight (Scale)"
                        }
                    )
                    fig.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode='markers'))
                    fig.update_xaxes(autorange="reversed") 
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_smoothing_{smoothing_val}")


    st.markdown("---")


    # Initialize session state for this specific view if not present
    if 'global_prophet_anomaly_results' not in st.session_state:
        st.session_state.global_prophet_anomaly_results = {}
    if 'global_prophet_analysis_results' not in st.session_state:
        st.session_state.global_prophet_analysis_results = {}
    if 'global_spike_analysis_results' not in st.session_state:
        st.session_state.global_spike_analysis_results = {}

    with st.spinner("Loading playlist tracklist..."):
        # Fetch all tracks for the given playlist, sorted by position
        full_tracklist = list(db_manager.collections['global_song'].find({'playlist_uuid': playlist_uuid}).sort('position', 1))

    if not full_tracklist:
        st.warning("Could not find a tracklist for this playlist in the 'global_song' collection.")
        return

    st.subheader("Individual Track Analysis")
    st.caption("Expand the analysis panel for any song to check for streaming spikes and other insights.")


    for track in full_tracklist:
        position, song_info = track.get('position'), track.get('song', {})
        song_uuid, song_name = song_info.get('uuid'), song_info.get('name', 'N/A')
        artist_name, image_url = song_info.get('creditName', 'N/A'), song_info.get('imageUrl')

        # Find the specific entry details for this song on this playlist
        entry_details = db_manager.collections['global_song_playlists'].find_one({'playlist.uuid': playlist_uuid, 'song_uuid': song_uuid})

        with st.container(border=True):
            col1, col2 = st.columns([1, 10])
            with col1:
                st.image(image_url or "https://i.imgur.com/3gMbdA5.png", use_container_width=True)
            with col2:
                st.write(f"**{position}. {song_name}** - *{artist_name}*")
                st.caption(f"Song UUID: {song_uuid}")

                if entry_details:
                    st.markdown("---")
                    entry_date_str = entry_details.get('entryDate')
                    entry_date_obj = None
                    if entry_date_str:
                        try:
                            # Convert ISO string to a date object
                            entry_date_obj = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date()
                        except (ValueError, TypeError):
                            pass

                    if not entry_date_obj:
                        st.error("Cannot display graph without a valid entry date for this song.")
                        continue

                    # Fetch audience data for a window around the entry date
                    with st.spinner(f"Loading audience data for '{song_name}'..."):
                        start_date = entry_date_obj - timedelta(days=90)
                        end_date = entry_date_obj + timedelta(days=90)
                        audience_data = get_global_song_audience_data(db_manager, song_uuid, start_date, end_date)

                    if not audience_data:
                        st.warning("No audience data could be found or fetched for this period.")
                        continue
                    
                    # Clean the raw cumulative data
                    df_cumulative, decreasing_dates = clean_and_prepare_cumulative_data(audience_data)
    
                    if df_cumulative.empty:
                        st.info("Audience data is present but could not be parsed for graphing after cleaning.")
                        continue
    
                    interpolated_df = pd.DataFrame()
                    df_daily = pd.DataFrame()
    
                    if not df_cumulative.empty and len(df_cumulative) > 1:
                        # Prepare data for interpolation
                        df_for_interp = df_cumulative.set_index('date')
                        full_date_range = pd.date_range(start=df_for_interp.index.min(), end=df_for_interp.index.max(), freq='D')
                        interpolated_df = df_for_interp.reindex(full_date_range)
                        
                        interpolated_df['interpolated'] = interpolated_df['value'].isna()
                        interpolated_df['value'] = interpolated_df['value'].interpolate(method='time')
                        
                        # Remove any points that were originally decreasing values
                        if not decreasing_dates.empty:
                            interpolated_df.loc[interpolated_df.index.isin(decreasing_dates), 'value'] = np.nan
                            interpolated_df.loc[interpolated_df.index.isin(decreasing_dates), 'interpolated'] = False
                        
                        interpolated_df['hover_text'] = np.where(interpolated_df['interpolated'], '<br>(interpolated)', '')

                        # Calculate daily values from the cleaned, interpolated data
                        daily_values = interpolated_df['value'].diff().clip(lower=0)
                        df_daily = pd.DataFrame({'date': daily_values.index, 'value': daily_values.values}).iloc[1:]

                    with st.expander("🔬 Analyze Streaming Spikes for this Song"):
                        
                        st.markdown("##### Analysis Parameters")
                        st.caption("These parameters are shared across the analysis tools below.")
                        param_cols = st.columns(3)
                        p_smooth = param_cols[0].number_input("Smoothing Window (days)", min_value=1, max_value=30, value=7, step=1, key=f"smooth_global_{song_uuid}")
                        p_step = param_cols[1].number_input("Discretization Step (days)", min_value=1, max_value=30, value=7, step=1, key=f"step_global_{song_uuid}", help="Used for Volatility analysis.")
                        p_sens = param_cols[2].number_input("Sensitivity", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key=f"sens_global_{song_uuid}", help="Used for Volatility analysis.")

                        st.markdown("---")
                        
                        tab1, tab2, tab3 = st.tabs(["Volatility Spike Detection", "Playlist Impact Analysis", "Prophet Anomaly Detection"])

                        with tab1:
                            st.markdown("This tool detects spikes by analyzing how quickly the stream rate changes compared to its smoothed trend.")
                            if st.button("Run Volatility Analysis", key=f"run_analysis_global_{song_uuid}", use_container_width=True):
                                with st.spinner("Analyzing song for anomalies..."):
                                    dates_for_analysis = df_cumulative['date'].dt.strftime('%Y-%m-%d').tolist()
                                    streams_for_analysis = df_cumulative['value'].tolist()
                                    data_tuple = (tuple(dates_for_analysis), tuple(streams_for_analysis))
                                    result_json = detect_anomalous_spikes(data_tuple, discretization_step=p_step, smoothing_window_size=p_smooth, sensitivity=p_sens)
                                    st.session_state.global_spike_analysis_results[song_uuid] = json.loads(result_json)
                                    st.rerun()
                            
                            spike_results = st.session_state.get('global_spike_analysis_results', {}).get(song_uuid)
                            if spike_results:
                                if spike_results.get('error'): st.error(f"Analysis failed: {spike_results['error']}")
                                elif not spike_results['anomalies']: st.success("Analysis complete. No significant anomalies detected.")
                                else: st.success(f"Detected {len(spike_results['anomalies'])} potential anomalies.")
                                
                                if spike_results.get('debug') and st.checkbox("Show Volatility Debug Plots", key=f"v_debug_cb_global_{song_uuid}"):
                                     # (Debug plot logic remains the same)
                                     pass

                        with tab2:
                            st.markdown("This tool uses Prophet to estimate the additional daily streams gained *after* the song was added to this playlist. It automatically finds the best weight for the event.")
                            
                            p_smooth_prophet = st.number_input("Smoothing Window (days)", min_value=1, max_value=30, value=3, step=1, key=f"p_smooth_prophet_{song_uuid}", help="Applies a rolling average to the daily data before analysis. Default is 3. Use 1 for no smoothing.")

                            if st.button("Estimate Playlist Effect", key=f"run_prophet_analysis_global_{song_uuid}", use_container_width=True, type="primary"):
                                if not df_daily.empty and entry_date_obj and not df_daily[df_daily['date'].dt.date >= entry_date_obj].empty:
                                    with st.spinner(f"Running Prophet analysis for '{song_name}'..."):
                                        prophet_input_data = {'dates': df_daily['date'].dt.strftime('%Y-%m-%d').tolist(), 'streams': df_daily['value'].tolist()}
                                        
                                        latest_date = df_daily['date'].max().date()
                                        duration_days = (latest_date - entry_date_obj).days + 1

                                        if duration_days > 1:
                                            result_json = detect_additional_contribution(
                                                data=prophet_input_data, 
                                                event_date=entry_date_obj.strftime('%Y-%m-%d'), 
                                                event_duration_days=duration_days, 
                                                smoothing_window_size=p_smooth_prophet
                                            )
                                            st.session_state.global_prophet_analysis_results[song_uuid] = json.loads(result_json)
                                        else:
                                            st.session_state.global_prophet_analysis_results[song_uuid] = {'error': 'Not enough data after the playlist entry date to perform analysis.'}
                                        st.rerun()
                                else:
                                    st.error("Cannot run Prophet analysis. Calculated daily stream data or a valid playlist entry date is missing.")
                            
                            prophet_results = st.session_state.get('global_prophet_analysis_results', {}).get(song_uuid)
                            if prophet_results:
                                if prophet_results.get('error'): 
                                    st.error(f"Prophet analysis failed: {prophet_results['error']}")
                                elif 'additional_contribution' in prophet_results and prophet_results['additional_contribution']:
                                    st.success("Prophet analysis complete.")
                                    effect_data = prophet_results['additional_contribution']
                                    avg_streams = effect_data.get('average_additional_streams_per_day', 0)
                                    optimal_scale = effect_data.get('optimal_prior_scale')
                                    
                                    res_col1, res_col2 = st.columns(2)
                                    res_col1.metric(label="Estimated Impact (Avg. Daily Streams)", value=f"{avg_streams:,.0f}")
                                    res_col2.metric(label="Optimal Event Weight (Scale)", value=f"{optimal_scale:.2f}")

                                    if prophet_results.get('forecast_data'):
                                        st.markdown("---")
                                        st.subheader("Forecast Trajectory vs. Actual Streams")
                                        st.caption("This plot shows the model's expected stream trajectory (dashed cyan line) versus the actual daily streams (white line).")

                                        forecast_df = pd.DataFrame(prophet_results['forecast_data'])
                                        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                                        
                                        fig = go.Figure()

                                        fig.add_trace(go.Scatter(x=df_daily['date'], y=df_daily['value'], mode='lines', line=dict(color='#FFFFFF'), name='Actual Streams'))
                                        
                                        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', line=dict(color='#00FFFF', dash='dash'), name='Forecast Trajectory'))

                                        fig.add_vline(x=datetime.combine(entry_date_obj, datetime.min.time()), line_width=2, line_dash="dash", line_color="red", name="Playlist Add")

                                        fig.update_layout(
                                            yaxis_title="Daily Streams",
                                            hovermode='x unified',
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            st.markdown("This tool uses Prophet's forecast uncertainty intervals to find specific dates that are statistically anomalous.")
                            pa_interval = st.slider("Uncertainty Interval", min_value=0.8, max_value=0.99, value=0.95, step=0.01, key=f"pa_interval_global_{song_uuid}", help="A higher value makes the detection stricter, finding fewer anomalies.")
                            
                            if st.button("Find Prophet Anomalies", key=f"run_prophet_anomaly_global_{song_uuid}", use_container_width=True):
                                if not df_daily.empty:
                                    with st.spinner("Finding anomalies with Prophet..."):
                                        prophet_input_data = {'dates': df_daily['date'].dt.strftime('%Y-%m-%d').tolist(), 'streams': df_daily['value'].tolist()}
                                        result_json = detect_prophet_anomalies(data=prophet_input_data, interval_width=pa_interval)
                                        st.session_state.global_prophet_anomaly_results[song_uuid] = json.loads(result_json)
                                        st.rerun()
                                else:
                                    st.error("Cannot run Prophet anomaly detection without calculated daily stream data.")
                            
                            prophet_anomaly_results = st.session_state.get('global_prophet_anomaly_results', {}).get(song_uuid)
                            if prophet_anomaly_results:
                                if prophet_anomaly_results.get('error'): 
                                    st.error(f"Prophet anomaly analysis failed: {prophet_anomaly_results['error']}")
                                elif 'anomalies' in prophet_anomaly_results:
                                    anomalies = prophet_anomaly_results['anomalies']
                                    best_seasonality = prophet_anomaly_results.get('best_seasonality', 'N/A')
                                    
                                    st.success(f"Analysis complete. Found **{len(anomalies)}** anomalous dates.")
                                    st.metric("Most Dominant Seasonality", best_seasonality)

                                    if anomalies:
                                        st.write("Anomalous Dates:")
                                        anomaly_df = pd.DataFrame(anomalies)
                                        anomaly_df['ds'] = pd.to_datetime(anomaly_df['ds']).dt.date
                                        anomaly_df.rename(columns={'ds': 'Date', 'y': 'Actual Streams', 'yhat': 'Forecasted Streams', 'difference': 'Difference'}, inplace=True)
                                        st.dataframe(anomaly_df[['Date', 'Actual Streams', 'Forecasted Streams', 'Difference']], use_container_width=True, hide_index=True)

                                    if prophet_anomaly_results.get('plots_json'):
                                        st.markdown("---")
                                        st.subheader("Interactive Forecast Plot with Anomalies")
                                        forecast_plot_json = prophet_anomaly_results['plots_json'].get('forecast_plot')
                                        if forecast_plot_json:
                                            fig_forecast = pio.from_json(forecast_plot_json)
                                            st.plotly_chart(fig_forecast, use_container_width=True)
                                        
                                        st.subheader("Interactive Seasonality Component Plots")
                                        components_plot_json = prophet_anomaly_results['plots_json'].get('components_plot')
                                        if components_plot_json:
                                            fig_components = pio.from_json(components_plot_json)
                                            st.plotly_chart(fig_components, use_container_width=True)

                    st.markdown("---")
                    st.write("##### Cumulative Stream Performance")
                    st.caption("Gaps from duplicate days are interpolated (indicated by markers), while gaps from decreasing values are not.")
                    fig_cumulative = go.Figure()

                    # Plot the main line, which will still show a tooltip for all points
                    fig_cumulative.add_trace(go.Scatter(
                        x=interpolated_df.index, y=interpolated_df['value'],
                        mode='lines', name='Total Streams', connectgaps=True,
                        customdata=interpolated_df['hover_text'],
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Streams</b>: %{y:,.0f}%{customdata}<extra></extra>'
                    ))

                    # Add markers for only the points that were interpolated
                    marker_df = interpolated_df[interpolated_df['interpolated'] == True]
                    if not marker_df.empty:
                        # This trace for the orange circle markers will now have its own tooltip disabled
                        fig_cumulative.add_trace(go.Scatter(
                            x=marker_df.index, y=marker_df['value'],
                            mode='markers', name='Interpolated Point',
                            marker=dict(symbol='circle-open', size=8, color='orange', line=dict(width=2)),
                            hoverinfo='skip' # This line prevents the repetitive tooltip
                        ))
                    
                    y_max_cumulative = (interpolated_df['value'].dropna().max() if not interpolated_df.empty else 0)
                    
                    fig_cumulative.add_vline(x=datetime.combine(entry_date_obj, datetime.min.time()), line_width=2, line_dash="dash", line_color="red")
                    fig_cumulative.add_annotation(x=datetime.combine(entry_date_obj, datetime.min.time()), y=y_max_cumulative, yref="y", text="Playlist Add", showarrow=True, arrowhead=1, ax=0, ay=-40)
                    fig_cumulative.update_layout(yaxis_title="Total Streams", showlegend=False, hovermode='x unified')
                    st.plotly_chart(fig_cumulative, use_container_width=True)

                    st.markdown("---")
                    st.write("##### Calculated Daily Stream Impact")
                    analysis_results = st.session_state.get('global_spike_analysis_results', {}).get(song_uuid)
                    if analysis_results:
                        if analysis_results.get('error'): st.error(f"Spike analysis failed: {analysis_results['error']}")
                        elif not analysis_results.get('anomalies'): pass
                        else: st.info(f"Volatility analysis detected {len(analysis_results['anomalies'])} potential spike(s) marked on the chart below.")

                    if df_daily.empty:
                        st.info("Not enough data points to calculate daily stream changes.")
                        continue

                    fig_daily = go.Figure()
                    
                    # MODIFICATION: Added 'connectgaps=True' to create a continuous line.
                    fig_daily.add_trace(go.Scatter(
                        x=df_daily['date'], 
                        y=df_daily['value'], 
                        mode='lines', 
                        name='Daily Streams', 
                        connectgaps=True
                    ))
                    
                    y_max_daily = df_daily['value'].max() if not df_daily.empty else 0

                    if analysis_results and analysis_results.get('anomalies'):
                        last_spike_pos = {}
                        for anomaly in analysis_results['anomalies']:
                            anomaly_date = datetime.strptime(anomaly['date'], '%Y-%m-%d').date()
                            start_period = anomaly_date - timedelta(days=p_step)
                            center_period = anomaly_date - timedelta(days=p_step / 2)

                            y_level = y_max_daily
                            if last_spike_pos and (center_period - last_spike_pos.get('date', date.min)).days < (p_step * 1.5):
                                y_level = last_spike_pos.get('y', y_max_daily) * 0.85
                            last_spike_pos = {'date': center_period, 'y': y_level}

                            fig_daily.add_vrect(x0=start_period, x1=anomaly_date, fillcolor="rgba(0,255,255,0.1)", layer="below", line_width=1, line_color="cyan")
                            jump_size_formatted = f"+{int(anomaly['total_streams_on_day']):,}"
                            fig_daily.add_annotation(x=center_period, y=y_level, text=f"Spike: {jump_size_formatted}", showarrow=False, yshift=10, font=dict(color="cyan"), bgcolor="rgba(0, 50, 100, 0.6)")

                    fig_daily.add_vline(x=datetime.combine(entry_date_obj, datetime.min.time()), line_width=2, line_dash="dash", line_color="red")
                    fig_daily.add_annotation(x=datetime.combine(entry_date_obj, datetime.min.time()), y=y_max_daily, yref="y", text="Playlist Add", showarrow=True, arrowhead=1, ax=0, ay=-40)
                    fig_daily.update_layout(yaxis_title="Calculated Daily Streams", showlegend=False, hovermode='x unified')
                    st.plotly_chart(fig_daily, use_container_width=True)

# --- MODIFIED: Handles inconsistent date column/index ---
def display_playlist_audience_chart(audience_data: list, entry_date_str: str, song_release_date: str, chart_key: str):
    """
    Takes playlist audience data and plots both the original cumulative data and the
    calculated daily change, each with a marker for the song's entry date and release date.
    """
    if not audience_data:
        st.info("No audience data available for this playlist in the local database. Please use the Backfill page to update it.")
        return

    parsed_data = []
    for entry in audience_data:
        value = entry.get('value')
        if entry.get('date') and value is not None:
            parsed_data.append({'date': entry['date'], 'value': value})

    if not parsed_data:
        st.warning("Playlist audience data appears to be empty or in an unexpected format.")
        return

    st.markdown("---")

    def add_markers_to_fig(fig, df):
        if entry_date_str:
            entry_date_val = pd.to_datetime(entry_date_str)
            fig.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            if not df.empty:
                fig.add_annotation(x=entry_date_val, y=df['value'].max(), text="Song Entry", showarrow=True, arrowhead=2, ax=0, ay=-40)
        if song_release_date and not df.empty:
            release_date_dt = pd.to_datetime(song_release_date)
            
            # Handle if 'date' is a column or the index
            if 'date' in df.columns:
                chart_start_date = df['date'].min()
                chart_end_date = df['date'].max()
            else:
                chart_start_date = df.index.min()
                chart_end_date = df.index.max()

            if not pd.isna(chart_start_date) and not pd.isna(chart_end_date) and chart_start_date <= release_date_dt <= chart_end_date:
                fig.add_vline(x=release_date_dt, line_width=2, line_dash="solid", line_color="green")
                fig.add_annotation(x=release_date_dt, y=df['value'].max(), text="Song Release", showarrow=True, arrowhead=1, ax=20, ay=-60, font=dict(color="green"))

    df_cumulative = pd.DataFrame(parsed_data)
    df_cumulative['date'] = pd.to_datetime(df_cumulative['date'])
    sorted_df_cumulative = df_cumulative.sort_values(by='date').reset_index(drop=True)
    
    decreasing_points = sorted_df_cumulative['value'].diff() < 0
    # MODIFIED: Erroneous rows are dropped
    sorted_df_cumulative = sorted_df_cumulative[~decreasing_points]

    interpolated_df = pd.DataFrame()
    df_daily = pd.DataFrame()

    if not sorted_df_cumulative.empty and len(sorted_df_cumulative) > 1:
        # Temporary interpolation for daily calculation
        df_for_interp = sorted_df_cumulative.set_index('date')
        full_date_range = pd.date_range(start=df_for_interp.index.min(), end=df_for_interp.index.max(), freq='D')
        interpolated_df = df_for_interp.reindex(full_date_range).interpolate(method='time')
        daily_values = interpolated_df['value'].diff().clip(lower=0)
        df_daily = pd.DataFrame({'date': daily_values.index, 'value': daily_values.values})

    st.write("##### Original Data: Cumulative Follower Count")
    st.caption("This chart shows the raw, cumulative follower data. Gaps may appear where data was erroneous and has been removed.")
    fig_cumulative = go.Figure()

    # MODIFIED: Plotting from original data to show gaps
    fig_cumulative.add_trace(go.Scatter(
        x=sorted_df_cumulative['date'], y=sorted_df_cumulative['value'], 
        mode='lines', name='Total Followers', connectgaps=False,
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Followers</b>: %{y:,.0f}<extra></extra>'
    ))

    y_range_cumul = _get_optimal_y_range(sorted_df_cumulative, ['value'])
    fig_cumulative.update_layout(title="Cumulative Playlist Followers", yaxis_range=y_range_cumul, yaxis_tickformat=",.0f", showlegend=False)
    add_markers_to_fig(fig_cumulative, interpolated_df if not interpolated_df.empty else sorted_df_cumulative)
    st.plotly_chart(fig_cumulative, use_container_width=True, key=f"{chart_key}_cumulative")
    st.markdown("---")

    st.write("##### Normalized Data: Daily Follower Change")
    st.caption("This chart shows the calculated day-over-day change in followers, with gaps filled by linear interpolation.")

    if df_daily.empty:
        st.warning("Could not calculate daily changes from the source data.")
        return

    sorted_df_daily = df_daily.sort_values(by='date').iloc[1:].copy()

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=sorted_df_daily['date'], y=sorted_df_daily['value'], mode='lines', name='Daily Change', hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily Change</b>: %{y:,.0f}<extra></extra>'))

    y_range_daily = _get_optimal_y_range(sorted_df_daily, ['value'])
    fig_daily.update_layout(title="Daily Change in Playlist Followers", yaxis_range=y_range_daily, yaxis_tickformat=",.0f", showlegend=False)
    add_markers_to_fig(fig_daily, sorted_df_daily)
    st.plotly_chart(fig_daily, use_container_width=True, key=f"{chart_key}_daily")

# --- MODIFIED: Handles inconsistent song data structures ---
def display_playlist_details(db_manager, item):
    """Displays the detailed view for a single playlist entry."""
    playlist = item.get('playlist', {})
    playlist_name = playlist.get('name', 'N/A')
    playlist_uuid = playlist.get('uuid')

    # Handle inconsistent data structures by checking for song_uuid in multiple places
    song_uuid = item.get('song_uuid')
    if not song_uuid:
        song_uuid = item.get('song', {}).get('uuid')

    song_name = "N/A"
    song_release_date = None
    if song_uuid:
        # If a UUID is found, fetch the definitive song details to ensure data is correct
        song_metadata = get_song_details(db_manager, song_uuid)
        if song_metadata:
            meta_obj = song_metadata.get('object', song_metadata)
            song_name = meta_obj.get('name', 'N/A')
            song_release_date = meta_obj.get('releaseDate')

    if playlist_uuid:
        st.code(f"Playlist UUID: {playlist_uuid}", language=None)

    st.write(f"**Song:** {song_name}")
    position = item.get('position')
    peak_position = item.get('peakPosition')
    track_count = playlist.get('latestTrackCount')

    if position and track_count and track_count > 0:
        st.write(f"**Current Position:** {position} / {track_count} (Peak: {peak_position})")
        st.progress(max(0.0, min(1.0, (track_count - position + 1) / track_count)), text=f"Current Rank: #{position}")
    else:
        st.write(f"Position: {position or 'N/A'}")
        st.write(f"Peak Position: {peak_position or 'N/A'}")

    st.markdown("---")
    entry_date = item.get('entryDate')
    peak_date = item.get('peakPositionDate')
    subscribers = playlist.get('latestSubscriberCount', 0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Playlist Subscribers", f"{subscribers:,}" if subscribers else "N/A")
    col2.metric("Song Entry Date", str(entry_date)[:10] if entry_date else "N/A")
    col3.metric("Song Peak Date", str(peak_date)[:10] if peak_date else "N/A")
    
    st.markdown("---")
    st.subheader("Performance Analysis")
    st.write("Analyze the playlist's follower growth and the song's popularity around its entry date.")

    if song_uuid and playlist_uuid:
        graph_button_key = f"graph_btn_{playlist_uuid}_{song_uuid}"
        session_key = f"show_graph_{playlist_uuid}_{song_uuid}"

        if st.button("Show Performance Charts", key=graph_button_key, use_container_width=True):
            st.session_state[session_key] = not st.session_state.get(session_key, False)

        if st.session_state.get(session_key, False):
            if entry_date:
                try:
                    _, db_manager_local = get_clients()
                    
                    entry_date_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).date()
                    start_date_req = entry_date_dt - timedelta(days=90)
                    end_date_req = entry_date_dt + timedelta(days=90)

                    with st.spinner(f"Loading data for playlist '{playlist_name}' and song '{song_name}'..."):
                        # Playlist Audience
                        audience_data = get_playlist_audience_data(db_manager_local, playlist_uuid, start_date_req, end_date_req)
                        st.write(f"##### Follower Growth for **{playlist_name}**")
                        display_playlist_audience_chart(
                            audience_data,
                            entry_date,
                            song_release_date,
                            chart_key=f"chart_playlist_{playlist_uuid}_{song_uuid}"
                        )

                        # Song Popularity
                        st.markdown("---")
                        st.write(f"##### Popularity Score for **{song_name}**")
                        popularity_data = get_song_popularity_data(db_manager, song_uuid, "spotify", start_date_req, end_date_req)
                        display_timeseries_chart(
                            popularity_data,
                            title="", # Title is handled by the st.write above
                            chart_key=f"chart_song_pop_{playlist_uuid}_{song_uuid}"
                        )


                except (ValueError, TypeError) as e:
                    st.error(f"Could not process the request. Error: {e}")
            else:
                st.info("Cannot display performance graph without a song entry date.")


def display_song_details(db_manager, song_uuid):
    """
    Displays the detailed view for a single song, including performance and popularity charts.
    """
    api_client, _ = get_clients()
    song_metadata = get_song_details(db_manager, song_uuid)
    song_playlist_entries = get_playlists_for_song(db_manager, song_uuid)

    # --- Date Calculation & Update Logic ---
    start_date_filter = None
    end_date_filter = None
    can_update = False

    if song_playlist_entries:
        entry_dates = []
        for entry in song_playlist_entries:
            if entry_date_str := entry.get('entryDate'):
                try:
                    entry_dates.append(datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date())
                except ValueError:
                    continue
        
        if entry_dates:
            min_entry_date = min(entry_dates)
            max_entry_date = max(entry_dates)
            
            start_date_filter = min_entry_date - timedelta(days=90)
            end_date_filter = max_entry_date + timedelta(days=90)
            can_update = True

    # Use a default 1-year range if no playlist entries are found
    if not can_update:
        end_date_filter = date.today()
        start_date_filter = end_date_filter - timedelta(days=1095)

    # --- Display Metadata ---
    release_date_for_chart = None
    if song_metadata:
        meta_obj = song_metadata.get('object', song_metadata)
        release_date_for_chart = meta_obj.get('releaseDate')

        st.write(f"**Song UUID:** `{song_uuid}`")
        col1, col2 = st.columns(2)
        col1.metric("Release Date", str(release_date_for_chart)[:10] if release_date_for_chart else "N/A")

        duration_seconds = meta_obj.get('duration')
        if duration_seconds is not None:
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            col2.metric("Duration", f"{minutes}:{seconds:02d}")
        else:
            col2.metric("Duration", "N/A")

        genres_list = meta_obj.get('genres', [])
        if genres_list:
            genre_tags = set()
            for genre_info in genres_list:
                if root_genre := genre_info.get('root'):
                    genre_tags.add(root_genre.capitalize())
                for sub_genre in genre_info.get('sub', []):
                    genre_tags.add(sub_genre.capitalize())
            st.write(f"**Genres:** {', '.join(sorted(list(genre_tags)))}")
    else:
        st.info("Additional song metadata not found. Please update artist data on the homepage.")

    st.markdown("---")

    st.write("**Featured in playlists:**")
    if song_playlist_entries:
        for entry in song_playlist_entries:
            playlist_name = entry.get('playlist', {}).get('name', 'N/A')
            entry_date_str = str(entry.get('entryDate', 'N/A'))[:10]
            st.write(f"- {playlist_name} (Added on: {entry_date_str})")
    else:
        st.info("This song has not been detected in any tracked playlists.")

    st.markdown("---")
    
    st.write("**Update Performance Data**")
    st.caption(f"This will fetch audience and popularity data from **{start_date_filter.strftime('%Y-%m-%d')}** to **{end_date_filter.strftime('%Y-%m-%d')}**.")
    if st.button("Fetch and Store Song Data", use_container_width=True, type="primary"):
        with st.spinner("Fetching and updating song data..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_aud = executor.submit(api_client.get_song_streaming_audience, song_uuid, 'spotify', start_date_filter, end_date_filter)
                future_pop = executor.submit(api_client.get_song_popularity, song_uuid, 'spotify', start_date_filter, end_date_filter)
                
                aud_data = future_aud.result()
                pop_data = future_pop.result()

                data_found = False
                if aud_data and aud_data.get('items'):
                    db_manager.store_song_audience_data(song_uuid, aud_data)
                    data_found = True
                
                if pop_data and pop_data.get('items'):
                    db_manager.store_song_popularity_data(song_uuid, pop_data)
                    data_found = True

                if data_found:
                    get_full_song_data_from_db.clear()
                    get_song_audience_data.clear()
                    get_song_popularity_data.clear()
                    st.success("Track data updated successfully.")
                    st.rerun()
                else:
                    st.info("No new data was found for the specified range.")
    st.markdown("---")

    st.subheader("Analysis Tools")
    session_key = f"show_song_graph_{song_uuid}"
    if st.button("Show Performance Graph", key=f"song_graph_btn_{song_uuid}", use_container_width=True):
        st.session_state[session_key] = not st.session_state.get(session_key, False)

    if st.session_state.get(session_key, False):
        # Display Audience Chart
        song_full_data = get_full_song_data_from_db(db_manager, song_uuid)
        if song_full_data and song_full_data.get('history'):
            song_name_for_title = song_metadata.get('object', {}).get('name', 'this song') if song_metadata else "this song"
            playlist_annotations = []
            for entry in song_playlist_entries:
                playlist_info = entry.get('playlist', {})
                playlist_annotations.append({
                    'name': playlist_info.get('name'),
                    'uuid': playlist_info.get('uuid'),
                    'entryDate': entry.get('entryDate'),
                    'entrySubscribers': None
                })
            with st.spinner(f"Loading aggregated performance for '{song_name_for_title}'..."):
                display_full_song_streaming_chart(
                    db_manager,
                    song_full_data.get('history', []),
                    playlist_annotations,
                    chart_key=f"full_chart_{song_uuid}",
                    song_release_date=release_date_for_chart
                )
        else:
            st.warning("Aggregated streaming history not found. Please use the 'Update Performance Data' section above to fetch it.")

        # --- NEW: Display Popularity Chart ---
        st.markdown("---")
        with st.spinner("Loading popularity data..."):
            popularity_data = get_song_popularity_data(db_manager, song_uuid, "spotify", start_date_filter, end_date_filter)
        
        if popularity_data:
            display_timeseries_chart(
               popularity_data,
               title="Song Popularity Over Time",
               chart_key=f"chart_song_pop_{song_uuid}"
           )
        else:
            st.info("Popularity data not found for this period. Use the update button to fetch it.")

# --- MODIFIED: This function now handles inconsistent data structures ---
def display_by_song_view(db_manager, playlist_items):
    """
    Displays songs found on playlists in an interactive grid, similar to the main playlist view.
    """
    selected_uuid = st.session_state.get('selected_song_uuid')

    if selected_uuid:
        st.subheader("Song Performance Details")
        if st.button("⬅️ Back to all songs"):
            st.session_state.selected_song_uuid = None
            st.rerun()
        display_song_details(db_manager, selected_uuid)
        return

    if not playlist_items:
        st.info("No playlist entries were found for this artist. Please try updating the artist data on the Home page.")
        return

    with st.spinner("Aggregating song information..."):
        # Handle both {'song_uuid':...} and {'song':{'uuid':...}} structures
        song_uuids_on_playlists = set()
        for item in playlist_items:
            song_uuid = item.get('song_uuid') or item.get('song', {}).get('uuid')
            if song_uuid:
                song_uuids_on_playlists.add(song_uuid)


        if not song_uuids_on_playlists:
            st.info("Could not identify any songs from the playlist data. The data may be malformed.")
            return

        songs_cursor = db_manager.collections['songs'].find(
            {'song_uuid': {'$in': list(song_uuids_on_playlists)}},
            {'song_uuid': 1, 'object.name': 1, 'object.imageUrl': 1, 'name': 1, 'imageUrl': 1}
        )

        songs = {}
        for song_doc in songs_cursor:
            meta_obj = song_doc.get('object', song_doc)
            song_uuid = song_doc.get('song_uuid')
            if song_uuid:
                songs[song_uuid] = {
                    'uuid': song_uuid,
                    'name': meta_obj.get('name', 'N/A'),
                    'imageUrl': meta_obj.get('imageUrl'),
                }

    st.markdown("---")
    controls_cols = st.columns([2, 1, 1])
    with controls_cols[0]:
        search_term = st.text_input("Search by song name...", key="song_search")
    with controls_cols[1]:
        sort_key = st.selectbox("Sort by", ["Alphabetical"], key="song_sort_key")
    with controls_cols[2]:
        sort_order_label = st.radio("Order", ["A-Z", "Z-A"], key="song_sort_order", horizontal=True)

    display_songs = list(songs.values())
    if search_term:
        display_songs = [
            song for song in display_songs if search_term.lower() in song.get("name", "").lower()
        ]

    reverse_order = (sort_order_label == "Z-A")
    sort_lambda = lambda song: (song.get('name') or "").lower()
    sorted_songs = sorted(display_songs, key=sort_lambda, reverse=reverse_order)

    if not sorted_songs:
        st.info("No songs match your search criteria.")
    else:
        st.write(f"Displaying **{len(sorted_songs)}** unique song(s) found on playlists.")
        st.markdown("---")

        for song_chunk in grouper(sorted_songs, 4):
            cols = st.columns(4)
            for i, song_data in enumerate(song_chunk):
                if song_data:
                    with cols[i]:
                        song_uuid = song_data['uuid']
                        song_name = song_data['name']
                        image_url = song_data['imageUrl']

                        st.image(image_url or "https://i.imgur.com/3gMbdA5.png", use_container_width=True)
                        st.caption(f"**{song_name}**")
                        st.caption(f"UUID: `{song_uuid}`")

                        if st.button("Details", key=f"btn_song_{song_uuid}", use_container_width=True):
                            st.session_state.selected_song_uuid = song_uuid
                            st.rerun()

def grouper(iterable, n, fillvalue=None):
    "Helper to collect data into fixed-length chunks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

# --- MODIFIED: Handles inconsistent song data structures ---
def display_playlists(db_manager, playlist_items):
    """Displays playlist data, now with searching and sorting."""
    st.subheader("Featured On Playlists (Spotify)")

    if not playlist_items:
        st.info("No playlist entries found for this artist.")
        return

    view_choice = st.radio("View by:", ("Playlist", "Song"), horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if view_choice == "Song":
        display_by_song_view(db_manager, playlist_items)
        return

    selected_id = st.session_state.get('selected_playlist_entry_id')

    if selected_id:
        selected_item = next((item for item in playlist_items if str(item.get('_id')) == selected_id), None)
        if selected_item:
            playlist_name = selected_item.get('playlist', {}).get('name', 'Unknown Playlist')
            st.subheader(f"Details for: {playlist_name}")
            if st.button("⬅️ Back to all playlists"):
                st.session_state.selected_playlist_entry_id = None
                st.rerun()
            display_playlist_details(db_manager, selected_item)
        else:
            st.error("Could not find selected playlist details.")
            st.session_state.selected_playlist_entry_id = None
            st.rerun()
    else:
        controls_cols = st.columns([2, 1, 1])
        with controls_cols[0]:
            search_term = st.text_input("Search by playlist name...")
        with controls_cols[1]:
            sort_key = st.selectbox("Sort by", ["Subscriber Count", "Alphabetical"])
        with controls_cols[2]:
            sort_order_label = st.radio("Order", ["High to Low", "Low to High"])

        display_items = playlist_items
        if search_term:
            display_items = [
                item for item in display_items
                if search_term.lower() in item.get('playlist', {}).get('name', '').lower()
            ]

        reverse_order = (sort_order_label == "High to Low")
        if sort_key == "Alphabetical":
            sort_lambda = lambda item: item.get('playlist', {}).get('name', '').lower()
        else:
            sort_lambda = lambda item: item.get('playlist', {}).get('latestSubscriberCount', 0)

        display_items = sorted(display_items, key=sort_lambda, reverse=reverse_order)

        if not display_items:
            st.info("No playlists match your search criteria.")

        for item_chunk in grouper(display_items, 4):
            cols = st.columns(4)
            for i, item in enumerate(item_chunk):
                if item:
                    with cols[i]:
                        playlist = item.get('playlist', {})
                        playlist_uuid = playlist.get('uuid')
                        playlist_name = playlist.get('name', 'N/A')
                        image_url = playlist.get('imageUrl')

                        doc_id = str(item.get('_id'))
                        button_key = f"btn_playlist_{doc_id}"

                        if image_url:
                            st.image(image_url, use_container_width=True)
                        else:
                            st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)

                        st.caption(f"**{playlist_name}**")
                        
                        # Handle inconsistent song name in grid view
                        song_name_in_grid = item.get('song', {}).get('name')
                        if song_name_in_grid:
                            st.write(f"Song: *{song_name_in_grid}*")
                        else:
                            # If name isn't embedded, check for a song_uuid to imply details exist
                            song_uuid_in_grid = item.get('song_uuid') or item.get('song', {}).get('uuid')
                            if song_uuid_in_grid:
                                st.write("Song: *(See Details)*")
                            else:
                                st.write("Song: N/A")

                        if playlist_uuid:
                            st.caption(f"UUID: `{playlist_uuid}`")

                        if st.button("Details", key=button_key, use_container_width=True):
                            st.session_state.selected_playlist_entry_id = doc_id
                            st.rerun()


def display_album_and_tracks(db_manager, album_data, tracklist_data, audience_data=None, start_date=None, end_date=None):
    """Displays album and tracklist data from the database."""
    if not album_data:
        st.warning("No album data available."); return

    unified_meta = album_data.get('album_metadata', {}).copy()
    if 'object' in unified_meta and isinstance(unified_meta['object'], dict):
        unified_meta.update(unified_meta['object'])

    album_uuid = unified_meta.get('album_uuid', unified_meta.get('uuid'))

    col1, col2, col3 = st.columns(3)
    col1.metric("Release Date", str(unified_meta.get('releaseDate', 'N/A'))[:10])
    col2.metric("Tracks", unified_meta.get('totalTracks', 'N/A'))
    col3.metric("Distributor", unified_meta.get('distributor', 'N/A'))
    st.write(f"**UPC:** `{unified_meta.get('upc', 'N/A')}`")
    st.markdown("---")

    if audience_data is not None:
        st.subheader("Album Audience")
        with st.expander("Show Raw Album Audience Data"):
            st.json(audience_data)
        display_timeseries_chart(audience_data, title="Album Audience Over Time")
        st.markdown("---")

    st.write("**Tracklist:**")
    if not tracklist_data:
        st.info("Tracklist not available."); return

    items_list = tracklist_data.get('object', tracklist_data).get('items', [])
    if not items_list:
        st.info("Tracklist is empty."); return

    for i, item in enumerate(items_list):
        song = item.get('song', {})
        song_uuid = song.get('uuid')
        song_name = song.get('name', 'Unknown Track')
        track_col, button_col = st.columns([4, 1])
        track_col.write(f"**{item.get('number', '#')}.** {song_name}")

        if song_uuid:
            button_key = f"btn_{album_uuid or 'unknown'}_{song_uuid}_{i}"
            if button_col.button("Details", key=button_key, use_container_width=True):
                 session_key = f"show_{album_uuid}_{song_uuid}"
                 st.session_state[session_key] = not st.session_state.get(session_key, False)

            if st.session_state.get(f"show_{album_uuid}_{song_uuid}", False):
                with st.container(border=True):
                    song_metadata = get_song_details(db_manager, song_uuid)
                    st.write(f"**Song UUID:** `{song_uuid}`")
                    if song_metadata:
                        meta_obj = song_metadata.get('object', song_metadata)
                        st.write(f"##### Details for **{meta_obj.get('name', song_name)}**")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Duration", f"{meta_obj.get('duration', 'N/A')}s")
                        rel_date = meta_obj.get('releaseDate', 'N/A')
                        sc2.metric("Release Date", str(rel_date)[:10])
                        genres = meta_obj.get('genres', [])
                        root_genre = genres[0].get('root', 'N/A') if genres else 'N/A'
                        sc3.metric("Genre", root_genre)
                        st.write(f"**Composers:** {', '.join(meta_obj.get('composers', ['N/A']))}")
                        st.write(f"**Producers:** {', '.join(meta_obj.get('producers', ['N/A']))}")
                    else:
                        st.warning(f"Full metadata for '{song_name}' not found.")

                    if start_date and end_date:
                        st.markdown("---")
                        # Display Song Audience Chart
                        song_aud_data = get_song_audience_data(db_manager, song_uuid, "spotify", start_date, end_date)
                        display_timeseries_chart(
                            song_aud_data,
                            title="Song Audience Over Time",
                            chart_key=f"aud_chart_{song_uuid}"
                        )
                        # --- ADDED: Display Song Popularity Chart ---
                        st.markdown("---")
                        song_pop_data = get_song_popularity_data(db_manager, song_uuid, "spotify", start_date, end_date)
                        display_timeseries_chart(
                            song_pop_data,
                            title="Song Popularity Over Time",
                            chart_key=f"pop_chart_{song_uuid}"
                        )


def display_timeseries_chart(chart_data, title="", chart_key=None, show_daily=False):
    """
    Displays a generic time-series chart, ensuring data is sorted and
    gaps are not connected. If show_daily=True, computes and plots daily changes on the left y-axis
    and cumulative on the right y-axis.
    """
    st.subheader(title)
    if not chart_data:
        st.info("No time-series data available to display.")
        return

    parsed_data = []
    data_source = chart_data.get('history') if isinstance(chart_data, dict) and 'history' in chart_data else chart_data

    for entry in data_source:
        date_val = entry.get('date')
        if not date_val: continue

        data_point = {'date': pd.to_datetime(date_val)}

        value = None
        if 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
            value = entry['plots'][0].get('value')
        elif 'value' in entry:
            value = entry.get('value')
        elif 'listenerCount' in entry:
            value = entry.get('listenerCount')
        elif 'followerCount' in entry:
            value = entry.get('followerCount')

        if value is not None:
            data_point['value'] = value
            parsed_data.append(data_point)

    if not parsed_data:
        st.warning("Could not parse any valid data points to plot.")
        return

    df_cum = pd.DataFrame(parsed_data)
    df_cum.sort_values(by='date', inplace=True)
    df_cum.set_index('date', inplace=True)

    if df_cum.empty or 'value' not in df_cum.columns:
        st.warning("Could not find any valid data to plot after parsing.")
        return

    if show_daily:
        if len(df_cum) < 2:
            st.info("Not enough data to compute daily changes. Displaying cumulative only.")
            show_daily = False  # Fallback to cumulative only
        else:
            daily_values = df_cum['value'].diff().clip(lower=0)
            df_daily = pd.DataFrame(daily_values, columns=['value'], index=df_cum.index)[1:]  # Drop first NaN

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Daily on primary (left) y-axis
            fig.add_trace(go.Scatter(
                x=df_daily.index, y=df_daily['value'], mode='lines', name='Daily Value', connectgaps=False,
                line=dict(color='#1f77b4'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily</b>: %{y:,.0f}<extra></extra>'
            ), secondary_y=False)

            # Cumulative on secondary (right) y-axis
            fig.add_trace(go.Scatter(
                x=df_cum.index, y=df_cum['value'], mode='lines', name='Cumulative Value', connectgaps=False,
                line=dict(color='#ff7f0e'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Cumulative</b>: %{y:,.0f}<extra></extra>'
            ), secondary_y=True)

            y_range_daily = _get_optimal_y_range(df_daily, ['value'])
            y_range_cum = _get_optimal_y_range(df_cum, ['value'])

            fig.update_layout(
                yaxis_range=y_range_daily if y_range_daily else None,
                yaxis2_range=y_range_cum if y_range_cum else None,
                yaxis_title="Daily Value",
                yaxis2_title="Cumulative Value",
                yaxis_tickformat=",.0f",
                yaxis2_tickformat=",.0f",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            return

    # Fallback to original single-axis plot if not show_daily
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_cum.index, y=df_cum['value'], mode='lines', name='Value', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.0f}<extra></extra>'))

    y_range = _get_optimal_y_range(df_cum, ['value'])
    fig.update_layout(
        title="",
        yaxis_range=y_range,
        yaxis_tickformat=",.0f",
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def display_audience_chart(audience_data):
    """Displays audience chart data."""
    st.subheader("Audience on Spotify")
    if not audience_data:
        st.info("No audience data available for the selected period.")
        return

    df = pd.DataFrame(audience_data).rename(columns={'followerCount': 'Followers'})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['Followers'], mode='lines', name='Followers', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Followers</b>: %{y:,.0f}<extra></extra>'))
    y_range = _get_optimal_y_range(df, ['Followers'])
    fig.update_layout(title="Followers Over Time", yaxis_range=y_range, yaxis_tickformat=",.0f", showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def display_popularity_chart(popularity_data):
    """Displays popularity chart data."""
    st.subheader("Popularity on Spotify")
    if not popularity_data:
        st.info("No popularity data available for the selected period.")
        return

    df = pd.DataFrame(popularity_data).rename(columns={'value': 'Popularity Score'})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['Popularity Score'], mode='lines', name='Score', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Score</b>: %{y:,.0f}<extra></extra>'))
    y_range = _get_optimal_y_range(df, ['Popularity Score'])
    fig.update_layout(title="Popularity Score Over Time", yaxis_range=y_range, yaxis_tickformat=",.0f", showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def display_streaming_audience_chart(db_manager, streaming_data, spikes=None, events=None):
    """MODIFIED: Displays streaming audience data and optional spikes and events."""
    if spikes is None:
        spikes = st.session_state.get('streaming_audience_spikes')
    
    st.subheader("Streaming Audience")
    if not streaming_data:
        st.info("No streaming audience data available for the selected period.")
        return
    df = pd.DataFrame(streaming_data).rename(columns={'value': 'Streams'})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['Streams'], mode='lines', name='Streams', connectgaps=False, line=dict(color='#2ca02c'), hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,.0f}<extra></extra>'))
    
    if spikes is not None and not spikes.empty:
        fig.add_trace(go.Scatter(
            x=spikes.index, y=spikes['streams'], mode='markers', name='Detected Spikes',
            marker=dict(color='red', size=8, symbol='x'),
            hoverinfo='text',
            hovertext=[f"Spike: {s:,.0f} streams<br>Date: {d.strftime('%Y-%m-%d')}<br>Type: {t}" for d, s, t in zip(spikes.index, spikes['streams'], spikes['type'])]
        ))

    # This block adds annotations for artist events.
    if events:
        y_max = df['Streams'].max() if not df.empty else 0
        chart_start_date = df['date'].min()
        chart_end_date = df['date'].max()

        filtered_events = [
            e for e in events if 'date' in e and pd.notna(e['date']) and
            chart_start_date <= pd.to_datetime(e['date']) <= chart_end_date
        ]
        
        sorted_events = sorted(filtered_events, key=lambda x: pd.to_datetime(x.get('date')))
        
        # Fetch venue and festival metadata
        venue_uuids = set()
        festival_uuids = set()
        for event in sorted_events:
            if event.get('type') == 'festival' and (festival := event.get('festival')) and isinstance(festival, dict):
                if uuid := festival.get('uuid'):
                    festival_uuids.add(uuid)
            elif (venue := event.get('venue')) and isinstance(venue, dict):
                if uuid := venue.get('uuid'):
                    venue_uuids.add(uuid)

        venues_dict = {}
        if venue_uuids:
            venues_cursor = db_manager.collections['venues'].find({'uuid': {'$in': list(venue_uuids)}})
            for venue_doc in venues_cursor:
                venues_dict[venue_doc['uuid']] = venue_doc

        festivals_dict = {}
        if festival_uuids:
            festivals_cursor = db_manager.collections['festivals'].find({'uuid': {'$in': list(festival_uuids)}})
            for festival_doc in festivals_cursor:
                festivals_dict[festival_doc['uuid']] = festival_doc

        def get_capacity(event):
            if event.get('type') == 'festival':
                if (festival := event.get('festival')) and isinstance(festival, dict):
                    uuid = festival.get('uuid')
                    meta = festivals_dict.get(uuid, {})
                    cap = meta.get('capacity')
                    return f"{int(cap):,}" if cap is not None else 'N/A'
            else:
                if (venue := event.get('venue')) and isinstance(venue, dict):
                    uuid = venue.get('uuid')
                    meta = venues_dict.get(uuid, {})
                    cap = meta.get('capacity')
                    return f"{int(cap):,}" if cap is not None else 'N/A'
            return 'N/A'
        
        last_event_date = None
        y_shift_offset = 20 # Initial vertical offset for the annotation text

        for event in sorted_events:
            try:
                event_date_dt = pd.to_datetime(event['date'])
                event_date_val = event_date_dt.date()

                # Prevents annotation text from overlapping if events are close together
                if last_event_date and (event_date_val - last_event_date).days < 45:
                    y_shift_offset += 40
                else:
                    y_shift_offset = 20 # Reset offset

                fig.add_vline(x=event_date_dt, line_width=1.5, line_dash="dot", line_color="rgba(171, 99, 250, 1)")

                event_name = event.get('name')
                venue_name = event.get('venue', {}).get('name')
                
                # Use event name if available, otherwise fallback to venue name
                display_name = event_name if event_name else venue_name if venue_name else "Event"

                # Wrap the name for better display
                wrapped_name = '<br>'.join(textwrap.wrap(display_name, width=20))
                
                capacity = get_capacity(event)

                annotation_text = f"{wrapped_name}<br>Capacity: {capacity}"
                hover_text = f"<b>Event:</b> {display_name}<br><b>Date:</b> {event_date_val.strftime('%Y-%m-%d')}<br><b>Capacity:</b> {capacity}"

                fig.add_annotation(
                    x=event_date_dt, y=y_max, text=annotation_text,
                    showarrow=True, arrowhead=4, arrowwidth=1.5, arrowcolor="rgba(171, 99, 250, 0.8)",
                    yshift=y_shift_offset, ax=0, ay=-40,
                    font=dict(color="white"), bgcolor="rgba(171, 99, 250, 0.8)",
                    borderpad=3, hovertext=hover_text
                )
                last_event_date = event_date_val
            except (ValueError, KeyError) as e:
                st.warning(f"Could not process an event annotation due to an error: {e}")
                continue

    y_range = _get_optimal_y_range(df, ['Streams'])
    fig.update_layout(yaxis_range=y_range, yaxis_tickformat=",.0f", showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    


def display_top_countries_trend_chart(df_top_countries, anomalies: list = None, spikes: dict = None):
    """
    Takes a DataFrame of daily stream data for the top countries and
    plots it as a multi-line trend chart. Can now overlay anomalies from divergence
    analysis and spikes from systematic peak detection.
    """
    if df_top_countries.empty:
        st.warning("No data available to plot.")
        return

    df_pivot = df_top_countries.pivot_table(
        index='date',
        columns='country',
        values='streams',
        aggfunc='sum'
    ).fillna(0)

    fig = px.line(
        df_pivot,
        x=df_pivot.index,
        y=df_pivot.columns,
        title="Streaming Trends for Top 10 Countries",
        labels={'value': 'Daily Streams', 'date': 'Date', 'country': 'Country'}
    )

    # This section handles overlaying anomalous periods from the divergence analysis
    if anomalies:
        for anomaly in anomalies:
            fig.add_vrect(
                x0=anomaly.get('start'),
                x1=anomaly.get('end'),
                fillcolor="rgba(255, 0, 0, 0.2)",
                layer="below",
                line_width=0,
                annotation_text=f"Divergence: {anomaly.get('country')}",
                annotation_position="top left"
            )

    # This new section handles overlaying the detected spikes
    if spikes:
        for country, spike_df in spikes.items():
            if not spike_df.empty:
                fig.add_trace(go.Scatter(
                    x=spike_df.index,
                    y=spike_df['streams'],
                    mode='markers',
                    name=f'{country} Spikes',
                    marker=dict(symbol='x', color='red', size=10, line=dict(width=2)),
                    hoverinfo='text',
                    # Create a custom hover text for each spike marker
                    hovertext=[f"Spike: {s:,.0f} streams<br>Date: {d.strftime('%Y-%m-%d')}<br>Country: {country}" for d, s in zip(spike_df.index, spike_df['streams'])]
                ))

    fig.update_layout(
        hovermode="x unified",
        yaxis_tickformat=",.0f",
        legend_title_text='Countries'
    )

    st.plotly_chart(fig, use_container_width=True)
    
def display_country_statistics_charts(df: pd.DataFrame, std_threshold: float, window_size: int):
    """
    Displays individual charts for each country showing the streaming data,
    its rolling average, and a shaded band for the rolling standard deviation.

    Args:
        df (pd.DataFrame): DataFrame containing 'date', 'country', and 'streams'.
        std_threshold (float): The multiplier for the standard deviation band.
        window_size (int): The window size for the rolling calculations.
    """
    countries = df['country'].unique()
    
    st.caption(f"Showing daily streams with a {window_size}-day rolling average and a {std_threshold:.1f}x standard deviation band for each country.")

    for country in countries:
        st.write(f"**Analysis for {country}**")
        country_df = df[df['country'] == country].set_index('date').sort_index()
        
        if country_df.empty or len(country_df) < window_size:
            st.info(f"Not enough data for {country} to calculate rolling statistics with a {window_size}-day window.")
            continue
            
        # Calculate rolling statistics
        rolling_mean = country_df['streams'].rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = country_df['streams'].rolling(window=window_size, center=True, min_periods=1).std()
        
        # Calculate the upper and lower bands using the threshold
        upper_band = rolling_mean + (std_threshold * rolling_std)
        lower_band = (rolling_mean - (std_threshold * rolling_std)).clip(lower=0)

        fig = go.Figure()
        
        # Original streams line
        fig.add_trace(go.Scatter(
            x=country_df.index, y=country_df['streams'], mode='lines', name='Daily Streams', line=dict(color='#1f77b4')
        ))

        # Standard deviation band
        fig.add_trace(go.Scatter(
            x=rolling_mean.index, y=upper_band, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=rolling_mean.index, y=lower_band, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', name=f'Mean +/- {std_threshold:.1f} SD', hoverinfo='skip'
        ))

        # Rolling average line
        fig.add_trace(go.Scatter(
            x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Average', line=dict(color='rgba(255, 127, 14, 0.8)', dash='dash')
        ))

        fig.update_layout(
            yaxis_title="Daily Streams", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"stats_chart_{country}")# --- NEW FUNCTION for the Tracks page grid view ---
def display_tracks_grid(songs: List[Dict[str, Any]]):
    """Displays a searchable and sortable grid of all artist songs."""
    st.subheader("All Artist Songs")
    st.markdown("---")
    
    controls_cols = st.columns([2, 1, 1])
    with controls_cols[0]:
        search_term = st.text_input("Search by song name...", key="track_search")
    with controls_cols[1]:
        sort_key = st.selectbox("Sort by", ["Alphabetical", "Release Date"], key="track_sort_key")
    with controls_cols[2]:
        sort_order_label = st.radio("Order", ["A-Z / Newest", "Z-A / Oldest"], key="track_sort_order", horizontal=True)

    display_songs = list(songs)
    if search_term:
        display_songs = [s for s in display_songs if search_term.lower() in s.get("name", "").lower()]

    reverse_order = (sort_order_label == "Z-A / Oldest")
    
    # --- MODIFICATION START ---
    # This logic is updated to ensure a string is always returned for comparison.
    if sort_key == "Alphabetical":
        # Provides an empty string for None names, which is safe.
        sort_lambda = lambda s: (s.get("name") or "").lower()
    else:  # Release Date
        # Uses 'or' to provide a fallback string if the release date is None.
        # This prevents the TypeError.
        sort_lambda = lambda s: s.get("releaseDate") or "1900-01-01"
    # --- MODIFICATION END ---

    # Sort songs ensuring None values don't crash the sort
    sorted_songs = sorted(
        [s for s in display_songs if s.get('uuid')], 
        key=sort_lambda, 
        reverse=reverse_order
    )

    if not sorted_songs:
        st.info("No songs match your search criteria.")
        return

    st.write(f"Displaying **{len(sorted_songs)}** song(s).")
    st.markdown("---")

    for song_chunk in grouper(sorted_songs, 4):
        cols = st.columns(4)
        for i, song_data in enumerate(song_chunk):
            if song_data:
                with cols[i]:
                    song_uuid = song_data['uuid']
                    song_name = song_data['name']
                    image_url = song_data.get('imageUrl')

                    st.image(image_url or "https://i.imgur.com/3gMbdA5.png", use_container_width=True)
                    st.caption(f"**{song_name}**")
                    # Use a fallback for displaying the release date as well
                    release_display = song_data.get('releaseDate') or 'N/A'
                    st.caption(f"Released: {release_display}")
                    
                    if st.button("Details", key=f"btn_track_{song_uuid}", use_container_width=True):
                        st.session_state.selected_track_uuid = song_uuid
                        st.rerun()



# Integrate debug into existing function
def display_track_details_page(api_client, db_manager, song_uuid: str):
    """Displays the comprehensive detail view for a single track."""
    if st.button("⬅️ Back to all tracks"):
        st.session_state.selected_track_uuid = None
        st.rerun()

    song_details = get_song_details(db_manager, song_uuid)
    if not song_details:
        st.error(f"Could not load details for song UUID: {song_uuid}")
        return

    meta_obj = song_details.get('object', song_details)
    song_name = meta_obj.get('name', 'Unknown Song')
    st.subheader(f"Details for: {song_name}")

    col1, col2 = st.columns([1, 4])
    with col1:
        if image_url := meta_obj.get("imageUrl"):
            st.image(image_url, use_container_width=True)
    with col2:
        m_col1, m_col2 = st.columns(2)
        release_date = meta_obj.get('releaseDate', 'N/A')
        m_col1.metric("Release Date", str(release_date)[:10])
        duration_seconds = meta_obj.get('duration')
        if duration_seconds is not None:
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            m_col2.metric("Duration", f"{minutes}:{seconds:02d}")
        st.write(f"ISRC: {meta_obj.get('isrc')}")
        st.caption(f"UUID: {song_uuid}")

    st.markdown("---")
    st.subheader("Performance Data")

    start_date_filter = st.date_input("Chart Start Date", datetime(2023, 1, 1), key=f"start_track_{song_uuid}")
    end_date_filter = st.date_input("Chart End Date", datetime.now(), key=f"end_track_{song_uuid}")

    if 'show_aud_success' not in st.session_state:
        st.session_state.show_aud_success = False
    if st.session_state.show_aud_success:
        st.success("Audience data updated successfully.")
        st.session_state.show_aud_success = False

    if 'show_pop_success' not in st.session_state:
        st.session_state.show_pop_success = False
    if st.session_state.show_pop_success:
        st.success("Popularity data updated successfully.")
        st.session_state.show_pop_success = False

    if st.button("Fetch and Update Data", use_container_width=True, type="primary"):
        with st.spinner("Fetching and updating Audience and Popularity data..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_aud = executor.submit(api_client.get_song_streaming_audience, song_uuid, 'spotify', start_date_filter, end_date_filter)
                future_pop = executor.submit(api_client.get_song_popularity, song_uuid, 'spotify', start_date_filter, end_date_filter)
                aud_data = future_aud.result()
                pop_data = future_pop.result()
                # Process and store audience data
                try:
                    if aud_data and not aud_data.get('error') and aud_data.get('items'):
                        raw_history = []
                        for item in aud_data['items']:
                            if 'date' in item:
                                plots = item.get('plots', [])
                                if plots:
                                    max_val = max(p.get('value', 0) for p in plots)
                                    raw_history.append({'date': item['date'], 'value': max_val})
                        # Fetch existing for merge
                        query_filter = {'song_uuid': song_uuid, 'platform': 'spotify'}
                        existing = db_manager.collections['song_audience'].find_one(query_filter)
                        history = existing.get('history', []) if existing else []
                        all_items = history + raw_history
                        max_values = {}
                        for item in all_items:
                            if 'date' in item:
                                date_str = item['date']
                                value = item.get('value', 0)
                                if date_str in max_values:
                                    max_values[date_str] = max(max_values[date_str], value)
                                else:
                                    max_values[date_str] = value
                        sorted_dates = sorted(
                            max_values.keys(),
                            key=lambda d: datetime.fromisoformat(d.replace('Z', '+00:00'))
                        )
                        cleaned_items = [{'date': d, 'value': max_values[d]} for d in sorted_dates]
                        if cleaned_items:
                            from analysis_tools import adjust_cumulative_history
                            adjusted_items = adjust_cumulative_history(cleaned_items)
                        else:
                            st.warning("No cleaned items to adjust, skipping storage.")
                        db_manager.store_song_audience_data(song_uuid, {'history': adjusted_items, 'platform': 'spotify'})
                        get_song_audience_data.clear()
                        st.session_state.show_aud_success = True
                    else:
                        st.warning("No valid audience data received from API.")
                except Exception as e:
                    st.error(f"Failed to update audience data: {str(e)}")
                # Process and store popularity data
                try:
                    if pop_data and not pop_data.get('error') and pop_data.get('items'):
                        db_manager.store_song_popularity_data(song_uuid, pop_data)
                        get_song_popularity_data.clear()
                        st.session_state.show_pop_success = True
                    else:
                        st.info("No new popularity data to update.")
                except Exception as e:
                    st.error(f"Failed to update popularity data: {str(e)}")
            if st.button("Refresh Page to See Updated Charts"):
                st.rerun()
            else:
                st.info("Data updated! Click the button above to refresh and see charts.")

    aud_data = get_song_audience_data(db_manager, song_uuid, "spotify", start_date_filter, end_date_filter)
    pop_data = get_song_popularity_data(db_manager, song_uuid, "spotify", start_date_filter, end_date_filter)

    def parse_timeseries_data(raw_data):
        source_list = raw_data.get('history') if isinstance(raw_data, dict) else raw_data
        if not source_list:
            return []
        parsed_list = []
        for entry in source_list:
            timestamp = entry.get('timestamp')
            value = entry.get('cumulative_streams')
            if timestamp and value is not None:
                parsed_list.append({'date': timestamp, 'value': value})  # Use 'date' for consistency with chart logic
        return parsed_list

    parsed_aud = parse_timeseries_data(aud_data)
    if parsed_aud:
        df_aud_cum = pd.DataFrame(parsed_aud)
        df_aud_cum['date'] = pd.to_datetime(df_aud_cum['date'], format='ISO8601')
        df_aud_cum = df_aud_cum.sort_values('date').set_index('date')
        adjusted_times = df_aud_cum.index.tolist()
        cum_streams = df_aud_cum['value'].tolist()
        n = len(adjusted_times)
        mid_times = []
        rates = []
        if n >= 2:
            mid_times = [adjusted_times[i] + (adjusted_times[i+1] - adjusted_times[i]) / 2 for i in range(n - 1)]
            rates = []
            for i in range(n - 1):
                delta_t = (adjusted_times[i + 1] - adjusted_times[i]).total_seconds() / 86400.0
                delta_c = cum_streams[i + 1] - cum_streams[i]
                rate = delta_c / delta_t if delta_t > 0 else 0
                rates.append(rate)
    else:
        df_aud_cum = pd.DataFrame()
        mid_times = []
        rates = []

    parsed_pop = parse_timeseries_data(pop_data)
    if parsed_pop:
        df_pop = pd.DataFrame(parsed_pop)
        df_pop['date'] = pd.to_datetime(df_pop['date'], format='ISO8601')
        df_pop = df_pop.set_index('date').sort_index()
    else:
        df_pop = pd.DataFrame()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Song Streaming Numbers Over Time")
        if not df_aud_cum.empty:
            fig_aud = go.Figure()
            fig_aud.add_trace(
                go.Scatter(x=df_aud_cum.index, y=df_aud_cum['value'], mode='lines', name='Cumulative Streams', line=dict(color='#ff7f0e'))
            )
            y_range_cum = _get_optimal_y_range(df_aud_cum, ['value'])
            fig_aud.update_layout(
                yaxis_range=y_range_cum,
                yaxis_title="Cumulative Streams",
                yaxis_tickformat=",.0f",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_aud, use_container_width=True, key=f"track_aud_{song_uuid}")
        else:
            st.info("No audience data to display.")
    with c2:
        st.subheader("Song Popularity Over Time")
        if not df_pop.empty:
            fig_pop = go.Figure()
            fig_pop.add_trace(
                go.Scatter(x=df_pop.index, y=df_pop['value'], mode='lines', name='Popularity Score', line=dict(color='#1f77b4'))
            )
            y_range_pop = _get_optimal_y_range(df_pop, ['value'])
            fig_pop.update_layout(
                yaxis_range=y_range_pop,
                yaxis_title="Popularity Score",
                yaxis_tickformat=",.0f",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            st.plotly_chart(fig_pop, use_container_width=True, key=f"track_pop_{song_uuid}")
        else:
            st.info("No popularity data to display.")

    st.markdown("---")
    st.subheader("Daily Song Streams Over Time")
    if rates:
        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(x=mid_times, y=rates, mode='lines+markers', marker=dict(size=3), name='Daily Streams', line=dict(color='#2ca02c'))
        )
        temp_df = pd.DataFrame({'daily_streams': rates})
        y_range_daily = _get_optimal_y_range(temp_df, ['daily_streams'])
        fig_daily.update_layout(
            yaxis_range=y_range_daily,
            yaxis_title="Daily Rate (streams/day)",
            yaxis_tickformat=",.0f",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_daily, use_container_width=True, key=f"track_daily_{song_uuid}")
    else:
        st.info("No daily streams data to display.")

    st.markdown("---")
    display_retention_chart(
        audience_data=aud_data,
        popularity_data=pop_data,
        chart_key=f"track_retention_{song_uuid}"
    )

    st.markdown("---")
    st.subheader("Playlist Placements")
    playlist_entries = get_playlists_for_song(db_manager, song_uuid)
    if not playlist_entries:
        st.info("This song has not been found on any playlists in the database.")
    else:
        st.write(f"Found this song on {len(playlist_entries)} playlist(s):")
        for entry in playlist_entries:
            playlist_info = entry.get('playlist', {})
            entry_date = entry.get('entryDate', 'N/A')
            with st.container(border=True):
                st.write(f"{playlist_info.get('name', 'N/A')}")
                st.caption(f"Added on: {str(entry_date)[:10]}")
                st.write(f"Subscribers on entry: {playlist_info.get('latestSubscriberCount', 'N/A'):,}")


def display_retention_chart(audience_data: list, popularity_data: list, chart_key: str):
    """
    Calculates and displays a retention rate chart based on audience and popularity data.
    Retention Rate = (Popularity Score / Audience Count)
    """
    st.subheader("Audience Engagement Rate (Popularity / Audience)")

    # --- MODIFIED SECTION ---
    # Helper function to correctly parse nested data structures
    def parse_timeseries_data(raw_data):
        source_list = raw_data.get('history') if isinstance(raw_data, dict) else raw_data
        if not source_list:
            return []

        parsed_list = []
        for entry in source_list:
            date_val = entry.get('date')
            value = None
            # Handle the nested 'plots' structure for popularity/audience
            if 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
                value = entry['plots'][0].get('value')
            # Handle simpler structures
            elif 'value' in entry:
                value = entry.get('value')

            if date_val and value is not None:
                parsed_list.append({'date': date_val, 'value': value})
        return parsed_list

    # Parse both audience and popularity data using the helper
    parsed_aud_data = parse_timeseries_data(audience_data)
    parsed_pop_data = parse_timeseries_data(popularity_data)

    if not parsed_aud_data or not parsed_pop_data:
        st.info("Both Audience and Popularity data are required to calculate the engagement rate.")
        return

    # Create DataFrames from the correctly parsed lists
    df_aud = pd.DataFrame(parsed_aud_data)
    df_pop = pd.DataFrame(parsed_pop_data)
    # --- END OF MODIFIED SECTION ---

    # The rest of the function remains the same
    df_aud['date'] = pd.to_datetime(df_aud['date'])
    df_aud.set_index('date', inplace=True)
    df_aud.rename(columns={'value': 'audience'}, inplace=True)

    df_pop['date'] = pd.to_datetime(df_pop['date'])
    df_pop.set_index('date', inplace=True)
    df_pop.rename(columns={'value': 'popularity'}, inplace=True)

    df_merged = pd.merge(df_aud, df_pop, left_index=True, right_index=True, how='inner')
    df_merged.sort_index(inplace=True)
    if df_merged.empty:
        st.warning("Could not find any overlapping dates between audience and popularity data.")
        return

    df_merged['retention_rate'] = (df_merged['popularity'] / df_merged['audience']).replace([np.inf, -np.inf], 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_merged.index,
        y=df_merged['retention_rate'],
        mode='lines',
        name='Engagement Rate',
        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Rate</b>: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title="",
        yaxis_tickformat=".2%",
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
def display_single_timeseries_stats_chart(df: pd.DataFrame, std_threshold: float, window_size: int):
    """
    Displays a single chart showing time-series data, its rolling average, 
    and a shaded band for the rolling standard deviation.

    Args:
        df (pd.DataFrame): DataFrame containing the time-series data in a 'streams' column.
        std_threshold (float): The multiplier for the standard deviation band.
        window_size (int): The window size for the rolling calculations.
    """
    st.caption(f"Showing daily streams with a {window_size}-day rolling average and a {std_threshold:.1f}x standard deviation band.")

    if df.empty or len(df) < window_size:
        st.info(f"Not enough data to calculate rolling statistics with a {window_size}-day window.")
        return
        
    # Calculate rolling statistics
    rolling_mean = df['streams'].rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = df['streams'].rolling(window=window_size, center=True, min_periods=1).std()
    
    # Calculate the upper and lower bands using the threshold
    upper_band = rolling_mean + (std_threshold * rolling_std)
    lower_band = (rolling_mean - (std_threshold * rolling_std)).clip(lower=0)

    fig = go.Figure()
    
    # Original streams line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['streams'], mode='lines', name='Daily Streams', line=dict(color='#2ca02c')
    ))

    # Standard deviation band
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=upper_band, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=lower_band, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', name=f'Mean +/- {std_threshold:.1f} SD', hoverinfo='skip'
    ))

    # Rolling average line
    fig.add_trace(go.Scatter(
        x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Average', line=dict(color='rgba(255, 127, 14, 0.8)', dash='dash')
    ))

    fig.update_layout(
        yaxis_title="Daily Streams", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, key="sa_stats_chart")

def display_single_wavelet_analysis(plot_json: str, spikes_df: pd.DataFrame, chart_key: str, title: str = ""):
    """
    Displays the results of a single wavelet analysis run.
    The plot_json is now expected to contain the spike markers already.
    """
    if title:
        st.write(f"**{title}**")

    # Display the table of detected spikes
    if spikes_df is None or spikes_df.empty:
        st.info("No significant spikes were detected with the current settings.")
    else:
        st.success(f"Detected {len(spikes_df)} potential spike(s).")
        if 'streams' in spikes_df.columns:
            st.dataframe(
                spikes_df.style.format({"streams": "{:,.0f}"}),
                use_container_width=True
            )
        else:
            st.dataframe(spikes_df, use_container_width=True)

    # Display the decomposition plot
    if not plot_json:
        st.warning("Decomposition plot is not available for display.")
        return

    try:
        # --- BUG FIX ---
        # The plot_json now contains everything needed. We just load and display it.
        # The previous logic that tried to add traces here has been removed.
        fig = pio.from_json(plot_json)
        st.plotly_chart(fig, use_container_width=True, key=f"wavelet_plot_{chart_key}")

    except Exception as e:
        st.error(f"Failed to render the wavelet decomposition plot. Error: {e}")

def display_artist_events(db_manager, events_data: List[Dict[str, Any]]):
    """
    Displays upcoming and past artist events in separate tables.
    Handles cases where ticketing and capacity information may be missing.
    """
    st.subheader("Artist Events")

    if not events_data:
        st.info("No event data found for this artist. Try updating the artist data.")
        return

    df = pd.DataFrame(events_data)
    if 'date' not in df.columns:
        st.warning("Event data is missing the 'date' column and cannot be displayed.")
        return
        
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Fetch venue and festival metadata for all unique UUIDs
    venue_uuids = set()
    festival_uuids = set()
    for _, row in df.iterrows():
        if row.get('type') == 'festival' and (festival := row.get('festival')) and isinstance(festival, dict):
            if uuid := festival.get('uuid'):
                festival_uuids.add(uuid)
        elif (venue := row.get('venue')) and isinstance(venue, dict):
            if uuid := venue.get('uuid'):
                venue_uuids.add(uuid)

    # Fetch venues
    venues_dict = {}
    if venue_uuids:
        venues_cursor = db_manager.collections['venues'].find({'uuid': {'$in': list(venue_uuids)}})
        for venue_doc in venues_cursor:
            venues_dict[venue_doc['uuid']] = venue_doc

    # Fetch festivals
    festivals_dict = {}
    if festival_uuids:
        festivals_cursor = db_manager.collections['festivals'].find({'uuid': {'$in': list(festival_uuids)}})
        for festival_doc in festivals_cursor:
            festivals_dict[festival_doc['uuid']] = festival_doc

    # Add columns with metadata
    def get_event_name(row):
        return row.get('name', 'N/A')

    def get_event_type(row):
        return row.get('type', 'Concert').capitalize()

    def get_venue_name(row):
        if row.get('type') == 'festival':
            if (festival := row.get('festival')) and isinstance(festival, dict):
                uuid = festival.get('uuid')
                meta = festivals_dict.get(uuid, {})
                return meta.get('name', festival.get('name', 'N/A'))
        else:
            if (venue := row.get('venue')) and isinstance(venue, dict):
                uuid = venue.get('uuid')
                meta = venues_dict.get(uuid, {})
                return meta.get('name', venue.get('name', 'N/A'))
        return 'N/A'

    def get_location(row):
        if row.get('type') == 'festival':
            if (festival := row.get('festival')) and isinstance(festival, dict):
                uuid = festival.get('uuid')
                meta = festivals_dict.get(uuid, {})
                city = meta.get('cityName', 'N/A')
                country = meta.get('countryCode', 'N/A')
                return f"{city}, {country}"
        else:
            if (venue := row.get('venue')) and isinstance(venue, dict):
                uuid = venue.get('uuid')
                meta = venues_dict.get(uuid, {})
                city = meta.get('cityName', 'N/A')
                country = meta.get('countryCode', 'N/A')
                return f"{city}, {country}"
        return 'N/A'

    def get_capacity(row):
        if row.get('type') == 'festival':
            if (festival := row.get('festival')) and isinstance(festival, dict):
                uuid = festival.get('uuid')
                meta = festivals_dict.get(uuid, {})
                cap = meta.get('capacity')
                return f"{int(cap):,}" if cap is not None else 'N/A'
        else:
            if (venue := row.get('venue')) and isinstance(venue, dict):
                uuid = venue.get('uuid')
                meta = venues_dict.get(uuid, {})
                cap = meta.get('capacity')
                return f"{int(cap):,}" if cap is not None else 'N/A'
        return 'N/A'

    def get_uuid(row):
        if row.get('type') == 'festival':
            return row.get('festival', {}).get('uuid', 'N/A')
        else:
            return row.get('venue', {}).get('uuid', 'N/A')

    df['event_name'] = df.apply(get_event_name, axis=1)
    df['event_type'] = df.apply(get_event_type, axis=1)
    df['venue_name'] = df.apply(get_venue_name, axis=1)
    df['location'] = df.apply(get_location, axis=1)
    df['capacity_formatted'] = df.apply(get_capacity, axis=1)
    df['uuid'] = df.apply(get_uuid, axis=1)

    # Safely access ticketing information
    if 'ticketing' in df.columns:
        df['tickets'] = df['ticketing'].apply(lambda x: x.get('url') if pd.notna(x) and isinstance(x, dict) else None)
    else:
        df['tickets'] = None

    # Separate into upcoming and past events
    today = datetime.now().date()
    upcoming_events = df[df['date'] >= today].sort_values(by='date', ascending=True)
    past_events = df[df['date'] < today].sort_values(by='date', ascending=False)

    st.write("#### Upcoming Events")
    if not upcoming_events.empty:
        display_df = upcoming_events[['date', 'event_type', 'event_name', 'venue_name', 'location', 'capacity_formatted', 'uuid', 'tickets']]
        display_df = display_df.rename(columns={
            'date': 'Date',
            'event_type': 'Type',
            'event_name': 'Event Name',
            'venue_name': 'Venue',
            'location': 'Location',
            'capacity_formatted': 'Capacity',
            'uuid': 'UUID',
            'tickets': 'Ticket Link'
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True,
            column_config={
                "Ticket Link": st.column_config.LinkColumn("Tickets", display_text="Buy Tickets")
            }
        )
    else:
        st.info("No upcoming events found.")

    st.write("#### Past Events")
    if not past_events.empty:
        with st.expander("Show Past Events"):
            display_df_past = past_events[['date', 'event_type', 'event_name', 'venue_name', 'location', 'capacity_formatted', 'uuid']]
            display_df_past = display_df_past.rename(columns={
                'date': 'Date',
                'event_type': 'Type',
                'event_name': 'Event Name',
                'venue_name': 'Venue',
                'location': 'Location',
                'capacity_formatted': 'Capacity',
                'uuid': 'UUID'
            })
            st.dataframe(display_df_past, use_container_width=True, hide_index=True)
    else:
        st.info("No past events found in the database.")
