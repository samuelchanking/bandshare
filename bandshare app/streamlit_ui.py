# streamlit_ui.py

# --- ADD THESE IMPORTS TO THE TOP OF THE FILE ---
import json
import numpy as np
from analysis_tools import detect_anomalous_spikes, convert_cumulative_to_daily
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_caching import (
    get_song_details, get_full_song_data_from_db,
    get_song_audience_data, get_playlist_audience_data
)
from datetime import datetime, date, timedelta
import plotly.express as px
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor
from client_setup import initialize_clients 
import config

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
            history = audience_data.get('history', audience_data)
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
    """
    Takes a full history of data points for a song. Displays the original
    cumulative data chart by default, followed by the calculated daily streams chart with spike analysis.
    """
    if not history_data:
        st.info("No streaming data available for this song.")
        return

    parsed_history = []
    for entry in history_data:
        if 'date' in entry and 'plots' in entry and isinstance(entry['plots'], list) and entry['plots']:
            value = entry['plots'][0].get('value')
            if value is not None:
                parsed_history.append({'date': entry['date'], 'value': value})

    if not parsed_history:
        st.warning("Streaming data for this song appears to be empty or in an unexpected format.")
        return

    daily_stream_data = convert_cumulative_to_daily(parsed_history)
    df_daily = pd.DataFrame(daily_stream_data)
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    df_cumulative_raw = pd.DataFrame(parsed_history)
    df_cumulative_raw['date'] = pd.to_datetime(df_cumulative_raw['date'])
    df_cumulative_raw.sort_values(by='date', inplace=True)
    df_cumulative = df_cumulative_raw.drop_duplicates(subset=['date'], keep='last')

    if len(df_daily) > 1:
        df_daily = df_daily.iloc[1:].copy()
    if len(df_cumulative) > 1:
        df_cumulative = df_cumulative.iloc[1:].copy()
    
    st.subheader("Raw Data: Total Accumulated Streams")
    st.caption("This chart shows the original, cumulative stream data for the song over time.")
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=df_cumulative['date'], y=df_cumulative['value'], mode='lines', name='Total Streams',
                                       hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Streams</b>: %{y:,.0f}<extra></extra>'))
    fig_cumulative.update_layout(showlegend=False, yaxis_title="Cumulative Streams")
    st.plotly_chart(fig_cumulative, use_container_width=True, key=f"cumulative_chart_{chart_key}")
    st.markdown("---")

    st.subheader("Calculated Data: Daily Streams & Playlist Adds")
    st.caption("*Note: The first data point is removed by default to improve chart readability.*")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=df_daily['date'], y=df_daily['value'], mode='lines', name='Daily Streams', connectgaps=False,
                                   hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily Streams</b>: %{y:,.0f}<extra></extra>'))
    
    sorted_entry_points = sorted(entry_points, key=lambda x: x.get('entryDate', ''))
    last_entry_date = None
    y_shift_offset = 15
    y_max_daily = df_daily['value'].max() if not df_daily.empty else 0

    for entry in sorted_entry_points:
        try:
            entry_date_dt = datetime.fromisoformat(entry['entryDate'].replace('Z', '+00:00'))
            entry_date_val = entry_date_dt.date()
            if last_entry_date and (entry_date_val - last_entry_date) < timedelta(days=30):
                y_shift_offset += 35
            else:
                y_shift_offset = 15
            fig_daily.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            playlist_name = entry.get('name', 'N/A')
            playlist_uuid = entry.get('uuid')
            entry_subscribers = entry.get('entrySubscribers')
            if entry_subscribers is None and playlist_uuid:
                entry_subscribers = db_manager.get_timeseries_value_for_date('playlist_audience', {'playlist_uuid': playlist_uuid}, entry_date_dt)
            subscribers_formatted = f"{entry_subscribers:,}" if entry_subscribers is not None else "N/A"
            annotation_text = f"{playlist_name}<br>{subscribers_formatted} subs"
            hover_text = f"Added to '{playlist_name}' ({subscribers_formatted} subs) on {entry_date_val.strftime('%Y-%m-%d')}"
            fig_daily.add_annotation(x=entry_date_val, y=y_max_daily, text=annotation_text, showarrow=False, yshift=y_shift_offset, font=dict(color="white"), bgcolor="rgba(255, 0, 0, 0.6)", borderpad=4, hovertext=hover_text)
            last_entry_date = entry_date_val
        except (ValueError, KeyError) as e:
            print(f"Could not process playlist annotation. Error: {e}")
            continue

    # Add Song Release Date marker
    if song_release_date:
        try:
            release_date_dt = pd.to_datetime(song_release_date)
            if not df_daily.empty and 'date' in df_daily.columns:
                chart_start_date = df_daily['date'].min()
                chart_end_date = df_daily['date'].max()

                if not pd.isna(chart_start_date) and not pd.isna(chart_end_date) and chart_start_date <= release_date_dt <= chart_end_date:
                    fig_daily.add_vline(x=release_date_dt, line_width=2, line_dash="solid", line_color="green")
                    fig_daily.add_annotation(
                        x=release_date_dt,
                        y=y_max_daily,
                        text="Song Release",
                        showarrow=True,
                        arrowhead=1,
                        ax=-25,
                        ay=-60,
                        font=dict(color="green"),
                        bgcolor="rgba(0, 200, 0, 0.6)"
                    )
        except (ValueError, TypeError) as e:
            st.warning(f"Could not plot release date marker due to an issue with the date format: {e}")

    analysis_session_key = f'analysis_results_{chart_key}'
    if st.session_state.get(analysis_session_key):
        results = st.session_state[analysis_session_key]
        if 'anomalies' in results and results['anomalies']:
            for anomaly in results['anomalies']:
                anomaly_date = datetime.strptime(anomaly['date'], '%Y-%m-%d').date()
                jump_size_formatted = f"+{int(anomaly['jump_size']):,}"
                fig_daily.add_vline(x=anomaly_date, line_width=1, line_dash="dot", line_color="cyan")
                fig_daily.add_annotation(
                    x=anomaly_date, y=y_max_daily, text=f"Spike: {jump_size_formatted}",
                    showarrow=False, yshift=10, font=dict(color="cyan"), bgcolor="rgba(0, 50, 100, 0.6)")

    y_range = _get_optimal_y_range(df_daily, ['value'])
    fig_daily.update_layout(yaxis_range=y_range, showlegend=False, hovermode="x unified", yaxis_tickformat=",.0f")
    st.plotly_chart(fig_daily, use_container_width=True, key=f"daily_chart_{chart_key}")

    with st.expander("🔬 Analyze Streaming Spikes"):
        st.markdown("""
        This tool detects anomalous spikes by analyzing the rate of change of the cumulative stream count.
        - **Discretization Step:** The window (in days) for analyzing the rate of change.
        - **Alpha (Smoothing Factor):** Controls how quickly the average adapts to new trends.
        - **Sensitivity:** How far from the average the rate of change must be to be flagged as a spike.
        """)
        
        param_cols = st.columns(3)
        p_step = param_cols[0].number_input("Discretization Step (days)", min_value=1, max_value=30, value=7, step=1, key=f"step_{chart_key}")
        p_alpha = param_cols[1].slider("Alpha (Smoothing Factor)", min_value=0.01, max_value=1.0, value=0.2, step=0.01, key=f"alpha_{chart_key}")
        p_sens = param_cols[2].number_input("Sensitivity", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key=f"sens_{chart_key}")
        
        button_cols = st.columns(2)
        if button_cols[0].button("Run Analysis", key=f"analyze_{chart_key}", use_container_width=True, type="primary"):
            dates_for_analysis = df_cumulative['date'].dt.strftime('%Y-%m-%d').tolist()
            streams_for_analysis = df_cumulative['value'].tolist()
            data_tuple = (tuple(dates_for_analysis), tuple(streams_for_analysis))
            with st.spinner("Analyzing data for anomalies..."):
                result_json = detect_anomalous_spikes(data_tuple, discretization_step=p_step, alpha=p_alpha, sensitivity=p_sens)
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
                anomaly_df['jump_size'] = anomaly_df['jump_size'].apply(lambda x: f"+{int(x):,}")
                anomaly_df.rename(columns={'date': 'Date', 'jump_size': 'Jump Size (in streams)'}, inplace=True)
                st.dataframe(anomaly_df, use_container_width=True)
            
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
    
    # --- Helper function to add markers to charts ---
    def add_markers_to_fig(fig, df):
        # Add Song Entry Date marker
        if entry_date_str:
            entry_date_val = pd.to_datetime(entry_date_str)
            fig.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            if not df.empty:
                fig.add_annotation(x=entry_date_val, y=df['value'].max(), text="Song Entry", showarrow=True, arrowhead=2, ax=0, ay=-40)
        
        # Add Song Release Date marker if it's within the chart's date range
        if song_release_date and not df.empty:
            release_date_dt = pd.to_datetime(song_release_date)
            chart_start_date = df['date'].min()
            chart_end_date = df['date'].max()
            if chart_start_date <= release_date_dt <= chart_end_date:
                fig.add_vline(x=release_date_dt, line_width=2, line_dash="solid", line_color="green")
                fig.add_annotation(x=release_date_dt, y=df['value'].max(), text="Song Release", showarrow=True, arrowhead=1, ax=20, ay=-60, font=dict(color="green"))


    # --- 1. Plot Original (Cumulative) Data ---
    st.write("##### Original Data: Cumulative Follower Count")
    st.caption("This chart shows the raw, cumulative follower data as stored in the database over time.")
    
    df_cumulative = pd.DataFrame(parsed_data)
    df_cumulative['date'] = pd.to_datetime(df_cumulative['date'])
    sorted_df_cumulative = df_cumulative.sort_values(by='date')

    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=sorted_df_cumulative['date'], y=sorted_df_cumulative['value'], mode='lines', name='Total Followers',
                                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Followers</b>: %{y:,.0f}<extra></extra>'))
    
    y_range_cumul = _get_optimal_y_range(sorted_df_cumulative, ['value'])
    fig_cumulative.update_layout(title="Cumulative Playlist Followers", yaxis_range=y_range_cumul, yaxis_tickformat=",.0f", showlegend=False)
    add_markers_to_fig(fig_cumulative, sorted_df_cumulative)
    st.plotly_chart(fig_cumulative, use_container_width=True, key=f"{chart_key}_cumulative")
    st.markdown("---")


    # --- 2. Plot Normalized (Daily) Data ---
    st.write("##### Normalized Data: Daily Follower Change")
    st.caption("This chart shows the calculated day-over-day change in followers.")

    daily_data = convert_cumulative_to_daily(parsed_data)
    
    if not daily_data:
        st.warning("Could not calculate daily changes from the source data.")
        return

    df_daily = pd.DataFrame(daily_data)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    sorted_df_daily = df_daily.sort_values(by='date')
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=sorted_df_daily['date'], y=sorted_df_daily['value'], mode='lines', name='Daily Change',
                                   hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Daily Change</b>: %{y:,.0f}<extra></extra>'))

    y_range_daily = _get_optimal_y_range(sorted_df_daily, ['value'])
    fig_daily.update_layout(title="Daily Change in Playlist Followers", yaxis_range=y_range_daily, yaxis_tickformat=",.0f", showlegend=False)
    add_markers_to_fig(fig_daily, sorted_df_daily)
    st.plotly_chart(fig_daily, use_container_width=True, key=f"{chart_key}_daily")


def display_playlist_details(db_manager, item):
    """Displays the detailed view for a single playlist entry."""
    playlist = item.get('playlist', {})
    song = item.get('song', {})
    playlist_name = playlist.get('name', 'N/A')
    song_name = song.get('name', 'N/A')
    playlist_uuid = playlist.get('uuid')
    song_uuid = song.get('uuid')

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

    if song_uuid and playlist_uuid:
        graph_button_key = f"graph_btn_{playlist_uuid}_{song_uuid}"
        session_key = f"show_graph_{playlist_uuid}_{song_uuid}"
        
        if st.button("Show Playlist Audience", key=graph_button_key, use_container_width=True):
            st.session_state[session_key] = not st.session_state.get(session_key, False)
        
        if st.session_state.get(session_key, False):
            if entry_date:
                try:
                    _, db_manager_local = get_clients()
                    
                    # --- NEW: Fetch song details to get the release date ---
                    song_details = get_song_details(db_manager_local, song_uuid)
                    song_release_date = None
                    if song_details:
                        song_release_date = song_details.get('object', song_details).get('releaseDate')
                    # ----------------------------------------------------

                    entry_date_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).date()
                    start_date_req = entry_date_dt - timedelta(days=90)
                    end_date_req = entry_date_dt + timedelta(days=90)
                    
                    with st.spinner(f"Loading audience data for playlist '{playlist_name}' from database..."):
                        audience_data = get_playlist_audience_data(db_manager_local, playlist_uuid, start_date_req, end_date_req)
                        st.write(f"##### Follower Growth for **{playlist_name}**")
                        # --- MODIFIED: Pass release date to the chart function ---
                        display_playlist_audience_chart(
                            audience_data, 
                            entry_date, 
                            song_release_date, 
                            chart_key=f"chart_playlist_{playlist_uuid}_{song_uuid}"
                        )

                except (ValueError, TypeError) as e:
                    st.error(f"Could not process the request. Error: {e}")
            else:
                st.info("Cannot display performance graph without a song entry date.")


def display_song_details(db_manager, song_uuid, song_data):
    """Displays the detailed view for a single song."""
    song_metadata = get_song_details(db_manager, song_uuid)
    if song_metadata:
        meta_obj = song_metadata.get('object', song_metadata)
        
        st.write(f"**Song UUID:** `{song_uuid}`")

        col1, col2 = st.columns(2)
        
        release_date = meta_obj.get('releaseDate', 'N/A')
        col1.metric("Release Date", str(release_date)[:10])
        
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
            st.write("**Genres:** N/A")

    else:
        st.info("Additional song metadata not found in the database. Please update artist data on the homepage.")
    
    st.markdown("---")
    st.write("**Featured in playlists:**")
    for p in song_data['playlists']:
        st.write(f"- {p['name']} (Added on: {str(p.get('entryDate', 'N/A'))[:10]})")

    st.markdown("---")
    session_key = f"show_song_graph_{song_uuid}"
    if st.button("Show Full Performance Graph", key=f"song_graph_btn_{song_uuid}", use_container_width=True):
        st.session_state[session_key] = not st.session_state.get(session_key, False)
    
    if st.session_state.get(session_key, False):
        with st.spinner(f"Loading aggregated performance for '{song_data['name']}'..."):
            _, db_manager_local = get_clients()
            song_full_data = get_full_song_data_from_db(db_manager, song_uuid)
            if song_full_data:
                display_full_song_streaming_chart(db_manager_local, song_full_data.get('history', []), song_full_data.get('playlists', []), chart_key=f"full_chart_{song_uuid}")
            else:
                st.warning("Aggregated streaming data not found. Please update the artist data on the homepage to fetch it.")
                
    

def display_by_song_view(db_manager, playlist_items):
    """Displays songs in an interactive grid, or a detail view for a selected song."""
    songs = {}
    for item in playlist_items:
        song = item.get('song', {})
        song_uuid = song.get('uuid')
        if not song_uuid: continue
        
        if song_uuid not in songs:
            songs[song_uuid] = {
                'name': song.get('name', 'N/A'),
                'imageUrl': song.get('imageUrl'),
                'playlists': []
            }
        songs[song_uuid]['playlists'].append({'name': item.get('playlist', {}).get('name', 'N/A'), 'entryDate': item.get('entryDate')})

    if not songs:
        st.info("Could not identify any songs from the playlist data.")
        return

    selected_uuid = st.session_state.get('selected_song_uuid')

    if selected_uuid:
        song_data = songs.get(selected_uuid)
        if song_data:
            st.subheader(f"Details for: {song_data['name']}")
            if st.button("⬅️ Back to all songs"):
                st.session_state.selected_song_uuid = None
                st.rerun()
            display_song_details(db_manager, selected_uuid, song_data)
        else:
            st.error("Could not find selected song details.")
            st.session_state.selected_song_uuid = None
            st.rerun()
    else:
        # Simplified display logic
        for song_uuid_chunk in grouper(songs.keys(), 4):
            cols = st.columns(4)
            for i, song_uuid in enumerate(song_uuid_chunk):
                if song_uuid:
                    with cols[i]:
                        song = songs[song_uuid]
                        image_url = song.get('imageUrl')
                        
                        if image_url:
                            st.image(image_url, use_container_width=True)
                        else:
                            st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)
                        
                        st.caption(f"**{song['name']}**")

                        if st.button("Details", key=f"btn_song_{song_uuid}", use_container_width=True):
                            st.session_state.selected_song_uuid = song_uuid
                            st.rerun()

def grouper(iterable, n, fillvalue=None):
    "Helper to collect data into fixed-length chunks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

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

    selected_uuid = st.session_state.get('selected_playlist_uuid')

    if selected_uuid:
        # Detailed view logic (unchanged)
        selected_item = next((item for item in playlist_items if item.get('playlist', {}).get('uuid') == selected_uuid), None)
        if selected_item:
            playlist_name = selected_item.get('playlist', {}).get('name', 'Unknown Playlist')
            st.subheader(f"Details for: {playlist_name}")
            if st.button("⬅️ Back to all playlists"):
                st.session_state.selected_playlist_uuid = None
                st.rerun()
            display_playlist_details(db_manager, selected_item)
        else:
            st.error("Could not find selected playlist details.")
            st.session_state.selected_playlist_uuid = None
            st.rerun()
    else:
        # --- NEW: UI Controls for Searching and Sorting ---
        controls_cols = st.columns([2, 1, 1])
        with controls_cols[0]:
            search_term = st.text_input("Search by playlist name...")
        with controls_cols[1]:
            sort_key = st.selectbox("Sort by", ["Subscriber Count", "Alphabetical"])
        with controls_cols[2]:
            sort_order_label = st.radio("Order", ["High to Low", "Low to High"])

        # --- NEW: Filtering and Sorting Logic ---
        display_items = playlist_items
        if search_term:
            display_items = [
                item for item in display_items
                if search_term.lower() in item.get('playlist', {}).get('name', '').lower()
            ]

        reverse_order = (sort_order_label == "High to Low")
        if sort_key == "Alphabetical":
            sort_lambda = lambda item: item.get('playlist', {}).get('name', '').lower()
        else:  # Subscriber Count
            sort_lambda = lambda item: item.get('playlist', {}).get('latestSubscriberCount', 0)

        display_items = sorted(display_items, key=sort_lambda, reverse=reverse_order)
        # -----------------------------------------

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

                        if image_url:
                            st.image(image_url, use_container_width=True)
                        else:
                            st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)

                        st.caption(f"**{playlist_name}**")
                        st.write(f"Song: *{item.get('song', {}).get('name', 'N/A')}*")

                        if playlist_uuid:
                            st.caption(f"UUID: `{playlist_uuid}`")

                        if st.button("Details", key=f"btn_playlist_{playlist_uuid}", use_container_width=True):
                            st.session_state.selected_playlist_uuid = playlist_uuid
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
                        song_aud_data = get_song_audience_data(db_manager, song_uuid, "spotify", start_date, end_date)
                        display_timeseries_chart(song_aud_data, title="Song Audience Over Time")


def display_timeseries_chart(chart_data, title=""):
    """
    Displays a generic time-series chart, ensuring data is sorted and
    gaps are not connected.
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

    df = pd.DataFrame(parsed_data)
    df.sort_values(by='date', inplace=True)
    df.set_index('date', inplace=True)

    if df.empty or 'value' not in df.columns:
        st.warning("Could not find any valid data to plot after parsing.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name='Value', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:,.0f}<extra></extra>'))
    
    y_range = _get_optimal_y_range(df, ['value'])
    fig.update_layout(
        title="",
        yaxis_range=y_range, 
        yaxis_tickformat=",.0f",
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)


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


def display_streaming_audience_chart(streaming_data):
    """Displays streaming audience data in a line chart."""
    st.subheader("Streaming Audience (Spotify)")
    if not streaming_data:
        st.info("No streaming audience data available for the selected period.")
        return

    df = pd.DataFrame(streaming_data).rename(columns={'value': 'Streams'})
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['Streams'], mode='lines', name='Streams', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,.0f}<extra></extra>'))
    y_range = _get_optimal_y_range(df, ['Streams'])
    fig.update_layout(title="Total Streams Over Time", yaxis_range=y_range, yaxis_tickformat=",.0f", showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def display_local_streaming_plots(local_streaming_data):
    """Creates interactive dropdowns to plot local streaming data."""
    st.subheader("Local Streaming Performance")
    if not local_streaming_data:
        st.info("No local streaming data available for this period.")
        return

    all_countries = set()
    country_to_cities = {}
    plot_data = []

    for daily_entry in local_streaming_data:
        date_val = daily_entry.get('date')
        for country_plot in daily_entry.get('countryPlots', []):
            country_name = country_plot.get('countryName')
            if country_name:
                all_countries.add(country_name)
                plot_data.append({'date': date_val, 'country': country_name, 'city': None, 'streams': country_plot.get('value')})
        for city_plot in daily_entry.get('cityPlots', []):
            country_name = city_plot.get('countryName')
            city_name = city_plot.get('cityName')
            if country_name and city_name:
                all_countries.add(country_name)
                if country_name not in country_to_cities: country_to_cities[country_name] = set()
                country_to_cities[country_name].add(city_name)
                plot_data.append({'date': date_val, 'country': country_name, 'city': city_name, 'streams': city_plot.get('value')})
    
    if not plot_data:
        st.info("Could not find any stream data in the local breakdown.")
        return
        
    df = pd.DataFrame(plot_data)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    
    sorted_countries = sorted(list(all_countries))
    selected_country = st.selectbox("Select a Country", sorted_countries)

    if selected_country:
        cities = sorted(list(country_to_cities.get(selected_country, [])))
        city_options = ["All Cities"] + cities
        selected_city = st.selectbox("Select a City", city_options)

        title = f"Daily Streams in {selected_country}"
        if selected_city and selected_city != "All Cities":
            filtered_df = df[(df['country'] == selected_country) & (df['city'] == selected_city)]
            title = f"Daily Streams in {selected_city}, {selected_country}"
        else:
            filtered_df = df[(df['country'] == selected_country) & (df['city'].isnull())]
        
        if not filtered_df.empty:
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['streams'], mode='lines', name='Streams', connectgaps=False,
                                     hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,.0f}<extra></extra>'))
            y_range = _get_optimal_y_range(filtered_df, ['streams'])
            fig.update_layout(title=title, yaxis_range=y_range, yaxis_tickformat=",.0f", showlegend=False, hovermode='x unified')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No streaming data found for the selected location.")
