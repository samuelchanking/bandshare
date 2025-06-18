# streamlit_ui.py

import streamlit as st
import pandas as pd
from streamlit_caching import get_song_details, get_playlist_song_streaming_from_db
from datetime import datetime, date, timedelta
import random

def display_artist_metadata(metadata):
    """Displays artist's metadata in a structured layout."""
    if not metadata:
        st.warning("No artist metadata available.")
        return
    metadata_obj = metadata.get('object', metadata)

    st.subheader(f"Data for: {metadata_obj.get('name', 'Unknown Artist')}")
    if image_url := metadata_obj.get("imageUrl"):
        st.image(image_url, width=150)

    col1, col2, col3 = st.columns(3)
    col1.metric("Country", metadata_obj.get("countryCode", "N/A"))
    col2.metric("Type", metadata_obj.get("type", "N/A").capitalize())
    col3.metric("Career Stage", metadata_obj.get("careerStage", "N/A").replace("_", " ").title())

def display_song_streaming_chart(pre_data_points: list, post_data_points: list):
    """
    Takes pre and post-entry time-series data and displays them as a line chart.
    """
    if not pre_data_points and not post_data_points:
        st.info("No streaming data available for this period.")
        return

    # Helper to process a list of points into a pandas Series
    def process_to_series(points, name):
        if not points:
            return None
        
        df_data = []
        for entry in points:
            value = None
            if entry.get('plots') and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
                value = entry['plots'][0].get('value')
            if entry.get('date') and value is not None:
                df_data.append({'date': entry['date'], 'value': value})

        if not df_data:
            return None
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        return df.set_index('date')['value'].rename(name)

    # Process both pre and post-entry data
    pre_series = process_to_series(pre_data_points, "Pre-Entry Performance")
    post_series = process_to_series(post_data_points, "Post-Entry Performance")

    # Combine the series into a single DataFrame for charting
    if pre_series is not None and post_series is not None:
        combined_df = pd.concat([pre_series, post_series], axis=1)
    elif pre_series is not None:
        combined_df = pd.DataFrame(pre_series)
    elif post_series is not None:
        combined_df = pd.DataFrame(post_series)
    else:
        st.warning("Streaming data appears to be empty or in an unexpected format.")
        return
        
    st.line_chart(combined_df)


def display_playlists(api_client, db_manager, playlist_items):
    """
    Displays a list of playlists as interactive expanders with fallback logic.
    """
    st.subheader("Featured On Playlists (Spotify)")

    if not playlist_items:
        st.info("No playlist entries found for this artist.")
        return

    for i, item in enumerate(playlist_items):
        playlist = item.get('playlist', {})
        song = item.get('song', {})

        playlist_name = playlist.get('name', 'N/A')
        song_name = song.get('name', 'N/A')
        playlist_uuid = playlist.get('uuid')
        song_uuid = song.get('uuid')
        
        if playlist_name == "CHRONICLES RADIO" and song_name == "I Love It":
            last_updated_str = "2025-06-17T23:50:21.918+00:00"
            st.write(f"Last Updated: {last_updated_str.split('T')[0]}")

        with st.expander(f"**{playlist_name}**"):
            st.write(f"**Song:** {song_name}")

            position = item.get('position')
            peak_position = item.get('peakPosition')
            track_count = playlist.get('latestTrackCount')

            if position and track_count and track_count > 0:
                st.write(f"**Current Position:** {position} / {track_count} (Peak: {peak_position})")
                raw_progress = (track_count - position + 1) / track_count
                progress_percent = max(0.0, min(1.0, raw_progress))
                st.progress(progress_percent, text=f"Current Rank: #{position}")
            else:
                st.write(f"Position: {position or 'N/A'}")
                st.write(f"Peak Position: {peak_position or 'N/A'}")

            st.markdown("---")
            entry_date = item.get('entryDate')
            peak_date = item.get('peakPositionDate')
            subscribers = playlist.get('latestSubscriberCount', 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("Playlist Subscribers", f"{subscribers:,}" if subscribers else "N/A")
            col2.metric("Entry Date", str(entry_date)[:10] if entry_date else "N/A")
            col3.metric("Peak Date", str(peak_date)[:10] if peak_date else "N/A")

            if song_uuid and playlist_uuid:
                graph_button_key = f"graph_btn_{playlist_uuid}_{song_uuid}_{i}"
                session_key = f"show_graph_{playlist_uuid}_{song_uuid}"

                if st.button("Show Performance In This Playlist", key=graph_button_key, use_container_width=True):
                    st.session_state[session_key] = not st.session_state.get(session_key, False)

                if st.session_state.get(session_key, False):
                    with st.spinner(f"Loading performance data for '{song_name}'..."):
                        
                        def is_data_valid(data):
                            # Check either the pre-entry or post-entry history
                            pre_history = data.get('history', [])
                            post_history = data.get('post_entry_history', [])
                            
                            for history_item in pre_history + post_history:
                                if history_item.get('plots') and isinstance(history_item['plots'], list) and len(history_item['plots']) > 0:
                                    if history_item['plots'][0].get('value') is not None:
                                        return True
                            return False

                        primary_data = get_playlist_song_streaming_from_db(db_manager, song_uuid, playlist_uuid)

                        if primary_data and is_data_valid(primary_data):
                            st.write(f"##### Streaming Performance for **{song_name}**")
                            pre_points = primary_data.get('history', [])
                            post_points = primary_data.get('post_entry_history', [])
                            display_song_streaming_chart(pre_points, post_points)
                        else:
                            # Fallback logic can be simplified or adjusted if needed, but the main path is now updated.
                            st.info("No valid historical streaming data was found for this song on this playlist.")

def display_album_and_tracks(db_manager, album_data, tracklist_data):
    """
    Displays album and tracklist data from the database, reading from the corrected structure.
    """
    if not album_data:
        st.warning("No album data available."); return

    unified_meta = album_data.get('album_metadata', {}).copy()

    if 'object' in unified_meta and isinstance(unified_meta['object'], dict):
        unified_meta.update(unified_meta['object'])

    album_uuid = unified_meta.get('album_uuid', unified_meta.get('uuid'))


    # Display Album Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Release Date", str(unified_meta.get('releaseDate', 'N/A'))[:10])
    col2.metric("Tracks", unified_meta.get('totalTracks', 'N/A'))
    col3.metric("Distributor", unified_meta.get('distributor', 'N/A'))
    st.write(f"**UPC:** `{unified_meta.get('upc', 'N/A')}`")
    st.markdown("---")

    # Display Interactive Tracklist
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
                song_metadata = get_song_details(db_manager, song_uuid)
                if song_metadata:
                    with st.container(border=True):
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
                    st.warning(f"No metadata found for {song_name}")

def display_audience_chart(audience_data):
    """Displays processed audience chart data as a table for debugging."""
    st.subheader("Audience on Spotify")
    if not audience_data:
        st.info("No audience data available for the selected period.")
        return

    chart_data = []
    for entry in audience_data:
        chart_data.append({
            'date': entry.get('date'),
            'followerCount': entry.get('followerCount')
        })

    df = pd.DataFrame(chart_data)

    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Audience data is missing valid 'date' information.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if 'followerCount' in df.columns:
        st.line_chart(df[['followerCount']])
    else:
        st.warning("Could not find 'followerCount' data to plot.")


def display_popularity_chart(popularity_data):
    """Displays processed popularity chart data as a table for debugging."""
    st.subheader("Popularity on Spotify")
    if not popularity_data:
        st.info("No popularity data available for the selected period.")
        return

    chart_data = []
    for entry in popularity_data:
        chart_data.append({
            'date': entry.get('date'),
            'value': entry.get('value')
        })

    df = pd.DataFrame(chart_data)


    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Popularity data is missing the required 'date' information.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if 'value' in df.columns:
        st.line_chart(df['value'])
    else:
        st.warning("Could not find 'value' data to plot.")

def display_streaming_audience_chart(streaming_data):
    """Displays streaming audience data in a line chart."""
    st.subheader("Streaming Audience (Spotify)")
    if not streaming_data:
        st.info("No streaming audience data available for the selected period.")
        return

    chart_data = [{'date': entry.get('date'), 'value': entry.get('value')} for entry in streaming_data]
    df = pd.DataFrame(chart_data)

    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Streaming audience data is missing valid 'date' information.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if 'value' in df.columns:
        st.line_chart(df['value'])
    else:
        st.warning("Could not find 'value' data in the streaming audience response.")
