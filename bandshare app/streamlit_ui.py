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

def display_song_streaming_chart(data_points: list):
    """
    Takes a list of time-series data points for a song and displays a line chart.
    """
    if not data_points:
        st.info("No streaming data available for this period.")
        return

    df_data = []
    for entry in data_points:
        # MODIFIED: Correctly access the nested 'value' inside the 'plots' array.
        # This handles cases where 'plots' might be missing or empty.
        value = None
        if entry.get('plots') and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
            value = entry['plots'][0].get('value')

        if entry.get('date') and value is not None:
            df_data.append({
                'date': entry.get('date'),
                'value': value
            })

    if not df_data:
        st.warning("Streaming data appears to be empty or in an unexpected format.")
        return

    df = pd.DataFrame(df_data)
    if 'date' not in df.columns or df['value'].isnull().all():
        st.warning("Could not find valid data to plot.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    st.line_chart(df['value'])


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
        
        # This is the entry for "I Love It" on "CHRONICLES RADIO" from your screenshot
        # The last_updated date is from the first screenshot provided
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
                        
                        # Helper to check if data is valid for charting
                        def is_data_valid(data):
                            if not data or not data.get('history'):
                                return False
                            # MODIFIED: Correctly check for the nested 'value'
                            for history_item in data['history']:
                                if history_item.get('plots') and isinstance(history_item['plots'], list) and len(history_item['plots']) > 0:
                                    if history_item['plots'][0].get('value') is not None:
                                        return True # Found at least one valid data point
                            return False

                        # 1. Try to get the primary song's data
                        primary_data = get_playlist_song_streaming_from_db(db_manager, song_uuid, playlist_uuid)

                        if is_data_valid(primary_data):
                            st.write(f"##### Streaming Performance for **{song_name}**")
                            display_song_streaming_chart(primary_data['history'])
                        else:
                            # 2. If primary data is invalid, search for fallbacks
                            st.info(f"No valid data for '{song_name}'. Searching for other songs on this playlist...")
                            
                            artist_uuid = item.get('artist_uuid')
                            fallback_entries = list(db_manager.collections['playlists'].find({
                                'artist_uuid': artist_uuid, 
                                'playlist.uuid': playlist_uuid, 
                                'song.uuid': {'$ne': song_uuid}
                            }))

                            fallback_found = False
                            if not fallback_entries:
                                st.warning("No other songs by this artist were found on this playlist.")
                            else:
                                for fallback_item in fallback_entries:
                                    fallback_song_uuid = fallback_item.get('song', {}).get('uuid')
                                    if not fallback_song_uuid: continue

                                    fallback_data = get_playlist_song_streaming_from_db(db_manager, fallback_song_uuid, playlist_uuid)
                                    if is_data_valid(fallback_data):
                                        fallback_song_name = fallback_item.get('song', {}).get('name', 'Unknown Song')
                                        st.success(f"Displaying performance data for alternate song: **{fallback_song_name}**")
                                        display_song_streaming_chart(fallback_data['history'])
                                        fallback_found = True
                                        break
                            
                            if not fallback_found:
                                 st.info("No valid historical streaming data was found for any song by this artist on this playlist.")

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
