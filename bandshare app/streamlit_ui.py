# streamlit_ui.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_caching import (
    get_song_details, get_full_song_data_from_db, download_image_bytes,
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


def display_full_song_streaming_chart(history_data: list, entry_points: list, chart_key: str):
    """
    Takes a full history of data points for a song and a list of playlist entries,
    and plots them on an interactive Plotly chart.
    """
    if not history_data:
        st.info("No streaming data available for this song.")
        return

    df_data = []
    for entry in history_data:
        value = None
        if entry.get('plots') and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
            value = entry['plots'][0].get('value')
        if entry.get('date') and value is not None:
            date_str = entry['date'].replace('Z', '+00:00')
            df_data.append({'date': datetime.fromisoformat(date_str).date(), 'value': value})


    if not df_data:
        st.warning("Streaming data for this song appears to be empty or in an unexpected format.")
        return

    df = pd.DataFrame(df_data)
    df.sort_values(by='date', inplace=True)
    df = df.drop_duplicates(subset=['date'], keep='last')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name='Streams', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,.0f}<extra></extra>'))
    y_range = _get_optimal_y_range(df, ['value'])
    fig.update_layout(title="Song Performance Across All Playlists", yaxis_range=y_range)

    sorted_entry_points = sorted(entry_points, key=lambda x: x.get('entryDate', ''))
    last_entry_date = None
    y_shift_offset = 15

    for entry in sorted_entry_points:
        try:
            entry_date_val = pd.to_datetime(entry['entryDate']).date()
            if last_entry_date and (entry_date_val - last_entry_date) < timedelta(days=30):
                y_shift_offset += 35
            else:
                y_shift_offset = 15

            fig.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
            
            playlist_name = entry.get('name', 'N/A')
            entry_subscribers = entry.get('entrySubscribers')
            latest_subscribers = entry.get('subscribers', 0)
            subscribers_to_show = entry_subscribers if entry_subscribers is not None else latest_subscribers
            subscribers_formatted = f"{subscribers_to_show:,}" if subscribers_to_show else "N/A"
            annotation_text = f"{playlist_name}<br>{subscribers_formatted} subs"
            hover_text = f"Added to '{playlist_name}' ({subscribers_formatted} subs) on {entry_date_val.strftime('%Y-%m-%d')}"

            fig.add_annotation(x=entry_date_val, y=df['value'].max(), text=annotation_text, showarrow=False, yshift=y_shift_offset, font=dict(color="white"), bgcolor="rgba(255, 0, 0, 0.6)", borderpad=4, hovertext=hover_text)
            last_entry_date = entry_date_val
        except (ValueError, KeyError):
            continue

    fig.update_layout(showlegend=False, hovermode="x", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# --- MODIFIED: This function is now more flexible ---
def display_playlist_audience_chart(audience_data: list, entry_date_str: str, chart_key: str):
    """
    Takes playlist audience data and plots its value (followerCount or value) over time,
    adding a vertical marker for the song's entry date.
    """
    if not audience_data:
        st.info("No audience data available for this playlist in the selected period.")
        return

    # Make parsing more flexible to handle 'followerCount' or 'value' keys.
    parsed_data = []
    for entry in audience_data:
        # Check for 'followerCount' first, then fall back to 'value'.
        value = entry.get('followerCount')
        if value is None:
            value = entry.get('value')

        if entry.get('date') and value is not None:
            parsed_data.append({'date': entry['date'], 'data_value': value})

    if not parsed_data:
        st.warning("Playlist audience data appears to be empty or in an unexpected format.")
        return

    df = pd.DataFrame(parsed_data)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    fig = go.Figure()
    # Use the generic column name 'data_value' for plotting
    fig.add_trace(go.Scatter(x=df['date'], y=df['data_value'], mode='lines', name='Followers', connectgaps=False,
                             hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Followers</b>: %{y:,.0f}<extra></extra>'))
    y_range = _get_optimal_y_range(df, ['data_value'])
    fig.update_layout(title="Playlist Followers Before and After Song Entry", yaxis_range=y_range)

    if entry_date_str:
        entry_date_val = pd.to_datetime(entry_date_str)
        fig.add_vline(x=entry_date_val, line_width=2, line_dash="dash", line_color="red")
        # Use the generic column name for positioning the annotation
        fig.add_annotation(x=entry_date_val, y=df['data_value'].max(), text="Song Entry Date", showarrow=True, arrowhead=2, ax=0, ay=-40)

    fig.update_layout(showlegend=False, hovermode="x", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def display_playlist_details(db_manager, item):
    """Displays the detailed view for a single playlist entry."""
    playlist = item.get('playlist', {})
    song = item.get('song', {})
    playlist_name = playlist.get('name', 'N/A')
    song_name = song.get('name', 'N/A')
    playlist_uuid = playlist.get('uuid')
    song_uuid = song.get('uuid')

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
        if st.button("Update and Show Playlist Audience", key=graph_button_key, use_container_width=True):
            st.session_state[session_key] = not st.session_state.get(session_key, False)
        
        if st.session_state.get(session_key, False):
            with st.spinner(f"Updating audience data for playlist '{playlist_name}'..."):
                if entry_date:
                    try:
                        api_client, db_manager_local = get_clients()
                        
                        entry_date_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).date()
                        start_date_req = entry_date_dt - timedelta(days=90)
                        end_date_req = entry_date_dt + timedelta(days=90)

                        query_filter = {'playlist_uuid': playlist_uuid}
                        min_db_date, max_db_date = db_manager_local.get_timeseries_data_range('playlist_audience', query_filter)

                        fetch_needed = False
                        if not min_db_date or not max_db_date:
                            fetch_needed = True
                        elif start_date_req < min_db_date.date() or end_date_req > max_db_date.date():
                            fetch_needed = True
                        
                        if fetch_needed:
                            new_audience_data = api_client.get_playlist_audience(playlist_uuid, start_date_req, end_date_req)
                            if new_audience_data and 'error' not in new_audience_data:
                                db_manager_local.store_playlist_audience_data(playlist_uuid, new_audience_data)
                                get_playlist_audience_data.clear()
                                st.success("Playlist audience data updated.")
                        
                        audience_data = get_playlist_audience_data(db_manager_local, playlist_uuid, start_date_req, end_date_req)
                        
                        st.write(f"##### Follower Growth for **{playlist_name}**")
                        display_playlist_audience_chart(audience_data, entry_date, chart_key=f"chart_playlist_{playlist_uuid}_{song_uuid}")

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
            song_full_data = get_full_song_data_from_db(db_manager, song_uuid)
            if song_full_data:
                display_full_song_streaming_chart(song_full_data.get('history', []), song_full_data.get('playlists', []), chart_key=f"full_chart_{song_uuid}")
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
        # DETAIL VIEW
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
        # GRID VIEW
        urls_to_download = {song['imageUrl'] for song in songs.values() if song.get('imageUrl')}
        
        prefetched_images = {}
        with st.spinner("Loading song art..."):
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(download_image_bytes, url): url for url in urls_to_download}
                for future in future_to_url:
                    url = future_to_url[future]
                    prefetched_images[url] = future.result()

        for song_uuid_chunk in grouper(songs.keys(), 4):
            cols = st.columns(4)
            for i, song_uuid in enumerate(song_uuid_chunk):
                if song_uuid:
                    with cols[i]:
                        song = songs[song_uuid]
                        image_url = song.get('imageUrl')
                        image_bytes = prefetched_images.get(image_url)
                        
                        if image_bytes:
                            st.image(image_bytes, use_container_width=True)
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
    """Displays playlist data, now with an interactive grid view."""
    st.subheader("Featured On Playlists (Spotify)")

    if not playlist_items:
        st.info("No playlist entries found for this artist.")
        return

    view_choice = st.radio("View by:", ("Playlist", "Song"), horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if view_choice == "Song":
        display_by_song_view(db_manager, playlist_items)
        return

    # --- Playlist View (Grid or Detail) ---
    selected_uuid = st.session_state.get('selected_playlist_uuid')

    if selected_uuid:
        # DETAIL VIEW
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
        # GRID VIEW
        urls_to_download = {item.get('playlist', {}).get('imageUrl') for item in playlist_items if item.get('playlist', {}).get('imageUrl')}
        
        prefetched_images = {}
        with st.spinner("Loading playlist art..."):
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_url = {executor.submit(download_image_bytes, url): url for url in urls_to_download}
                for future in future_to_url:
                    url = future_to_url[future]
                    prefetched_images[url] = future.result()

        for item_chunk in grouper(playlist_items, 4):
            cols = st.columns(4)
            for i, item in enumerate(item_chunk):
                if item:
                    with cols[i]:
                        playlist = item.get('playlist', {})
                        playlist_uuid = playlist.get('uuid')
                        playlist_name = playlist.get('name', 'N/A')
                        image_url = playlist.get('imageUrl')
                        
                        image_bytes = prefetched_images.get(image_url)
                        if image_bytes:
                            st.image(image_bytes, use_container_width=True)
                        else:
                            st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)
                        
                        st.caption(f"**{playlist_name}**")
                        st.write(f"Song: *{item.get('song', {}).get('name', 'N/A')}*")

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
