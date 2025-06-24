# pages/1_Albums.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import (
    get_artist_details, get_album_details, 
    download_image_bytes, get_album_audience_data, get_song_audience_data
)
from streamlit_ui import display_album_and_tracks
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Albums", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State for Album Selection ---
if 'selected_album_uuid' not in st.session_state:
    st.session_state.selected_album_uuid = None

# --- Helper to chunk data for grid layout ---
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their albums.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Albums for {artist_name}")

artist_details = get_artist_details(db_manager, artist_uuid)
all_albums = artist_details.get("albums", [])

if not all_albums:
    st.warning("No album data found for this artist.")
    st.stop()

# --- View Toggle Logic ---

# DETAILED VIEW (if an album has been selected)
if st.session_state.selected_album_uuid:
    selected_album_data = next((album for album in all_albums if album.get("album_uuid") == st.session_state.selected_album_uuid), None)
    
    if selected_album_data:
        album_uuid = st.session_state.selected_album_uuid
        album_details_from_db = get_album_details(db_manager, album_uuid)
        tracklist_data = album_details_from_db.get("tracklist") if album_details_from_db else None

        album_name = selected_album_data.get("album_metadata", {}).get("name", "Unknown Album")
        st.subheader(f"Details for: {album_name}")

        if st.button("⬅️ Back to all albums"):
            st.session_state.selected_album_uuid = None
            st.rerun()

        # --- Date filters and Update button for album and track audience ---
        st.markdown("### Audience Charts")
        start_date_filter = st.date_input("Chart Start Date", date.today() - timedelta(days=365), key=f"start_{album_uuid}")
        end_date_filter = st.date_input("Chart End Date", date.today(), key=f"end_{album_uuid}")

        if st.button("Update Album & Track Audience Data", use_container_width=True, type="primary"):
            
            tasks_to_run = []
            with st.spinner("Checking for date ranges to update..."):
                # --- Task for Parent Album ---
                query_filter_album = {'album_uuid': album_uuid, 'platform': 'spotify'}
                min_db_album, max_db_album = db_manager.get_timeseries_data_range('album_audience', query_filter_album)
                
                if min_db_album and start_date_filter < min_db_album.date():
                    tasks_to_run.append({'type': 'album', 'uuid': album_uuid, 'start': start_date_filter, 'end': min_db_album.date() - timedelta(days=1)})
                
                forward_start_album = max_db_album.date() + timedelta(days=1) if max_db_album else start_date_filter
                if forward_start_album <= end_date_filter:
                    tasks_to_run.append({'type': 'album', 'uuid': album_uuid, 'start': forward_start_album, 'end': end_date_filter})

                # --- Tasks for Each Track ---
                track_items = tracklist_data.get('object', tracklist_data).get('items', []) if tracklist_data else []
                for item in track_items:
                    if song_uuid := item.get('song', {}).get('uuid'):
                        query_filter_song = {'song_uuid': song_uuid, 'platform': 'spotify'}
                        min_db_song, max_db_song = db_manager.get_timeseries_data_range('song_audience', query_filter_song)

                        if min_db_song and start_date_filter < min_db_song.date():
                            tasks_to_run.append({'type': 'song', 'uuid': song_uuid, 'start': start_date_filter, 'end': min_db_song.date() - timedelta(days=1)})

                        forward_start_song = max_db_song.date() + timedelta(days=1) if max_db_song else start_date_filter
                        if forward_start_song <= end_date_filter:
                            tasks_to_run.append({'type': 'song', 'uuid': song_uuid, 'start': forward_start_song, 'end': end_date_filter})
            
            if not tasks_to_run:
                st.info("Album and all track audience data are already up-to-date for the selected range.")
            else:
                data_found = False
                progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_task = {}
                    for task in tasks_to_run:
                        if task['type'] == 'album':
                            future = executor.submit(api_client.get_album_audience, task['uuid'], 'spotify', task['start'], task['end'])
                        else: # song
                            future = executor.submit(api_client.get_song_streaming_audience, task['uuid'], 'spotify', task['start'], task['end'])
                        future_to_task[future] = task

                    completed_count = 0
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            data = future.result()
                            if data and not data.get('error') and data.get('items'):
                                if task['type'] == 'album':
                                    db_manager.store_album_audience_data(task['uuid'], data)
                                else: # song
                                    db_manager.store_song_audience_data(task['uuid'], data)
                                data_found = True
                        except Exception as exc:
                            st.warning(f"A data fetch task for {task['type']} {task['uuid']} generated an exception: {exc}")
                        
                        completed_count += 1
                        progress_bar.progress(completed_count / len(tasks_to_run), text=f"Completed {completed_count}/{len(tasks_to_run)} tasks...")
                
                progress_bar.empty()
                if data_found:
                    # Clear all relevant caches
                    get_album_audience_data.clear()
                    get_song_audience_data.clear()
                    st.success("Album and track audience data updated successfully.")
                    st.rerun()
                else:
                    st.info("Checked for new data, but none was found for the specified range(s).")
        
        # --- Fetch and display data ---
        album_aud_data = get_album_audience_data(db_manager, album_uuid, "spotify", start_date_filter, end_date_filter)
        display_album_and_tracks(db_manager, selected_album_data, tracklist_data, album_aud_data, start_date_filter, end_date_filter)
    else:
        st.error("Could not find the selected album details. Returning to album list.")
        st.session_state.selected_album_uuid = None
        st.rerun()

# GRID VIEW (default view)
else:
    st.subheader("Select an album to view details")
    st.markdown("---")

    # --- Step 1: Gather all unique image URLs ---
    urls_to_download = set()
    url_map = {} # To easily find the url for an album later
    for album in all_albums:
        metadata = album.get("album_metadata", {})
        nested_object = metadata.get("object", {})
        image_url = metadata.get("imageUrl") or nested_object.get("imageUrl")
        if image_url:
            urls_to_download.add(image_url)
            url_map[album.get("album_uuid")] = image_url

    # --- Step 2: Download images in parallel using the cached function ---
    prefetched_images = {}
    with st.spinner("Loading album art..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(download_image_bytes, url): url for url in urls_to_download}
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    prefetched_images[url] = future.result()
                except Exception:
                    prefetched_images[url] = None
    
    # --- Step 3: Display the grid using the prefetched images ---
    for album_chunk in grouper(all_albums, 4):
        cols = st.columns(4)
        for i, album in enumerate(album_chunk):
            if album:
                with cols[i]:
                    album_uuid = album.get("album_uuid")
                    metadata = album.get("album_metadata", {})
                    album_name = metadata.get("name", "Unknown Album")
                    
                    image_url = url_map.get(album_uuid)
                    image_bytes = prefetched_images.get(image_url)
                    
                    if image_bytes:
                        st.image(image_bytes, use_container_width=True)
                    else:
                        st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)

                    st.caption(album_name)
                    
                    if st.button("Details", key=f"btn_details_{album_uuid}", use_container_width=True):
                        st.session_state.selected_album_uuid = album_uuid
                        st.rerun()
