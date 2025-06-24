# pages/5_Backfill_Data.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import (
    get_artist_details,
    get_album_details,
    get_album_audience_data,
    get_song_audience_data,
)
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Backfill Data", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Backfill Logic ---
def run_backfill(artist_uuid, start_date, end_date):
    """
    Fetches and stores all missing album and track audience data for a given artist.
    """
    st.info("Gathering all albums and tracks for the artist...")
    
    # Get all albums and their corresponding tracks
    artist_data = get_artist_details(db_manager, artist_uuid)
    albums = artist_data.get("albums", [])
    
    if not albums:
        st.warning("No albums found for this artist in the database.")
        return

    tasks_to_run = []
    
    # --- Determine all required data fetches ---
    with st.spinner("Checking for missing data across all albums and tracks..."):
        # Check parent albums
        for album in albums:
            album_uuid = album.get("album_uuid")
            if not album_uuid: continue
            
            # Check for missing album audience data
            query_filter_album = {'album_uuid': album_uuid, 'platform': 'spotify'}
            min_db_album, max_db_album = db_manager.get_timeseries_data_range('album_audience', query_filter_album)
            
            if not min_db_album or not max_db_album: # If no data exists, fetch the whole range
                 tasks_to_run.append({'type': 'album', 'uuid': album_uuid, 'start': start_date, 'end': end_date})
            
            # Check tracks within the album
            tracklist_data = get_album_details(db_manager, album_uuid)
            track_items = tracklist_data.get('tracklist', {}).get('object', {}).get('items', [])
            
            for item in track_items:
                if song_uuid := item.get('song', {}).get('uuid'):
                    query_filter_song = {'song_uuid': song_uuid, 'platform': 'spotify'}
                    min_db_song, max_db_song = db_manager.get_timeseries_data_range('song_audience', query_filter_song)
                    
                    if not min_db_song or not max_db_song: # If no data exists, fetch the whole range
                        tasks_to_run.append({'type': 'song', 'uuid': song_uuid, 'start': start_date, 'end': end_date})

    if not tasks_to_run:
        st.success("All album and track audience data appears to be populated for the last year.")
        return

    # --- Execute all fetches in parallel ---
    st.info(f"Found {len(tasks_to_run)} items missing data. Starting parallel fetch...")
    data_found = False
    progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
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
        st.success("Backfill process completed successfully. New data was stored.")
        st.rerun()
    else:
        st.info("The backfill process ran, but the API did not return any new data for the missing items.")


# --- Page UI ---
st.title("Backfill Audience Data")
st.markdown("""
This page allows you to retroactively fetch and store audience data for the currently selected artist. 
It will check all albums and tracks associated with the artist and download any missing audience data from the past year.

This is useful for artists who were added to the database before the album/track audience feature was implemented.
""")

if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to begin.")
    st.stop()

artist_name = st.session_state.get('artist_name', 'the selected artist')
st.markdown(f"### Current Artist: **{artist_name}**")

if st.button(f"Start Backfill for {artist_name}", use_container_width=True, type="primary"):
    artist_uuid = st.session_state.artist_uuid
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    run_backfill(artist_uuid, start_date, end_date)