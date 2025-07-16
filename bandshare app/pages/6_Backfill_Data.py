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
    get_playlist_audience_data,
)
from datetime import date, timedelta, datetime
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
    
    artist_data = get_artist_details(db_manager, artist_uuid)
    albums = artist_data.get("albums", [])
    
    if not albums:
        st.warning("No albums found for this artist in the database.")
        return

    tasks_to_run = []
    
    with st.spinner("Checking for missing data across all albums and tracks..."):
        for album in albums:
            album_uuid = album.get("album_uuid")
            if not album_uuid: continue
            
            query_filter_album = {'album_uuid': album_uuid, 'platform': 'spotify'}
            min_db_album, max_db_album = db_manager.get_timeseries_data_range('album_audience', query_filter_album)
            
            if not min_db_album or not max_db_album:
                 tasks_to_run.append({'type': 'album', 'uuid': album_uuid, 'start': start_date, 'end': end_date})
            
            tracklist_data = get_album_details(db_manager, album_uuid)
            track_items = tracklist_data.get('tracklist', {}).get('object', {}).get('items', [])
            
            for item in track_items:
                if song_uuid := item.get('song', {}).get('uuid'):
                    query_filter_song = {'song_uuid': song_uuid, 'platform': 'spotify'}
                    min_db_song, max_db_song = db_manager.get_timeseries_data_range('song_audience', query_filter_song)
                    
                    if not min_db_song or not max_db_song:
                        tasks_to_run.append({'type': 'song', 'uuid': song_uuid, 'start': start_date, 'end': end_date})

    if not tasks_to_run:
        st.success("All album and track audience data appears to be populated for the last year.")
        return

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
        get_album_audience_data.clear()
        get_song_audience_data.clear()
        st.success("Backfill process completed successfully. New data was stored.")
        st.rerun()
    else:
        st.info("The backfill process ran, but the API did not return any new data for the missing items.")

# --- MODIFIED: The main logic of this function is updated to use the correct collection ---
def run_playlist_audience_backfill(artist_uuid):
    """
    Fetches and stores missing audience data for all playlists associated with an artist,
    specifically for the date ranges required by the UI charts (+/- 90 days around song entry).
    """
    st.info("Gathering all playlist entries for the artist from the definitive source...")
    
    # --- CRITICAL FIX: Use the 'songs_playlists' collection, not the old 'playlists' collection ---
    playlist_entries = list(db_manager.collections['songs_playlists'].find({'artist_uuid': artist_uuid}))
    
    if not playlist_entries:
        st.warning("No playlist entries found for this artist in the database.")
        return

    tasks_to_run = []
    with st.spinner("Checking for missing audience data around each song's entry date..."):
        for entry in playlist_entries:
            playlist_info = entry.get('playlist', {})
            playlist_uuid = playlist_info.get('uuid')
            entry_date_str = entry.get('entryDate')

            if not playlist_uuid or not entry_date_str:
                continue

            try:
                entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date()
                required_start = entry_date - timedelta(days=90)
                required_end = entry_date + timedelta(days=90)

                query_filter = {'playlist_uuid': playlist_uuid}
                min_db, max_db = db_manager.get_timeseries_data_range('playlist_audience', query_filter)

                fetch_needed = False
                if not min_db or not max_db:
                    fetch_needed = True
                elif max_db.date() < required_end or min_db.date() > required_start:
                    fetch_needed = True

                if fetch_needed:
                    tasks_to_run.append({
                        'uuid': playlist_uuid,
                        'name': playlist_info.get('name', 'N/A'),
                        'start': required_start,
                        'end': required_end
                    })
            except (ValueError, TypeError):
                continue
    
    final_tasks = {}
    for task in tasks_to_run:
        uuid = task['uuid']
        if uuid not in final_tasks:
            final_tasks[uuid] = task
        else:
            final_tasks[uuid]['start'] = min(final_tasks[uuid]['start'], task['start'])
            final_tasks[uuid]['end'] = max(final_tasks[uuid]['end'], task['end'])
    
    tasks_to_run = list(final_tasks.values())

    if not tasks_to_run:
        st.success("All required playlist audience data is already in the database.")
        return

    st.info(f"Found {len(tasks_to_run)} playlist(s) needing audience data updates. Starting parallel fetch...")
    data_found = False
    progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {
            executor.submit(api_client.get_playlist_audience, task['uuid'], task['start'], task['end']): task
            for task in tasks_to_run
        }

        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if data and not data.get('error') and data.get('items'):
                    db_manager.store_playlist_audience_data(task['uuid'], data)
                    data_found = True
            except Exception as exc:
                st.warning(f"A data fetch task for playlist '{task['name']}' generated an exception: {exc}")
            
            completed_count += 1
            progress_bar.progress(completed_count / len(tasks_to_run), text=f"Completed {completed_count}/{len(tasks_to_run)} tasks...")
    
    progress_bar.empty()
    if data_found:
        get_playlist_audience_data.clear()
        st.success("Playlist audience backfill completed successfully. New data was stored.")
        st.rerun()
    else:
        st.info("The backfill process ran, but the API did not return any new data for the missing items.")


# --- Page UI ---
st.title("Backfill Audience Data")

if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to begin.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.markdown(f"### Current Artist: **{artist_name}**")

end_date = date.today()
start_date = end_date - timedelta(days=365)

# --- Album/Track Backfill Section ---
with st.container(border=True):
    st.subheader("Backfill Album & Track Data")
    st.markdown("""
    This process checks all albums and tracks associated with the artist and downloads any missing audience data from the past year. 
    This is useful for artists who were added to the database before this feature was implemented.
    """)
    if st.button(f"Start Album/Track Backfill for {artist_name}", use_container_width=True, type="primary"):
        run_backfill(artist_uuid, start_date, end_date)

# --- Playlist Backfill Section ---
with st.container(border=True):
    st.subheader("Backfill Playlist Audience Data")
    st.markdown("""
    This process finds every playlist a song has been on and downloads the playlist's follower history for the 90 days before and after the song's entry date, if that data is missing.
    """)
    if st.button(f"Start Playlist Audience Backfill for {artist_name}", use_container_width=True, type="primary"):
        run_playlist_audience_backfill(artist_uuid)
