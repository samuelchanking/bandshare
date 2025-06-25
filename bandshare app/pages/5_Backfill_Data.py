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
    get_full_song_data_from_db,
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
def run_album_backfill(artist_uuid, start_date, end_date):
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
            else: 
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
                    else: 
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

# --- NEW FUNCTION for Song Backfill ---
def run_song_playlist_backfill(artist_uuid):
    """
    Fetches and stores missing historical audience data for songs based on their
    earliest playlist entry date.
    """
    st.info("Finding all songs with playlist data for the artist...")
    
    songs_with_playlist_data = db_manager.collections['song_audience'].find(
        {'artist_uuid': artist_uuid, 'playlists': {'$exists': True, '$ne': []}}
    )
    
    tasks_to_run = []
    
    with st.spinner("Checking for missing historical data for each song..."):
        for song_doc in songs_with_playlist_data:
            song_uuid = song_doc.get("song_uuid")
            playlists = song_doc.get("playlists", [])
            
            if not playlists:
                continue

            # Find the earliest entry date for the song
            earliest_entry_date = min([datetime.fromisoformat(p['entryDate']).date() for p in playlists if p.get('entryDate')])
            
            # Define the target 90-day window before the entry
            target_end_date = earliest_entry_date - timedelta(days=1)
            target_start_date = target_end_date - timedelta(days=89)

            # Check if we already have data covering this period
            query_filter = {'song_uuid': song_uuid, 'platform': 'spotify'}
            min_db_date, _ = db_manager.get_timeseries_data_range('song_audience', query_filter)
            
            if not min_db_date or min_db_date.date() > target_start_date:
                # If we have no data, or our earliest data is after the target start, we need to fetch.
                tasks_to_run.append({'uuid': song_uuid, 'start': target_start_date, 'end': target_end_date})

    if not tasks_to_run:
        st.success("All historical song audience data appears to be up-to-date.")
        return

    st.info(f"Found {len(tasks_to_run)} songs missing historical data. Starting parallel fetch...")
    data_found = False
    progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {
            executor.submit(api_client.get_song_streaming_audience, task['uuid'], 'spotify', task['start'], task['end']): task
            for task in tasks_to_run
        }

        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if data and not data.get('error') and data.get('items'):
                    db_manager.store_song_audience_data(task['uuid'], data)
                    data_found = True
            except Exception as exc:
                st.warning(f"A data fetch task for song {task['uuid']} generated an exception: {exc}")
            
            completed_count += 1
            progress_bar.progress(completed_count / len(tasks_to_run), text=f"Completed {completed_count}/{len(tasks_to_run)} tasks...")
    
    progress_bar.empty()
    if data_found:
        get_song_audience_data.clear()
        get_full_song_data_from_db.clear()
        st.success("Song backfill process completed successfully. New historical data was stored.")
        st.rerun()
    else:
        st.info("The song backfill process ran, but the API did not return any new historical data.")


# --- Page UI ---
st.title("Backfill Audience Data")

if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to begin.")
    st.stop()

artist_name = st.session_state.get('artist_name', 'the selected artist')
st.markdown(f"### Current Artist: **{artist_name}**")
artist_uuid = st.session_state.artist_uuid

st.markdown("---")
st.subheader("Album & Track Backfill")
st.markdown("""
This tool checks all albums and their tracks for the artist. If any are completely missing audience data, it will fetch the data for the last 365 days.
""")
if st.button(f"Start Album & Track Backfill for {artist_name}", use_container_width=True):
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    run_album_backfill(artist_uuid, start_date, end_date)

st.markdown("---")
# --- NEW UI for Song Backfill ---
st.subheader("Song Historical Backfill (From Playlists)")
st.markdown("""
This tool finds songs that have been on playlists and checks if their historical data is missing. It will fetch audience data for the **90 days prior** to a song's earliest known playlist entry.
""")
if st.button(f"Start Song Historical Backfill for {artist_name}", use_container_width=True, type="primary"):
    run_song_playlist_backfill(artist_uuid)
