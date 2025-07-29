# pages/6_Backfill_Data.py

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
    get_all_songs_for_artist_from_db,
    get_song_popularity_data
)
from analysis_tools import adjust_cumulative_history
from datetime import date, timedelta, datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Backfill Data", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- MODIFIED: Backfill Logic with Automated Song Ranges ---
def run_backfill(artist_uuid, start_date, end_date, debug_mode=False):
    """
    Fetches and stores missing data. For songs on playlists, it uses a dynamic date range.
    For albums and other songs, it uses the provided fixed date range.
    """
    st.info("Gathering all albums and tracks for the artist...")
    
    # --- Album Tasks (Standard 365-day range) ---
    artist_data = get_artist_details(db_manager, artist_uuid)
    albums = artist_data.get("albums", [])
    tasks_to_run = []
    
    with st.spinner("Checking for missing album data..."):
        for album in albums:
            if album_uuid := album.get("album_uuid"):
                query_filter_album = {'album_uuid': album_uuid, 'platform': 'spotify'}
                min_db, max_db = db_manager.get_timeseries_data_range('album_audience', query_filter_album)
                
                if not min_db or not max_db or max_db.date() < end_date or min_db.date() > start_date:
                    tasks_to_run.append({'type': 'album', 'uuid': album_uuid, 'start': start_date, 'end': end_date})

    # --- Song Tasks (Automated range based on playlists) ---
    with st.spinner("Checking for missing song data with dynamic ranges..."):
        all_songs = get_all_songs_for_artist_from_db(db_manager, artist_uuid)
        playlist_entries = list(db_manager.collections['songs_playlists'].find({'artist_uuid': artist_uuid}))

        # Group playlist entry dates by song_uuid for efficient lookup
        song_playlist_dates = defaultdict(list)
        for entry in playlist_entries:
            song_uuid = entry.get('song', {}).get('uuid')
            if entry_date_str := entry.get('entryDate'):
                try:
                    song_playlist_dates[song_uuid].append(datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date())
                except ValueError:
                    continue
        
        for song in all_songs:
            song_uuid = song['uuid']
            start_fetch, end_fetch = start_date, end_date # Default range

            # If song has playlist entries, calculate its specific date range
            if song_uuid in song_playlist_dates:
                min_entry = min(song_playlist_dates[song_uuid])
                max_entry = max(song_playlist_dates[song_uuid])
                start_fetch = min_entry - timedelta(days=90)
                end_fetch = max_entry + timedelta(days=90)

            # Check for missing audience data in the calculated range
            query_song_aud = {'song_uuid': song_uuid, 'platform': 'spotify'}
            min_aud, max_aud = db_manager.get_timeseries_data_range('song_audience', query_song_aud)
            if not min_aud or not max_aud or max_aud.date() < end_fetch or min_aud.date() > start_fetch:
                tasks_to_run.append({'type': 'song_audience', 'uuid': song_uuid, 'start': start_fetch, 'end': end_fetch})

            # Check for missing popularity data in the calculated range
            query_song_pop = {'song_uuid': song_uuid, 'platform': 'spotify'}
            min_pop, max_pop = db_manager.get_timeseries_data_range('song_popularity', query_song_pop)
            if not min_pop or not max_pop or max_pop.date() < end_fetch or min_pop.date() > start_fetch:
                tasks_to_run.append({'type': 'song_popularity', 'uuid': song_uuid, 'start': start_fetch, 'end': end_fetch})

    if not tasks_to_run:
        st.success("All album and track audience/popularity data appears to be populated.")
        return

    st.info(f"Found {len(tasks_to_run)} items missing data. Starting parallel fetch...")
    data_found = False
    progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {}
        for task in tasks_to_run:
            if task['type'] == 'album':
                future = executor.submit(api_client.get_album_audience, task['uuid'], 'spotify', task['start'], task['end'])
            elif task['type'] == 'song_audience':
                future = executor.submit(api_client.get_song_streaming_audience, task['uuid'], 'spotify', task['start'], task['end'])
            elif task['type'] == 'song_popularity':
                future = executor.submit(api_client.get_song_popularity, task['uuid'], 'spotify', task['start'], task['end'])
            future_to_task[future] = task

        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()

                if debug_mode:
                    st.markdown("---")
                    st.write(f"**Debug: Fetched data for {task['type']} `{task['uuid']}`**")
                    st.json(data)
                
                if data and not data.get('error') and data.get('items'):
                    if task['type'] == 'album':
                        db_manager.store_album_audience_data(task['uuid'], data)
                        data_found = True
                    elif task['type'] == 'song_audience':
                        raw_history = []
                        for item in data['items']:
                            if 'date' in item:
                                plots = item.get('plots', [])
                                if plots:
                                    max_val = max(p.get('value', 0) for p in plots)
                                    raw_history.append({'date': item['date'], 'value': max_val})
                        
                        max_values = {}
                        for item in raw_history:
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
                            adjusted_items = adjust_cumulative_history(cleaned_items)
                            db_manager.store_song_audience_data(task['uuid'], {'history': adjusted_items, 'platform': 'spotify'})
                            data_found = True
                        else:
                            st.warning(f"No cleaned items for song audience {task['uuid']}, skipping storage.")
                    elif task['type'] == 'song_popularity':
                        db_manager.store_song_popularity_data(task['uuid'], data)
                        data_found = True
            except Exception as exc:
                st.warning(f"A data fetch task for {task['type']} {task['uuid']} generated an exception: {exc}")
            
            completed_count += 1
            progress_bar.progress(completed_count / len(tasks_to_run), text=f"Completed {completed_count}/{len(tasks_to_run)} tasks...")
    
    progress_bar.empty()
    if data_found:
        get_album_audience_data.clear()
        get_song_audience_data.clear()
        get_song_popularity_data.clear()
        st.success("Backfill process completed successfully. New data was stored.")
        st.rerun()
    else:
        st.info("The backfill process ran, but the API did not return any new data for the missing items.")


# --- NEW: Function to Fetch and Store All Songs Audience Data ---
def fetch_and_store_all_songs_audience_and_popularity(api_client, db_manager, artist_uuid, start_date, end_date, debug_mode=False):
    """Fetches streaming audience and popularity data for all songs of an artist from the API and stores them, overwriting existing audience data."""
    with st.spinner("Gathering all tracks for the artist..."):
        all_songs = get_all_songs_for_artist_from_db(db_manager, artist_uuid)
        if not all_songs:
            st.warning("No songs found in the database for this artist.")
            return

    tasks_to_run = []
    for song in all_songs:
        song_uuid = song['uuid']
        tasks_to_run.append({'type': 'song_audience', 'uuid': song_uuid, 'start': start_date, 'end': end_date})
        tasks_to_run.append({'type': 'song_popularity', 'uuid': song_uuid, 'start': start_date, 'end': end_date})

    if not tasks_to_run:
        st.info("No tasks to run.")
        return

    st.info(f"Starting update for {len(all_songs)} songs' streaming audience and popularity data...")
    data_found = False
    progress_bar = st.progress(0, text=f"Submitting {len(tasks_to_run)} update tasks...")

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {}
        for task in tasks_to_run:
            if task['type'] == 'song_audience':
                future = executor.submit(api_client.get_song_streaming_audience, task['uuid'], 'spotify', task['start'], task['end'])
            elif task['type'] == 'song_popularity':
                future = executor.submit(api_client.get_song_popularity, task['uuid'], 'spotify', task['start'], task['end'])
            future_to_task[future] = task

        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if debug_mode:
                    st.markdown("---")
                    st.write(f"**Debug: Fetched data for {task['type']} `{task['uuid']}`**")
                    st.json(data)
                
                if data and not data.get('error') and data.get('items'):
                    if task['type'] == 'song_audience':
                        # Process new audience data without merging existing data
                        raw_history = []
                        for item in data['items']:
                            if 'date' in item:
                                plots = item.get('plots', [])
                                if plots:
                                    max_val = max(p.get('value', 0) for p in plots)
                                    raw_history.append({'date': item['date'], 'value': max_val})
                        
                        # Sort and clean new data only
                        max_values = {}
                        for item in raw_history:
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
                            adjusted_items = adjust_cumulative_history(cleaned_items)
                            db_manager.store_song_audience_data(task['uuid'], {'history': adjusted_items, 'platform': 'spotify'})
                        else:
                            st.warning(f"No cleaned items for song audience {task['uuid']}, skipping storage.")
                    elif task['type'] == 'song_popularity':
                        db_manager.store_song_popularity_data(task['uuid'], data)
                    data_found = True
            except Exception as exc:
                st.warning(f"A data fetch task for {task['type']} {task['uuid']} generated an exception: {exc}")
            
            completed_count += 1
            progress_bar.progress(completed_count / len(tasks_to_run), text=f"Completed {completed_count}/{len(tasks_to_run)} tasks...")
    
    progress_bar.empty()
    if data_found:
        get_song_audience_data.clear()
        get_song_popularity_data.clear()
        st.success("Update process completed successfully. New data was stored.")
        st.rerun()
    else:
        st.info("The update process ran, but the API did not return any new data.")



def run_playlist_audience_backfill(artist_uuid, debug_mode=False):
    st.info("Gathering all playlist entries for the artist from the definitive source...")
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
            if not playlist_uuid or not entry_date_str: continue
            try:
                entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date()
                required_start = entry_date - timedelta(days=90)
                required_end = entry_date + timedelta(days=90)
                query_filter = {'playlist_uuid': playlist_uuid}
                min_db, max_db = db_manager.get_timeseries_data_range('playlist_audience', query_filter)
                fetch_needed = False
                if not min_db or not max_db or max_db.date() < required_end or min_db.date() > required_start:
                    fetch_needed = True
                if fetch_needed:
                    tasks_to_run.append({'uuid': playlist_uuid, 'name': playlist_info.get('name', 'N/A'), 'start': required_start, 'end': required_end})
            except (ValueError, TypeError):
                continue
    final_tasks = {}
    for task in tasks_to_run:
        uuid = task['uuid']
        if uuid not in final_tasks: final_tasks[uuid] = task
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
        future_to_task = {executor.submit(api_client.get_playlist_audience, task['uuid'], task['start'], task['end']): task for task in tasks_to_run}
        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                data = future.result()
                if debug_mode:
                    st.markdown("---")
                    st.write(f"**Debug: Fetched data for playlist '{task['name']}' (`{task['uuid']}`)**")
                    st.json(data)
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
st.markdown("---")
debug_mode = st.checkbox("Enable Debug Mode", help="If checked, the raw JSON response from the API will be displayed for each backfill task.")
st.markdown("---")
end_date = date.today()
start_date = end_date - timedelta(days=1095)

# --- Album/Track Backfill Section ---
with st.container(border=True):
    st.subheader("Backfill Album & Track Data")
    st.markdown("""
    This process checks all albums and tracks for the artist. 
    - For **songs on playlists**, it downloads missing audience and popularity data for a dynamic range (+/- 90 days around the playlist entry dates).
    - For **all other songs and albums**, it downloads missing data for the past 3 years.
    """)
    if st.button(f"Start Album/Track Backfill for {artist_name}", use_container_width=True, type="primary"):
        run_backfill(artist_uuid, start_date, end_date, debug_mode)

# --- Playlist Backfill Section ---
with st.container(border=True):
    st.subheader("Backfill Playlist Audience Data")
    st.markdown("""
    This process finds every playlist a song has been on and downloads the playlist's follower history for the 90 days before and after the song's entry date, if that data is missing.
    """)
    if st.button(f"Start Playlist Audience Backfill for {artist_name}", use_container_width=True, type="primary"):
        run_playlist_audience_backfill(artist_uuid, debug_mode)
        
# --- NEW: Tracks Streaming Audience Update Section (Add after the Playlist Backfill Section) ---
with st.container(border=True):
    st.subheader("Update All Tracks Streaming Audience and Popularity Data")
    st.markdown("""
    This process fetches the streaming audience and popularity data for all songs associated with the artist from the API for the past 3 years 
    and updates the database using upsert logic to ensure the latest data is stored.
    """)
    if st.button(f"Update Tracks Performance for {artist_name}", use_container_width=True, type="primary"):
        fetch_and_store_all_songs_audience_and_popularity(api_client, db_manager, artist_uuid, start_date, end_date, debug_mode)
