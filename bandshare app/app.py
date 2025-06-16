# app.py (Intelligent Date Handling)

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients, 
    get_artist_details, get_album_details, 
    get_artist_playlists_from_db,
    get_audience_data, get_popularity_data,
    get_streaming_audience_from_db
)
from streamlit_ui import (
    display_artist_metadata, display_album_and_tracks, 
    display_playlists, display_audience_chart, 
    display_popularity_chart,
    display_streaming_audience_chart
)
from datetime import date, timedelta, datetime

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Soundcharts Analytics", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}"); st.stop()

# --- Session State ---
if 'artist_uuid' not in st.session_state: st.session_state.artist_uuid = None
if 'show_metadata' not in st.session_state: st.session_state.show_metadata = {}
if 'db_start_date' not in st.session_state: st.session_state.db_start_date = None
if 'db_end_date' not in st.session_state: st.session_state.db_end_date = None

# --- Helper Functions ---
def fetch_and_store_initial_data(artist_name):
    """Fetches all data for a new artist and stores it."""
    st.info(f"Fetching full dataset for new artist '{artist_name}'...")
    start_date = date.today() - timedelta(days=90)
    end_date = date.today()
    
    with st.spinner("Fetching data..."):
        fetched_data = api_client.fetch_full_artist_data(artist_name, start_date, end_date)

    if 'error' in fetched_data:
        st.error(f"API Error: {fetched_data['error']}"); return

    with st.spinner("Saving data to database..."):
        result = db_manager.store_artist_data(fetched_data)
    
    if result.get('status') == 'success':
        st.success("Database updated successfully!")
        st.session_state.artist_uuid = fetched_data.get('artist_uuid')
        st.cache_data.clear()
        st.rerun()
    else:
        st.error(f"Database Error: {result.get('message')}")

def update_static_data(artist_name, artist_uuid):
    """Fetches and updates only the non-time-series data (metadata, albums, playlists)."""
    with st.spinner("Fetching latest album and playlist data..."):
        secondary_data = api_client.fetch_secondary_artist_data(artist_uuid)
    if 'error' in secondary_data:
        st.warning(f"Could not fetch secondary data: {secondary_data['error']}"); return
    
    db_manager.store_secondary_artist_data(artist_uuid, secondary_data)
    get_artist_details.clear()
    get_album_details.clear()
    get_artist_playlists_from_db.clear()
    st.success(f"Updated static info for {artist_name}.")
    st.rerun()

def fetch_missing_timeseries_data(artist_uuid, start_date, end_date):
    """Intelligently fetches only the missing time-series data."""
    time_series_configs = [
        {'name': 'audience', 'platform_key': 'platform', 'api_func': api_client.get_artist_audience, 'cache_clear_func': get_audience_data},
        {'name': 'popularity', 'platform_key': 'source', 'api_func': api_client.get_artist_popularity, 'cache_clear_func': get_popularity_data},
        {'name': 'streaming_audience', 'platform_key': 'platform', 'api_func': api_client.get_artist_streaming_audience, 'cache_clear_func': get_streaming_audience_from_db}
    ]
    for config in time_series_configs:
        db_start, db_end = db_manager.get_timeseries_data_range(config['name'], {'artist_uuid': artist_uuid, config['platform_key']: 'spotify'})
        req_start_dt = datetime.combine(start_date, datetime.min.time())
        req_end_dt = datetime.combine(end_date, datetime.min.time())
        
        gaps_to_fetch = []
        if db_start is None:
            gaps_to_fetch.append((start_date, end_date))
        else:
            if req_start_dt < db_start.replace(tzinfo=None): gaps_to_fetch.append((start_date, (db_start - timedelta(days=1)).date()))
            if req_end_dt > db_end.replace(tzinfo=None): gaps_to_fetch.append(((db_end + timedelta(days=1)).date(), end_date))
        
        if gaps_to_fetch:
            with st.spinner(f"Fetching {len(gaps_to_fetch)} gap(s) of {config['name']} data..."):
                for start, end in gaps_to_fetch:
                    new_data = config['api_func'](artist_uuid, 'spotify', start, end)
                    if 'items' in new_data:
                        db_manager.append_timeseries_data(config['name'], {'artist_uuid': artist_uuid, config['platform_key']: 'spotify'}, new_data['items'])
                config['cache_clear_func'].clear()
    st.success("Time-series data updated!")
    st.rerun()

# --- Main App UI ---
st.title("Soundcharts Artist Database")
st.write("Enter an artist's name. The app will first search your local database.")

artist_name_input = st.text_input("Search for an Artist", placeholder="e.g., Daft Punk")

if st.button("Find Artist", use_container_width=True, type="primary"):
    if not artist_name_input:
        st.warning("Please enter an artist name.")
        st.session_state.artist_uuid = None
    else:
        with st.spinner(f"Searching for '{artist_name_input}'..."):
            existing_artist = db_manager.search_artist_by_name(artist_name_input)
        if existing_artist:
            st.success(f"Found '{artist_name_input}' in the database.")
            st.session_state.artist_uuid = existing_artist.get('artist_uuid')
        else:
            st.info(f"'{artist_name_input}' not found locally. Fetching from API...")
            fetch_and_store_initial_data(artist_name_input)

st.markdown("---")

# --- UI Display Area ---
if st.session_state.artist_uuid:
    artist_details = get_artist_details(db_manager, st.session_state.artist_uuid)
    
    if artist_details and artist_details.get("metadata"):
        artist_name = artist_details["metadata"].get("object", artist_details["metadata"]).get("name", "Unknown Artist")
        
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            display_artist_metadata(artist_details.get('metadata'))
            if st.button(f"Update Artist Info (Albums/Playlists)", use_container_width=True):
                update_static_data(artist_name, st.session_state.artist_uuid)

            st.markdown("### Time-Series Data")
            
            # Get the date range of stored data to set as default
            db_start, db_end = db_manager.get_timeseries_data_range('audience', {'artist_uuid': st.session_state.artist_uuid, 'platform': 'spotify'})
            
            default_start = db_start.date() if db_start else date.today() - timedelta(days=30)
            default_end = db_end.date() if db_end else date.today()

            start_date_filter = st.date_input("Chart Start Date", value=default_start)
            end_date_filter = st.date_input("Chart End Date", value=default_end)

            # Logic to show the update button only when needed
            needs_update = False
            if db_start and start_date_filter < db_start.date(): needs_update = True
            if db_end and end_date_filter > db_end.date(): needs_update = True
            if not db_start: needs_update = True # Always need to fetch if no data exists

            if needs_update:
                if st.button("Fetch New Chart Data for this Range", use_container_width=True, type="primary"):
                    fetch_missing_timeseries_data(st.session_state.artist_uuid, start_date_filter, end_date_filter)

            audience_data = get_audience_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)
            popularity_data = get_popularity_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)
            streaming_data = get_streaming_audience_from_db(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)

            display_audience_chart(audience_data)
            display_popularity_chart(popularity_data)
            display_streaming_audience_chart(streaming_data)

        with right_col:
            st.subheader("Albums")
            for album in artist_details.get("albums", []):
                album_uuid = album.get("album_uuid")
                album_details_from_db = get_album_details(db_manager, album_uuid)
                album_name = album.get("album_metadata", {}).get("name", "Unknown Album")
                with st.expander(f"💿 {album_name}"):
                    tracklist_data = album_details_from_db.get("tracklist") if album_details_from_db else None
                    display_album_and_tracks(db_manager, album, tracklist_data)
            
            st.markdown("---")
            playlist_details = get_artist_playlists_from_db(db_manager, st.session_state.artist_uuid)
            display_playlists(playlist_details)
    else:
        st.warning(f"Could not load details for artist UUID: {st.session_state.artist_uuid}. The data might be missing or corrupted.")
else:
    st.info("Search for an artist to view their data.")
