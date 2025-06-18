# app.py (Corrected Workflow)

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Soundcharts Analytics", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}"); st.stop()

# --- Session State ---
if 'artist_uuid' not in st.session_state: st.session_state.artist_uuid = None
if 'show_metadata' not in st.session_state: st.session_state.show_metadata = {}

# --- Helper Functions ---

def fetch_and_store_all_data(artist_name):
    """Fetches all data for a new artist and stores it using the split DB functions."""
    st.info(f"Fetching full dataset for new artist '{artist_name}'...")
    
    with st.spinner("Fetching primary artist data..."):
        static_data = api_client.fetch_static_artist_data(artist_name)
        if 'error' in static_data:
            st.error(f"API Error: {static_data['error']}"); return
        artist_uuid = static_data.get('artist_uuid')
        if not artist_uuid:
             st.error("Could not retrieve artist UUID."); return
        # --- Store static data ---
        db_manager.store_static_artist_data(artist_uuid, static_data)

    with st.spinner("Fetching secondary data (albums, playlists)..."):
        secondary_data = api_client.fetch_secondary_artist_data(artist_uuid)
        if 'error' not in secondary_data:
             # --- Store secondary data ---
             db_manager.store_secondary_artist_data(artist_uuid, secondary_data)
        else:
             st.warning(f"Could not fetch secondary data: {secondary_data['error']}")
    
    # --- Fetch and store time-series data for a default range ---
    start_date = date.today() - timedelta(days=90)
    end_date = date.today()
    with st.spinner("Fetching time-series data..."):
        time_series_data = {
            'audience': api_client.get_artist_audience(artist_uuid, 'spotify', start_date, end_date),
            'popularity': api_client.get_artist_popularity(artist_uuid, 'spotify', start_date, end_date),
            'streaming_audience': api_client.get_artist_streaming_audience(artist_uuid, 'spotify', start_date, end_date)
        }
        # --- Store time-series data ---
        db_manager.store_timeseries_data(artist_uuid, time_series_data)

    st.success("Database updated successfully!")
    st.session_state.artist_uuid = artist_uuid
    st.cache_data.clear()
    st.rerun()

def update_static_data(artist_uuid):
    """Refreshes just the secondary data (albums, playlists) for an existing artist."""
    with st.spinner("Updating album and playlist info..."):
        secondary_data = api_client.fetch_secondary_artist_data(artist_uuid)
        if 'error' not in secondary_data:
            db_manager.store_secondary_artist_data(artist_uuid, secondary_data)
            get_artist_details.clear()
            get_album_details.clear()
            get_artist_playlists_from_db.clear()
            st.success("Artist info updated.")
            st.rerun()
        else:
            st.warning(f"Could not update secondary data: {secondary_data['error']}")

def update_timeseries_data(artist_uuid, start_date, end_date):
    """Intelligently fetches and appends only missing time-series data."""
    # (This logic remains largely the same, but now calls the dedicated store function)
    with st.spinner("Checking and updating time-series data..."):
        # This part could be refactored to fetch in parallel if desired
        time_series_to_fetch = {
            'audience': api_client.get_artist_audience,
            'popularity': api_client.get_artist_popularity,
            'streaming_audience': api_client.get_artist_streaming_audience
        }
        full_ts_data = {}
        for name, func in time_series_to_fetch.items():
            new_data = func(artist_uuid, 'spotify', start_date, end_date)
            if 'error' not in new_data:
                full_ts_data[name] = new_data

        db_manager.store_timeseries_data(artist_uuid, full_ts_data)
    st.cache_data.clear()
    st.success("Chart data updated.")
    st.rerun()


# --- Main App UI ---
st.title("Soundcharts Artist Database")
st.write("Enter an artist's name. The app will search your local database first. If the artist isn't found, it will fetch their data automatically.")

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
            fetch_and_store_all_data(artist_name_input)

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
                update_static_data(st.session_state.artist_uuid)

            st.markdown("### Time-Series Data")
            start_date_filter = st.date_input("Chart Start Date", date.today() - timedelta(days=30))
            end_date_filter = st.date_input("Chart End Date", date.today())
            
            if st.button("Get/Update Chart Data", use_container_width=True, type="primary"):
                update_timeseries_data(st.session_state.artist_uuid, start_date_filter, end_date_filter)

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
            display_playlists(api_client, db_manager, playlist_details) # Pass both clients
    else:
        st.warning(f"Could not load details for artist UUID: {st.session_state.artist_uuid}.")

else:
    st.info("Search for an artist to view their data.")
