# app.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import (
    get_artist_details,
    get_album_details,
    get_artist_playlists_from_db,
    get_audience_data,
    get_popularity_data,
    get_streaming_audience_from_db,
    get_local_streaming_history_from_db
)
from streamlit_ui import (
    display_artist_metadata
)
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Home", layout="wide") 

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State ---
if 'artist_uuid' not in st.session_state:
    st.session_state.artist_uuid = None
if 'artist_name' not in st.session_state:
    st.session_state.artist_name = None
if 'show_metadata' not in st.session_state:
    st.session_state.show_metadata = {}

# --- Helper Functions ---

def fetch_and_store_all_data(artist_name):
    """Fetches all data for a new artist in parallel after getting the UUID."""
    st.info(f"Fetching full dataset for new artist '{artist_name}'...")
    
    with st.spinner("Searching for artist..."):
        artist_info = api_client.search_artist(artist_name)
        if 'error' in artist_info:
            st.error(f"API Error: {artist_info['error']}")
            return
        artist_uuid = artist_info.get('uuid')
        if not artist_uuid:
            st.error("Could not retrieve artist UUID.")
            return
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        with st.spinner("Fetching all artist data in parallel..."):
            future_static = executor.submit(api_client.get_artist_metadata, artist_uuid)
            future_secondary = executor.submit(api_client.fetch_secondary_artist_data, artist_uuid)
            future_demographic = executor.submit(api_client.get_local_audience, artist_uuid)
            
            start_date = date.today() - timedelta(days=365)
            end_date = date.today()
            future_audience = executor.submit(api_client.get_artist_audience, artist_uuid, 'spotify', start_date, end_date)
            future_popularity = executor.submit(api_client.get_artist_popularity, artist_uuid, 'spotify', start_date, end_date)
            future_streaming = executor.submit(api_client.get_artist_streaming_audience, artist_uuid, 'spotify', start_date, end_date)
            future_local_streaming = executor.submit(api_client.get_local_streaming_audience, artist_uuid, 'spotify', start_date, end_date)

            static_data = {'metadata': future_static.result()}
            db_manager.store_static_artist_data(artist_uuid, static_data)

            secondary_data = future_secondary.result()
            if 'error' not in secondary_data:
                db_manager.store_secondary_artist_data(artist_uuid, secondary_data)

            demographic_data = {'local_audience': future_demographic.result()}
            db_manager.store_demographic_data(artist_uuid, demographic_data)

            time_series_data = {
                'audience': future_audience.result(),
                'popularity': future_popularity.result(),
                'streaming_audience': future_streaming.result(),
                'local_streaming_audience': future_local_streaming.result()
            }
            db_manager.store_timeseries_data(artist_uuid, time_series_data)

    st.success("Database updated successfully!")
    st.session_state.artist_uuid = artist_uuid
    st.session_state.artist_name = artist_name
    st.cache_data.clear()
    st.rerun()

# --- MODIFIED: Renamed and expanded the update function ---
def refresh_all_artist_data(artist_uuid):
    """
    Refreshes secondary data (albums, playlists) AND all primary time-series data 
    for an existing artist for the past 365 days.
    """
    with st.spinner("Updating all artist data (albums, playlists, charts)..."):
        secondary_data_updated = False
        timeseries_data_updated = False
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit secondary data fetch
            future_secondary = executor.submit(api_client.fetch_secondary_artist_data, artist_uuid)
            
            # Submit all time-series fetches
            start_date = date.today() - timedelta(days=365)
            end_date = date.today()
            future_audience = executor.submit(api_client.get_artist_audience, artist_uuid, 'spotify', start_date, end_date)
            future_popularity = executor.submit(api_client.get_artist_popularity, artist_uuid, 'spotify', start_date, end_date)
            future_streaming = executor.submit(api_client.get_artist_streaming_audience, artist_uuid, 'spotify', start_date, end_date)
            future_local_streaming = executor.submit(api_client.get_local_streaming_audience, artist_uuid, 'spotify', start_date, end_date)
            
            # Process secondary data
            secondary_data = future_secondary.result()
            if 'error' not in secondary_data:
                db_manager.store_secondary_artist_data(artist_uuid, secondary_data)
                secondary_data_updated = True
            else:
                st.warning(f"Could not update secondary data: {secondary_data['error']}")
            
            # Process time-series data
            time_series_data = {
                'audience': future_audience.result(),
                'popularity': future_popularity.result(),
                'streaming_audience': future_streaming.result(),
                'local_streaming_audience': future_local_streaming.result()
            }
            
            # Check if any new time-series data was actually fetched
            if any(data.get('items') for data in time_series_data.values() if data and 'error' not in data):
                db_manager.store_timeseries_data(artist_uuid, time_series_data)
                timeseries_data_updated = True

        if secondary_data_updated or timeseries_data_updated:
            # Clear all relevant caches
            get_artist_details.clear()
            get_album_details.clear()
            get_artist_playlists_from_db.clear()
            get_audience_data.clear()
            get_popularity_data.clear()
            get_streaming_audience_from_db.clear()
            get_local_streaming_history_from_db.clear()
            
            st.success("Artist info and chart data updated.")
            st.rerun()
        else:
            st.info("No new data was found to update.")


# --- Main App UI ---
st.title("Soundcharts Artist Database")

# --- NEW: Clear Cache Button ---
with st.sidebar:
    st.write("---")
    st.write("**App Controls**")
    if st.button("Clear Cache & Rerun App"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared! The app will now fetch fresh data.")
        st.rerun()

artist_name_input = st.text_input("Search for an Artist", placeholder="e.g., Daft Punk")

if st.button("Find Artist", use_container_width=True, type="primary"):
    if not artist_name_input:
        st.warning("Please enter an artist name.")
        st.session_state.artist_uuid = None
        st.session_state.artist_name = None
    else:
        with st.spinner(f"Searching for '{artist_name_input}'..."):
            existing_artist = db_manager.search_artist_by_name(artist_name_input)
        
        if existing_artist:
            st.success(f"Found '{artist_name_input}' in the database.")
            st.session_state.artist_uuid = existing_artist.get('artist_uuid')
            metadata_obj = existing_artist.get('object', existing_artist)
            st.session_state.artist_name = metadata_obj.get('name', "Unknown Artist")
        else:
            st.info(f"'{artist_name_input}' not found locally. Fetching from API...")
            fetch_and_store_all_data(artist_name_input)

st.markdown("---")

# --- UI Display Area ---
if st.session_state.artist_uuid:
    
    artist_details = get_artist_details(db_manager, st.session_state.artist_uuid)
    
    if artist_details and artist_details.get("metadata"):
        display_artist_metadata(artist_details.get('metadata'))
        
        # --- MODIFIED: This button now calls the new comprehensive refresh function ---
        if st.button(f"Update All Artist Data (Info & Charts)", use_container_width=True, help="Refreshes album, playlist, and all time-series chart data from the API for the last year."):
            refresh_all_artist_data(st.session_state.artist_uuid)

        st.info("Select a page from the sidebar to view detailed charts, albums, playlists, and demographics.")

    else:
        st.warning(f"Could not load details for artist UUID: {st.session_state.artist_uuid}.")
        st.info("Try searching for the artist again.")
else:
    st.info("Search for an artist to view their data.")
