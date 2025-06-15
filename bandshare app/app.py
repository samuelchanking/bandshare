# app.py (Database-First Workflow with Date Range)

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients, 
    get_artist_details, 
    get_album_details, 
    get_artist_playlists_from_db,
    # Restore time-series caching functions
    get_audience_data, 
    get_popularity_data
)
from streamlit_ui import (
    display_artist_metadata, 
    display_album_and_tracks, 
    display_playlists,
    # Restore time-series display functions
    display_audience_chart, 
    display_popularity_chart
)
from datetime import date, timedelta

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Soundcharts Artist DB", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State ---
if 'artist_uuid' not in st.session_state: st.session_state.artist_uuid = None
if 'show_metadata' not in st.session_state: st.session_state.show_metadata = {}

# --- Helper function for fetching and storing data ---
def fetch_and_store_full_data(artist_name, start_date, end_date):
    """Fetches all data from the API for a given date range and stores it."""
    st.info(f"Fetching latest data for '{artist_name}' from Soundcharts...")
    with st.spinner("Fetching..."):
        # The fetch_full_artist_data function gets all data points at once
        fetched_data = api_client.fetch_full_artist_data(artist_name, start_date, end_date)
    
    if 'error' in fetched_data:
        st.error(f"API Error: {fetched_data['error']}"); return

    with st.spinner("Saving updated data to database..."):
        # The store_artist_data function handles the replacement logic
        result = db_manager.store_artist_data(fetched_data)
        if result.get('status') == 'success':
            st.success("Database updated successfully!")
            # Set the UUID and clear caches to load the new data from the DB
            st.session_state.artist_uuid = fetched_data.get('artist_uuid')
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Database Error: {result.get('message')}")


# --- Main App UI ---
st.title("Soundcharts Artist Database")
st.write("Enter an artist's name and a date range. The app will check your local database, then fetch from the API if needed.")

# --- Search Inputs with Date Range ---
col1, col2, col3 = st.columns(3)
artist_name_input = col1.text_input("Artist Name", placeholder="e.g., Daft Punk")
start_date_input = col2.date_input("Start Date", date.today() - timedelta(days=90))
end_date_input = col3.date_input("End Date", date.today())


if st.button("Find Artist", use_container_width=True, type="primary"):
    if not artist_name_input:
        st.warning("Please enter an artist name.")
        st.session_state.artist_uuid = None
    else:
        # --- Database-First Workflow ---
        with st.spinner(f"Searching for '{artist_name_input}' in your database..."):
            existing_artist = db_manager.search_artist_by_name(artist_name_input)
        
        if existing_artist:
            st.success(f"Found '{artist_name_input}' in the database. Displaying stored data for the selected date range.")
            st.session_state.artist_uuid = existing_artist.get('artist_uuid')
        else:
            st.info(f"'{artist_name_input}' not found locally. Fetching from API...")
            # If not found, fetch everything for the specified range and store it.
            fetch_and_store_full_data(artist_name_input, start_date_input, end_date_input)

st.markdown("---")

# --- UI Display Area ---
if st.session_state.artist_uuid:
    
    # Fetch all data for display, using the date range for time-series data
    artist_details = get_artist_details(db_manager, st.session_state.artist_uuid)
    playlist_details = get_artist_playlists_from_db(db_manager, st.session_state.artist_uuid)
    audience_data = get_audience_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_input, end_date_input)
    popularity_data = get_popularity_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_input, end_date_input)

    if artist_details and artist_details.get("metadata"):
        artist_name = artist_details["metadata"].get("object", artist_details["metadata"]).get("name", "Unknown Artist")
        
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            display_artist_metadata(artist_details.get('metadata'))
            # Add an update button to manually refresh all data for this artist
            if st.button(f"Refresh All Data for {artist_name}", use_container_width=True):
                # When refreshing, we use the current date range from the UI
                fetch_and_store_full_data(artist_name, start_date_input, end_date_input)

            st.markdown("---")
            # Display time-series charts
            display_audience_chart(audience_data)
            st.markdown("---")
            display_popularity_chart(popularity_data)

        with right_col:
            st.subheader("Albums")
            # Album display is not affected by the date range after being stored
            for album in artist_details.get("albums", []):
                album_uuid = album.get("album_uuid")
                album_details_from_db = get_album_details(db_manager, album_uuid)
                album_name = album.get("album_metadata", {}).get("name", "Unknown Album")
                with st.expander(f"💿 {album_name}"):
                    tracklist_data = album_details_from_db.get("tracklist") if album_details_from_db else None
                    display_album_and_tracks(db_manager, album, tracklist_data)
            
            st.markdown("---")
            display_playlists(playlist_details)
    else:
        st.warning(f"Could not load details for artist UUID: {st.session_state.artist_uuid}. The data might be missing or corrupted.")

else:
    st.info("Search for an artist to view their data.")
