# app.py (Corrected Workflow)

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients, 
    get_artist_details, get_album_details, 
    get_artist_playlists_from_db,
    get_audience_data, get_popularity_data,
    get_streaming_audience_from_db,
    get_local_audience_from_db,
    get_local_streaming_history_from_db
)
from streamlit_ui import (
    display_artist_metadata, display_album_and_tracks, 
    display_playlists, display_audience_chart, 
    display_popularity_chart,
    display_streaming_audience_chart,
    display_demographics,
    display_local_streaming_plots
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
    """Fetches all data for a new artist in parallel after getting the UUID."""
    st.info(f"Fetching full dataset for new artist '{artist_name}'...")
    
    # --- Step 1: Sequential call to get the essential UUID ---
    with st.spinner("Searching for artist..."):
        # Simplified initial search to just get the UUID
        artist_info = api_client.search_artist(artist_name)
        if 'error' in artist_info:
            st.error(f"API Error: {artist_info['error']}"); return
        artist_uuid = artist_info.get('uuid')
        if not artist_uuid:
             st.error("Could not retrieve artist UUID."); return
    
    # --- Step 2: Parallel calls for everything else ---
    with ThreadPoolExecutor(max_workers=20) as executor:
        with st.spinner("Fetching all artist data in parallel..."):
            # Create futures for all independent API calls
            future_static = executor.submit(api_client.get_artist_metadata, artist_uuid)
            future_secondary = executor.submit(api_client.fetch_secondary_artist_data, artist_uuid)
            future_demographic = executor.submit(api_client.get_local_audience, artist_uuid)
            
            start_date = date.today() - timedelta(days=90)
            end_date = date.today()
            future_audience = executor.submit(api_client.get_artist_audience, artist_uuid, 'spotify', start_date, end_date)
            future_popularity = executor.submit(api_client.get_artist_popularity, artist_uuid, 'spotify', start_date, end_date)
            future_streaming = executor.submit(api_client.get_artist_streaming_audience, artist_uuid, 'spotify', start_date, end_date)
            future_local_streaming = executor.submit(api_client.get_local_streaming_audience, artist_uuid, 'spotify', start_date, end_date)

            # --- Step 3: Store results as they complete ---
            # Store static data
            static_data = {'metadata': future_static.result()}
            db_manager.store_static_artist_data(artist_uuid, static_data)

            # Store secondary data
            secondary_data = future_secondary.result()
            if 'error' not in secondary_data:
                db_manager.store_secondary_artist_data(artist_uuid, secondary_data)

            # Store demographic data
            demographic_data = {'local_audience': future_demographic.result()}
            db_manager.store_demographic_data(artist_uuid, demographic_data)

            # Store time-series data
            time_series_data = {
                'audience': future_audience.result(),
                'popularity': future_popularity.result(),
                'streaming_audience': future_streaming.result(),
                'local_streaming_audience': future_local_streaming.result()
            }
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
    """Fetches and appends missing time-series data in parallel."""
    with st.spinner("Checking and updating time-series data..."):
        tasks = {
            'audience': lambda: api_client.get_artist_audience(artist_uuid, 'spotify', start_date, end_date),
            'popularity': lambda: api_client.get_artist_popularity(artist_uuid, 'spotify', start_date, end_date),
            'streaming_audience': lambda: api_client.get_artist_streaming_audience(artist_uuid, 'spotify', start_date, end_date),
            'local_streaming_audience': lambda: api_client.get_local_streaming_audience(artist_uuid, 'spotify', start_date, end_date)
        }
        
        full_ts_data = {}
        with ThreadPoolExecutor() as executor:
            future_to_name = {executor.submit(func): name for name, func in tasks.items()}
            
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    data = future.result()
                    if 'error' not in data:
                        full_ts_data[name] = data
                except Exception as exc:
                    st.warning(f'{name} data fetching generated an exception: {exc}')

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

            local_audience = get_local_audience_from_db(db_manager, st.session_state.artist_uuid, "instagram")
            display_demographics(local_audience)


            st.markdown("### Time-Series Data")
            start_date_filter = st.date_input("Chart Start Date", date.today() - timedelta(days=90))
            end_date_filter = st.date_input("Chart End Date", date.today())
            
            if st.button("Get/Update Chart Data", use_container_width=True, type="primary"):
                update_timeseries_data(st.session_state.artist_uuid, start_date_filter, end_date_filter)

            audience_data = get_audience_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)
            popularity_data = get_popularity_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)
            streaming_data = get_streaming_audience_from_db(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)
            local_streaming_data = get_local_streaming_history_from_db(db_manager, st.session_state.artist_uuid, "spotify", start_date_filter, end_date_filter)

            display_audience_chart(audience_data)
            display_popularity_chart(popularity_data)
            display_streaming_audience_chart(streaming_data)
            display_local_streaming_plots(local_streaming_data)

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
            display_playlists(db_manager, playlist_details)
    else:
        st.warning(f"Could not load details for artist UUID: {st.session_state.artist_uuid}.")

else:
    st.info("Search for an artist to view their data.")
