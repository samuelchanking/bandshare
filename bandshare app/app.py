# app.py (with Popularity Chart)

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients, 
    get_audience_data, 
    get_popularity_data, # Added
    get_artist_details,
    get_album_details,
    get_artist_playlists_from_db
)
from streamlit_ui import (
    display_artist_metadata, 
    display_audience_chart, 
    display_popularity_chart, # Added
    display_album_and_tracks,
    display_playlists
)
from datetime import date, timedelta, datetime

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Soundcharts Analytics", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State ---
if 'artist_uuid' not in st.session_state: st.session_state.artist_uuid = None
if 'show_metadata' not in st.session_state: st.session_state.show_metadata = {}


# --- Main Logic Handler ---
def process_data_request(artist_name, start_date, end_date):
    """The core logic for fetching and updating all artist data."""
    # Step 1: Find or create the main artist document
    with st.spinner(f"Finding '{artist_name}'..."):
        artist_doc = db_manager.search_artist_by_name(artist_name)
        if not artist_doc:
            static_data = api_client.fetch_static_artist_data(artist_name)
            if 'error' in static_data:
                st.error(f"API Error: {static_data['error']}"); return
            db_manager.store_static_artist_data(static_data)
            artist_uuid = static_data['artist_uuid']
        else:
            artist_uuid = artist_doc['artist_uuid']
        
        st.session_state.artist_uuid = artist_uuid

    # Step 2: Fetch and store secondary data (albums, playlists)
    with st.spinner("Fetching latest album and playlist data..."):
        secondary_data = api_client.fetch_secondary_artist_data(artist_uuid)
        if 'error' not in secondary_data:
            db_manager.store_secondary_artist_data(artist_uuid, secondary_data)
            get_artist_details.clear()
            get_album_details.clear()
            get_artist_playlists_from_db.clear()
        else:
            st.warning("Could not fetch album/playlist data.")

    # Step 3: Intelligently fetch Audience data
    with st.spinner("Checking local audience data..."):
        platform = "spotify"
        audience_filter = {'artist_uuid': artist_uuid, 'platform': platform}
        db_start, db_end = db_manager.get_timeseries_data_range('audience', audience_filter)
        
        req_start_dt = datetime.combine(start_date, datetime.min.time())
        req_end_dt = datetime.combine(end_date, datetime.min.time())
        
        gaps_to_fetch = []
        if db_start is None:
            gaps_to_fetch.append((start_date, end_date))
        else:
            if req_start_dt < db_start.replace(tzinfo=None):
                gaps_to_fetch.append((start_date, (db_start - timedelta(days=1)).date()))
            if req_end_dt > db_end.replace(tzinfo=None):
                gaps_to_fetch.append(((db_end + timedelta(days=1)).date(), end_date))
        
    if gaps_to_fetch:
        with st.spinner(f"Fetching {len(gaps_to_fetch)} gap(s) of audience data..."):
            for start, end in gaps_to_fetch:
                new_data = api_client.get_artist_audience(artist_uuid, platform, start, end)
                if 'items' in new_data:
                    db_manager.append_timeseries_data('audience', audience_filter, new_data['items'])
            get_audience_data.clear()
            
    # --- ADDED: Step 4: Intelligently fetch Popularity data ---
    with st.spinner("Checking local popularity data..."):
        source = "spotify"
        popularity_filter = {'artist_uuid': artist_uuid, 'source': source}
        db_start, db_end = db_manager.get_timeseries_data_range('popularity', popularity_filter)

        gaps_to_fetch_pop = []
        if db_start is None:
            gaps_to_fetch_pop.append((start_date, end_date))
        else:
            if req_start_dt < db_start.replace(tzinfo=None):
                gaps_to_fetch_pop.append((start_date, (db_start - timedelta(days=1)).date()))
            if req_end_dt > db_end.replace(tzinfo=None):
                gaps_to_fetch_pop.append(((db_end + timedelta(days=1)).date(), end_date))

    if gaps_to_fetch_pop:
        with st.spinner(f"Fetching {len(gaps_to_fetch_pop)} gap(s) of popularity data..."):
            for start, end in gaps_to_fetch_pop:
                new_data = api_client.get_artist_popularity(artist_uuid, source, start, end)
                if 'items' in new_data:
                    db_manager.append_timeseries_data('popularity', popularity_filter, new_data['items'])
            get_popularity_data.clear()
    
    st.success("Data is up to date!")


# --- Main App UI ---
st.title("Soundcharts Artist Analytics")
st.write("Enter an artist's name and date range to fetch and display their data.")

search_col1, search_col2, search_col3 = st.columns(3)
artist_name_input = search_col1.text_input("Artist Name", placeholder="e.g., Daft Punk")
start_date_input = search_col2.date_input("Start Date", value=date.today() - timedelta(days=90))
end_date_input = search_col3.date_input("End Date", value=date.today())

if st.button("Get Analytics", use_container_width=True, type="primary"):
    if artist_name_input:
        process_data_request(artist_name_input, start_date_input, end_date_input)
    else:
        st.warning("Please enter an artist name.")

# --- UI Display Area ---
if st.session_state.artist_uuid:
    st.markdown("---")
    
    # Fetch all data for display
    artist_details = get_artist_details(db_manager, st.session_state.artist_uuid)
    audience_data = get_audience_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_input, end_date_input)
    popularity_data = get_popularity_data(db_manager, st.session_state.artist_uuid, "spotify", start_date_input, end_date_input)
    playlist_data = get_artist_playlists_from_db(db_manager, st.session_state.artist_uuid)
    
    # Main layout
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        display_artist_metadata(artist_details.get('metadata'))
        st.markdown("---")
        display_audience_chart(audience_data)
        st.markdown("---")
        display_popularity_chart(popularity_data) # Added

    with right_col:
        st.subheader("Albums")
        for album in artist_details.get("albums", []):
            album_uuid = album.get("album_uuid")
            album_details = get_album_details(db_manager, album_uuid)
            album_name = album.get('album_metadata', album).get('name', 'Unknown Album')
            with st.expander(f"💿 {album_name}"):
                tracklist_data = album_details.get("tracklist") if album_details else None
                display_album_and_tracks(db_manager, album, tracklist_data)
        
        st.markdown("---")
        display_playlists(playlist_data)
