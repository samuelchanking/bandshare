# app.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients,
    get_artist_details,
    get_album_details,
    get_artist_playlists_from_db # Added
)
from streamlit_ui import (
    display_artist_metadata, 
    display_album_and_tracks,
    display_playlists # Added
)

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Soundcharts Artist DB", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State ---
if 'selected_artist_uuid' not in st.session_state: 
    st.session_state.selected_artist_uuid = None
if 'show_metadata' not in st.session_state: 
    st.session_state.show_metadata = {}

# --- Main App ---
st.title("Soundcharts Artist Database")
st.write("Enter an artist's name. The app will search your local database first. If the artist isn't found, it will fetch their data from the Soundcharts API and save it.")

# --- Search Workflow ---
artist_name_input = st.text_input("Artist Name", placeholder="e.g., Daft Punk")

if st.button("Find Artist", use_container_width=True, type="primary"):
    if not artist_name_input:
        st.warning("Please enter an artist name.")
        st.session_state.selected_artist_uuid = None
    else:
        with st.spinner(f"Searching for '{artist_name_input}' in the local database..."):
            existing_artist = db_manager.search_artist_by_name(artist_name_input)
        
        if existing_artist:
            st.success(f"Found '{artist_name_input}' in the local database.")
            st.session_state.selected_artist_uuid = existing_artist.get('artist_uuid')
        else:
            st.info(f"'{artist_name_input}' not found locally. Fetching from Soundcharts...")
            with st.spinner(f"Fetching data for {artist_name_input}..."):
                fetched_data = api_client.fetch_full_artist_data(artist_name_input)
            
            if 'error' in fetched_data:
                st.error(f"API Error: {fetched_data['error']}")
                st.session_state.selected_artist_uuid = None
            else:
                st.success("Data fetched successfully. Now saving to the database...")
                with st.spinner("Saving new artist..."):
                    result = db_manager.store_artist_data(fetched_data)
                
                if result.get('status') == 'success':
                    st.success(f"Successfully added '{artist_name_input}' to the database!")
                    st.session_state.selected_artist_uuid = fetched_data.get('artist_uuid')
                    get_artist_details.clear()
                    get_album_details.clear()
                    st.rerun()
                else:
                    st.error(f"Database Error: {result.get('message')}")
                    st.session_state.selected_artist_uuid = None

st.markdown("---")


# --- UI: Display Area for the Selected Artist ---
if st.session_state.selected_artist_uuid:
    left_col, right_col = st.columns(2, gap="large")
    
    # Fetch all data for the selected artist
    artist_details = get_artist_details(db_manager, st.session_state.selected_artist_uuid)
    playlist_details = get_artist_playlists_from_db(db_manager, st.session_state.selected_artist_uuid)


    if artist_details:
        artist_name = artist_details["metadata"].get("object", artist_details["metadata"]).get("name", "Unknown Artist")
        with left_col:
            st.subheader(f"Data for: {artist_name}")
            display_artist_metadata(artist_details.get("metadata"))
            
            # Display playlists below artist metadata
            st.markdown("---")
            display_playlists(playlist_details)

        with right_col:
            st.subheader("Stored Albums")
            for album in artist_details.get("albums", []):
                album_uuid = album.get("album_uuid")
                album_details = get_album_details(db_manager, album_uuid)
                album_name = album.get('object', album).get('name', 'Unknown Album')
                with st.expander(f"💿 {album_name}"):
                    tracklist_data = album_details.get("tracklist") if album_details else None
                    # Call the simplified display function
                    display_album_and_tracks(db_manager, album, tracklist_data)
else:
    st.info("Search for an artist to view their data.")
