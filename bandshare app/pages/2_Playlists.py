# pages/2_Playlists.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import initialize_clients, get_artist_playlists_from_db
from streamlit_ui import display_playlists

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Playlists", layout="wide")

try:
    _, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their playlist features.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Playlist Features for {artist_name}")

playlist_details = get_artist_playlists_from_db(db_manager, artist_uuid)
display_playlists(db_manager, playlist_details)