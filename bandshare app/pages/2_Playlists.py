# pages/2_Playlists.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import get_artist_playlists_from_db
from streamlit_ui import display_playlists

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Playlists", layout="wide")

try:
    _, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- MODIFIED: Session State Management ---
# Standardized keys to match the UI components
if 'selected_playlist_uuid' not in st.session_state:
    st.session_state.selected_playlist_uuid = None
if 'selected_song_uuid' not in st.session_state:
    st.session_state.selected_song_uuid = None

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their playlist features.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Playlist Features for {artist_name}")

# This function now correctly pulls from the 'songs_playlists' collection
playlist_items = get_artist_playlists_from_db(db_manager, artist_uuid)

# The main display function handles all the logic
display_playlists(db_manager, playlist_items)
