# pages/3_Tracks.py

import streamlit as st
import config
import pandas as pd
from pymongo.errors import ConnectionFailure
import plotly.express as px
from client_setup import initialize_clients
from streamlit_caching import get_all_songs_for_artist_from_db, get_playlist_song_uuids_for_artist, get_song_details, get_song_audience_data, get_song_popularity_data, get_playlists_for_song
from streamlit_ui import display_tracks_grid, display_retention_chart, display_track_details_page
from datetime import datetime
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Tracks", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State for Track Selection ---
if 'selected_track_uuid' not in st.session_state:
    st.session_state.selected_track_uuid = None

# --- Helper Function for this page ---
def fetch_and_store_all_tracks(api_client, db_manager, artist_uuid):
    """Fetches all songs for an artist from the API and stores them."""
    with st.spinner("Finding all songs for artist from API... This may take a moment."):
        songs_list = api_client.get_artist_songs(artist_uuid)
        
        if not songs_list:
            st.warning("API did not return any songs for this artist.")
            return

        for song_data in songs_list:
            if song_uuid := song_data.get('uuid'):
                db_manager.store_song_metadata(song_uuid, song_data)
        
        get_all_songs_for_artist_from_db.clear()
        st.success(f"Successfully updated {len(songs_list)} tracks.")
        st.rerun()


# --- Rest of the original 3_Tracks.py remains unchanged ---
# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their tracks.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"All Tracks for {artist_name}")

if st.button("Find & Update All Tracks for Artist", type="primary"):
    fetch_and_store_all_tracks(api_client, db_manager, artist_uuid)

st.markdown("---")

# --- Sorting and Filtering controls ---
col1, col2 = st.columns(2)
with col1:
    sort_option = st.selectbox(
        "Sort tracks by:",
        ("Default", "Release Date (Newest First)", "Release Date (Oldest First)"),
        key='track_sort_option'
    )
with col2:
    filter_on_playlist = st.checkbox(
        "Show only songs featured on playlists",
        key='playlist_filter_toggle'
    )

# --- Data Caching & Filtering Logic ---
with st.spinner("Loading all songs from database..."):
    all_songs = get_all_songs_for_artist_from_db(db_manager, artist_uuid)

    if filter_on_playlist and all_songs:
        playlist_song_uuids = get_playlist_song_uuids_for_artist(db_manager, artist_uuid)
        all_songs = [
            song for song in all_songs
            if song.get('uuid') in playlist_song_uuids
        ]

# --- Sorting Logic ---
if sort_option != "Default" and all_songs:
    def get_release_date(song):
        release_date_str = song.get('releaseDate')
        if release_date_str:
            try:
                return datetime.fromisoformat(release_date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                return datetime.min
        return datetime.min

    reverse_sort = (sort_option == "Release Date (Newest First)")
    all_songs = sorted(all_songs, key=get_release_date, reverse=reverse_sort)

# --- View Toggle Logic ---
if st.session_state.selected_track_uuid:
    display_track_details_page(api_client, db_manager, st.session_state.selected_track_uuid)
else:
    if not all_songs:
        st.info("No songs found in the database for this artist or matching the filter. Click the update button to fetch them.")
    else:
        display_tracks_grid(all_songs)
