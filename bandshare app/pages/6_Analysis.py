# 6_Analysis.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import get_typed_playlists_from_db
from streamlit_ui import display_typed_playlists, display_global_playlist_tracks

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Playlist Analysis", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State ---
if 'selected_playlist_uuid' not in st.session_state:
    st.session_state.selected_playlist_uuid = None
if 'selected_global_song_uuid' not in st.session_state:
    st.session_state.selected_global_song_uuid = None
# This is still needed to store results from the per-song analysis
if 'global_spike_analysis_results' not in st.session_state:
    st.session_state.global_spike_analysis_results = {}

# --- Page UI ---
st.title("Global Playlist Analysis")
st.markdown("""
This page shows top playlists on Spotify that feature songs by your tracked artists. 
Click 'Details' on any playlist to see its full tracklist.
""")
st.markdown("---")


# --- Display Logic ---
if st.session_state.get('selected_playlist_uuid'):
    all_playlists = get_typed_playlists_from_db(db_manager, 'editorial', 'spotify') + get_typed_playlists_from_db(db_manager, 'algorithmic', 'spotify')
    selected_playlist_meta = next((p for p in all_playlists if p['uuid'] == st.session_state.selected_playlist_uuid), None)
    playlist_name = selected_playlist_meta.get('name', 'N/A') if selected_playlist_meta else 'N/A'
    display_global_playlist_tracks(db_manager, st.session_state.selected_playlist_uuid, playlist_name)

else:
    # Main page view - displays playlists directly without buttons
    st.subheader("Display Options")
    sort_cols = st.columns([2, 1, 1])
    with sort_cols[0]:
        pass
    with sort_cols[1]:
        sort_key = st.selectbox("Sort by", ["Subscriber Count", "Alphabetical"], key="playlist_sort_key")
    with sort_cols[2]:
        sort_order_label = st.radio("Order", ["High to Low", "Low to High"], key="playlist_sort_order", horizontal=True)

    st.markdown("---")

    with st.spinner("Loading playlists from database..."):
        editorial_playlists = get_typed_playlists_from_db(db_manager, 'editorial', 'spotify')
        algorithmic_playlists = get_typed_playlists_from_db(db_manager, 'algorithmic', 'spotify')

    reverse_order = (sort_order_label == "High to Low")
    if sort_key == "Alphabetical":
        sort_lambda = lambda p: (p.get('name') is None, (p.get('name') or "").lower())
    else: # Subscriber Count
        sort_lambda = lambda p: (p.get('latestSubscriberCount') is None, p.get('latestSubscriberCount', 0))

    sorted_editorial = sorted(editorial_playlists, key=sort_lambda, reverse=reverse_order)
    sorted_algorithmic = sorted(algorithmic_playlists, key=sort_lambda, reverse=reverse_order)

    display_typed_playlists(sorted_editorial, "Top Editorial Playlists on Spotify")
    st.markdown("---")
    display_typed_playlists(sorted_algorithmic, "Top Algorithmic Playlists on Spotify")
