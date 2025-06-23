# pages/1_Albums.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import initialize_clients, get_artist_details, get_album_details, download_image_bytes
from streamlit_ui import display_album_and_tracks
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Albums", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State for Album Selection ---
if 'selected_album_uuid' not in st.session_state:
    st.session_state.selected_album_uuid = None

# --- Helper to chunk data for grid layout ---
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their albums.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Albums for {artist_name}")

artist_details = get_artist_details(db_manager, artist_uuid)
all_albums = artist_details.get("albums", [])

if not all_albums:
    st.warning("No album data found for this artist.")
    st.stop()

# --- View Toggle Logic ---

# DETAILED VIEW (if an album has been selected)
if st.session_state.selected_album_uuid:
    selected_album_data = next((album for album in all_albums if album.get("album_uuid") == st.session_state.selected_album_uuid), None)
    
    if selected_album_data:
        album_details_from_db = get_album_details(db_manager, st.session_state.selected_album_uuid)
        tracklist_data = album_details_from_db.get("tracklist") if album_details_from_db else None

        album_name = selected_album_data.get("album_metadata", {}).get("name", "Unknown Album")
        st.subheader(f"Details for: {album_name}")

        if st.button("⬅️ Back to all albums"):
            st.session_state.selected_album_uuid = None
            st.rerun()

        display_album_and_tracks(db_manager, selected_album_data, tracklist_data)
    else:
        st.error("Could not find the selected album details. Returning to album list.")
        st.session_state.selected_album_uuid = None
        st.rerun()

# GRID VIEW (default view)
else:
    st.subheader("Select an album to view details")
    st.markdown("---")

    # --- Step 1: Gather all unique image URLs ---
    urls_to_download = set()
    for album in all_albums:
        metadata = album.get("album_metadata", {})
        nested_object = metadata.get("object", {})
        image_url = metadata.get("imageUrl") or nested_object.get("imageUrl")
        if image_url:
            urls_to_download.add(image_url)

    # --- Step 2: Download images in parallel using the cached function ---
    prefetched_images = {}
    with st.spinner("Loading album art..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_url = {executor.submit(download_image_bytes, url): url for url in urls_to_download}
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    prefetched_images[url] = future.result()
                except Exception:
                    prefetched_images[url] = None
    
    # --- Step 3: Display the grid using the prefetched images ---
    for album_chunk in grouper(all_albums, 4):
        cols = st.columns(4)
        for i, album in enumerate(album_chunk):
            if album:
                with cols[i]:
                    album_uuid = album.get("album_uuid")
                    metadata = album.get("album_metadata", {})
                    album_name = metadata.get("name", "Unknown Album")
                    
                    nested_object = metadata.get("object", {})
                    image_url = metadata.get("imageUrl") or nested_object.get("imageUrl")
                    
                    image_bytes = prefetched_images.get(image_url)
                    
                    if image_bytes:
                        st.image(image_bytes, use_container_width=True)
                    else:
                        st.image("https://i.imgur.com/3gMbdA5.png", use_container_width=True)

                    st.caption(album_name)
                    
                    if st.button("Details", key=f"btn_details_{album_uuid}", use_container_width=True):
                        st.session_state.selected_album_uuid = album_uuid
                        st.rerun()
