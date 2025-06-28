# pages/6_Analysis.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from client_setup import initialize_clients
from streamlit_caching import get_typed_playlists_from_db
from streamlit_ui import display_typed_playlists
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Playlist Analysis", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Session State to control UI visibility ---
if 'show_playlists' not in st.session_state:
    st.session_state.show_playlists = False

# --- Data Fetching and Storing (Unchanged) ---
def fetch_and_store_typed_playlists(platform='spotify'):
    """
    Fetches both editorial and algorithmic playlists in parallel, adds the necessary
    type and platform fields, and then stores them.
    """
    playlist_types = ['editorial', 'algorithmic']
    all_playlists = []
    
    with st.spinner(f"Fetching {', '.join(playlist_types)} playlists for {platform}..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_type = {
                executor.submit(api_client.get_playlists_by_type, platform, p_type): p_type
                for p_type in playlist_types
            }
            
            for future in as_completed(future_to_type):
                p_type = future_to_type[future]
                try:
                    data = future.result()
                    if data:
                        st.write(f"Found {len(data)} '{p_type}' playlists.")
                        for playlist_item in data:
                            playlist_item['type'] = p_type
                            playlist_item['platform'] = platform
                        all_playlists.extend(data)
                    else:
                        st.warning(f"No playlists found for type '{p_type}'.")
                except Exception as exc:
                    st.error(f"Fetching '{p_type}' playlists generated an exception: {exc}")
    
    if all_playlists:
        with st.spinner("Storing playlists in the database..."):
            db_manager.store_typed_playlists(all_playlists)
        st.success(f"Successfully fetched and stored {len(all_playlists)} playlists.")
        st.cache_data.clear()
        st.session_state.show_playlists = True # Automatically show after fetching
        st.rerun()
    else:
        st.info("No new playlist data was fetched.")


# --- Page UI ---
st.title("Global Playlist Analysis")
st.markdown("""
This page shows the top playlists on Spotify, categorized by type, based on the `playlist/by-type` endpoint. 
The data is fetched from the Soundcharts API and sorted by the total number of subscribers.
""")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    if st.button("Fetch/Update Global Playlists", use_container_width=True, type="primary"):
        fetch_and_store_typed_playlists('spotify')
with col2:
    button_text = "Hide Current Playlists" if st.session_state.show_playlists else "Show Current Playlists"
    if st.button(button_text, use_container_width=True):
        st.session_state.show_playlists = not st.session_state.show_playlists
        st.rerun()

st.markdown("---")

if st.session_state.show_playlists:
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

    # --- MODIFIED: More robust sorting logic ---
    reverse_order = (sort_order_label == "High to Low")
    if sort_key == "Alphabetical":
        sort_lambda = lambda p: (p.get('name') is None, (p.get('name') or "").lower())
    else: # Subscriber Count
        sort_lambda = lambda p: (p.get('latestSubscriberCount') is None, p.get('latestSubscriberCount', 0))

    sorted_editorial = sorted(editorial_playlists, key=sort_lambda, reverse=reverse_order)
    sorted_algorithmic = sorted(algorithmic_playlists, key=sort_lambda, reverse=reverse_order)
    # --- END MODIFICATION ---

    display_typed_playlists(sorted_editorial, "Top Editorial Playlists on Spotify")
    st.markdown("---")
    display_typed_playlists(sorted_algorithmic, "Top Algorithmic Playlists on Spotify")
else:
    st.info("Click 'Show Current Playlists' to view the data stored in the database.")