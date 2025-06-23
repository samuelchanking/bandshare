# pages/4_Demographics.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import initialize_clients, get_local_audience_from_db
from streamlit_ui import display_demographics

# --- Page Configuration & Initialization ---
st.set_page_config(page_title="Artist Demographics", layout="wide")

try:
    _, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their demographics.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Demographics for {artist_name}")

local_audience = get_local_audience_from_db(db_manager, artist_uuid, "instagram")
display_demographics(local_audience)