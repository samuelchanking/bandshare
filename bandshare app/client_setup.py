# client_setup.py

import streamlit as st
from database_manager import DatabaseManager
from soundcharts_client import SoundchartsAPIClient

@st.cache_resource
def initialize_clients(config):
    """Initializes and caches the API and database manager clients."""
    api_client = SoundchartsAPIClient(app_id=config.APP_ID, api_key=config.API_KEY)
    db_manager = DatabaseManager(mongo_uri=config.MONGO_URI, db_name=config.DB_NAME)
    return api_client, db_manager