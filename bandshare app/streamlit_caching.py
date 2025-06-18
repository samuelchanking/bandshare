# streamlit_caching.py

import streamlit as st
from database_manager import DatabaseManager
from soundcharts_client import SoundchartsAPIClient

@st.cache_resource
def initialize_clients(config):
    api_client = SoundchartsAPIClient(app_id=config.APP_ID, api_key=config.API_KEY)
    db_manager = DatabaseManager(mongo_uri=config.MONGO_URI, db_name=config.DB_NAME)
    return api_client, db_manager

@st.cache_data
def get_all_artists(_db_manager): # This can be removed if not used in app.py
    return list(_db_manager.collections['artists'].find(
        {'$or': [{'object.name': {'$exists': True}}, {'name': {'$exists': True}}], 'artist_uuid': {'$exists': True}},
        {'object.name': 1, 'name': 1, 'artist_uuid': 1, '_id': 0}
    ))

@st.cache_data
def get_artist_details(_db_manager, artist_uuid: str):
    if not artist_uuid: return None
    artist_metadata = _db_manager.collections['artists'].find_one({'artist_uuid': artist_uuid})
    albums = list(_db_manager.collections['albums'].find({'artist_uuid': artist_uuid}))
    return {"metadata": artist_metadata, "albums": albums}

@st.cache_data
def get_album_details(_db_manager, album_uuid: str):
    if not album_uuid: return None
    return {"tracklist": _db_manager.collections['tracklists'].find_one({'album_uuid': album_uuid})}

@st.cache_data
def get_song_details(_db_manager, song_uuid: str):
    if not song_uuid: return None
    if 'songs' in _db_manager.collections:
        if song := _db_manager.collections['songs'].find_one({'song_uuid': song_uuid}):
            return song
    return {}

@st.cache_data
def get_artist_playlists_from_db(_db_manager, artist_uuid: str):
    """Fetches playlist data from the database for a given artist."""
    if not artist_uuid: return None
    if 'playlists' in _db_manager.collections:
        return list(_db_manager.collections['playlists'].find({'artist_uuid': artist_uuid}))
    return []

@st.cache_data
def get_audience_data(_db_manager, artist_uuid: str, platform: str, start_date, end_date):
    """Gets audience data for a given date range from the database."""
    if not all([artist_uuid, platform, start_date, end_date]):
        return []
    query_filter = {'artist_uuid': artist_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('audience', query_filter, start_date, end_date)

@st.cache_data
def get_popularity_data(_db_manager, artist_uuid: str, source: str, start_date, end_date):
    """Gets popularity data for a given date range from the database."""
    if not all([artist_uuid, source, start_date, end_date]):
        return []
    query_filter = {'artist_uuid': artist_uuid, 'source': source}
    return _db_manager.get_timeseries_data_for_display('popularity', query_filter, start_date, end_date)

@st.cache_data
def get_streaming_audience_from_db(_db_manager, artist_uuid: str, platform: str, start_date, end_date):
    """Gets streaming audience data for a given date range from the database."""
    if not all([artist_uuid, platform, start_date, end_date]): return []
    query_filter = {'artist_uuid': artist_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('streaming_audience', query_filter, start_date, end_date)

# MODIFIED: This function now gets data for a specific song-playlist combo
@st.cache_data
def get_playlist_song_streaming_from_db(_db_manager, song_uuid: str, playlist_uuid: str):
    """Fetches stored time-series streaming data for a single song on a specific playlist."""
    if not song_uuid or not playlist_uuid: return None
    # Use the compound key to find the specific document
    return _db_manager.collections['songs_audience'].find_one({
        'song_uuid': song_uuid,
        'playlist_uuid': playlist_uuid
    })
