# streamlit_caching.py

import requests
import streamlit as st
from database_manager import DatabaseManager
from soundcharts_client import SoundchartsAPIClient

@st.cache_data
def get_all_artists(_db_manager): # This can be removed if not used in app.py
    return list(_db_manager.collections['artists'].find(
        {'$or': [{'object.name': {'$exists': True}}, {'name': {'$exists': True}}], 'artist_uuid': {'$exists': True}},
        {'object.name': 1, 'name': 1, 'artist_uuid': 1, '_id': 0}
    ))

@st.cache_data
def get_all_songs_for_artist_from_db(_db_manager, artist_uuid: str):
    """Fetches all songs (uuid and name) for a given artist from the database."""
    if not artist_uuid: return []
    return _db_manager.get_all_songs_for_artist(artist_uuid)

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

# --- MODIFIED: Made this function more robust ---
@st.cache_data
def get_song_details(_db_manager, song_uuid: str):
    """
    Fetches details for a single song from the database.
    Returns None if not found to prevent caching empty results incorrectly.
    """
    if not song_uuid: 
        return None
    if 'songs' in _db_manager.collections:
        song = _db_manager.collections['songs'].find_one({'song_uuid': song_uuid})
        return song  # Returns the document or None
    return None

# --- NEW FUNCTION ---
@st.cache_data
def get_playlists_for_song(_db_manager, song_uuid: str):
    """
    Fetches all playlist entries for a given song from the 'songs_playlists' collection.
    """
    if not song_uuid: return []
    # Find all documents for the song and sort by entry date, most recent first
    return list(_db_manager.collections['songs_playlists'].find(
        {'song_uuid': song_uuid}
    ).sort('entryDate', -1))


@st.cache_data
def get_artist_playlists_from_db(_db_manager, artist_uuid: str):
    """Fetches ALL playlist data from the new songs_playlists collection for an artist."""
    if not artist_uuid: return []
    return list(_db_manager.collections['songs_playlists'].find({'artist_uuid': artist_uuid}))


@st.cache_data
def get_local_audience_from_db(_db_manager, artist_uuid: str, platform: str):
    """Gets local audience data from the database."""
    if not all([artist_uuid, platform]):
        return None
    return _db_manager.collections['demographic_followers'].find_one({'artist_uuid': artist_uuid, 'platform': platform})

@st.cache_data
def get_local_streaming_history_from_db(_db_manager, artist_uuid: str, platform: str, start_date, end_date):
    """Gets local streaming history data for a given date range from the database."""
    if not all([artist_uuid, platform, start_date, end_date]): return []
    query_filter = {'artist_uuid': artist_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('local_streaming_history', query_filter, start_date, end_date)

@st.cache_data
def get_album_audience_data(_db_manager, album_uuid: str, platform: str, start_date, end_date):
    """Gets album audience data for a given date range from the database."""
    if not all([album_uuid, platform, start_date, end_date]): return []
    query_filter = {'album_uuid': album_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('album_audience', query_filter, start_date, end_date)

@st.cache_data
def get_playlist_audience_data(_db_manager, playlist_uuid: str, start_date, end_date):
    """Gets and caches playlist audience data for a given date range from the DATABASE."""
    if not all([playlist_uuid, start_date, end_date]): return []
    query_filter = {'playlist_uuid': playlist_uuid}
    return _db_manager.get_timeseries_data_for_display('playlist_audience', query_filter, start_date, end_date)


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
    query_filter = {'source': source, 'artist_uuid': artist_uuid}
    return _db_manager.get_timeseries_data_for_display('popularity', query_filter, start_date, end_date)

@st.cache_data
def get_streaming_audience_from_db(_db_manager, artist_uuid: str, platform: str, start_date, end_date):
    """Gets streaming audience data for a given date range from the database."""
    if not all([artist_uuid, platform, start_date, end_date]): return []
    query_filter = {'artist_uuid': artist_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('streaming_audience', query_filter, start_date, end_date)

@st.cache_data
def get_song_audience_data(_db_manager, song_uuid: str, platform: str, start_date, end_date):
    """Gets song audience data for a given date range from the database."""
    if not all([song_uuid, platform, start_date, end_date]): return []
    query_filter = {'song_uuid': song_uuid, 'platform': platform}
    return _db_manager.get_timeseries_data_for_display('song_audience', query_filter, start_date, end_date)

@st.cache_data
def get_full_song_data_from_db(_db_manager, song_uuid: str):
    """Fetches the complete, unified data document for a single song from 'song_audience'."""
    if not song_uuid: return None
    return _db_manager.collections['song_audience'].find_one({'song_uuid': song_uuid})

@st.cache_data
def get_typed_playlists_from_db(_db_manager, playlist_type: str, platform: str):
    """Fetches typed playlists from the database, sorted by subscriber count."""
    if not playlist_type: return []
    if 'typed_playlists' in _db_manager.collections:
        return list(_db_manager.collections['typed_playlists'].find(
            {'type': playlist_type, 'platform': platform}
        ).sort('latestSubscriberCount', -1)) # Sort descending
    return []
