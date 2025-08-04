# streamlit_caching.py

import re
import requests
import streamlit as st
from database_manager import DatabaseManager
from soundcharts_client import SoundchartsAPIClient
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

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

@st.cache_data
def get_playlist_song_uuids_for_artist(_db_manager, artist_uuid: str):
    """Fetches a set of all song UUIDs for an artist that appear in the 'songs_playlists' collection."""
    if not artist_uuid:
        return set()
    
    playlist_entries = _db_manager.collections['songs_playlists'].find(
        {'artist_uuid': artist_uuid},
        {'song.uuid': 1, 'song_uuid': 1, '_id': 0} # Projection to get only necessary fields
    )
    
    song_uuids = set()
    for item in playlist_entries:
        # Handle both possible structures for song UUID
        song_uuid = item.get('song_uuid') or item.get('song', {}).get('uuid')
        if song_uuid:
            song_uuids.add(song_uuid)
            
    return song_uuids

@st.cache_data
def get_playlists_for_song(_db_manager, song_uuid: str):
    """
    Fetches all playlist entries for a given song from the 'songs_playlists' collection.
    """
    if not song_uuid: return []
    
    # CORRECTED: Use dot notation 'song.uuid' to query the nested field.
    return list(_db_manager.collections['songs_playlists'].find(
        {'song.uuid': song_uuid}
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
    """Gets song audience data for a given date range from the database, returned per identifier."""
    if not all([song_uuid, platform, start_date, end_date]):
        return {}
    start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "+00:00"
    end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "+00:00"
    result = {}
    cursor = _db_manager.collections['song_audience'].find({'song_uuid': song_uuid, 'platform': platform})
    for doc in cursor:
        identifier = doc.get('identifier')
        history = doc.get('history', [])
        filtered = [item for item in history if start_iso <= item['date'] <= end_iso]
        if filtered:
            result[identifier] = sorted(filtered, key=lambda x: x['date'])
    return result

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

@st.cache_data
def get_tracks_for_playlist(_db_manager, playlist_uuid: str):
    """Fetches all tracks for a given playlist from the 'global_song' collection."""
    if not playlist_uuid: return []
    return list(_db_manager.collections['global_song'].find({'playlist_uuid': playlist_uuid}))


@st.cache_data
def get_playlist_placements_for_songs(_db_manager, song_uuids: list):
    """
    For a given list of song UUIDs, finds all their placements in the 'songs_playlists' collection.
    """
    if not song_uuids: return []
    # Find all entries where the song_uuid is in our list of songs
    placements_cursor = _db_manager.collections['songs_playlists'].find(
        {'song_uuid': {'$in': song_uuids}}
    )
    # Group the results by song_uuid for easy lookup
    placements_by_song = {}
    for placement in placements_cursor:
        s_uuid = placement.get('song_uuid')
        if s_uuid not in placements_by_song:
            placements_by_song[s_uuid] = []
        placements_by_song[s_uuid].append(placement.get('playlist', {}).get('name', 'N/A'))
    
    return placements_by_song

@st.cache_data
def get_global_song_audience_data(_db_manager, song_uuid: str, start_date, end_date):
    """Gets song audience data from the 'global_song_audience' collection."""
    if not all([song_uuid, start_date, end_date]): return []
    query_filter = {'song_uuid': song_uuid}
    # Uses the generic display function pointing to the new collection
    return _db_manager.get_timeseries_data_for_display('global_song_audience', query_filter, start_date, end_date)

@st.cache_data
def get_typed_playlists_from_db(_db_manager, playlist_type: str, platform: str):
    """
    Fetches typed playlists from the 'typed_playlists' collection using a case-insensitive search.
    """
    if not playlist_type:
        return []

    # Use a case-insensitive regex to match the 'type' field.
    query = {
        'type': re.compile(f'^{playlist_type}$', re.IGNORECASE),
        'platform': platform
    }

    if 'typed_playlists' in _db_manager.collections:
        return list(_db_manager.collections['typed_playlists'].find(query).sort('latestSubscriberCount', -1))

    return []

@st.cache_data
def get_song_popularity_data(_db_manager, song_uuid: str, platform: str, start_date, end_date):
    """Gets song popularity data for a given date range from the database."""
    if not all([song_uuid, platform, start_date, end_date]): return []
    query_filter = {'song_uuid': song_uuid, 'platform': platform}
    # Assumes storage in a 'song_popularity' collection.
    return _db_manager.get_timeseries_data_for_display('song_popularity', query_filter, start_date, end_date)

@st.cache_data
def get_artist_events(_db_manager, artist_uuid: str) -> List[Dict[str, Any]]:
    """
    Fetches and caches all events for a given artist from the database.
    """
    if not artist_uuid:
        return []
    return _db_manager.get_artist_events_from_db(artist_uuid)

@st.cache_data
def get_cum_song_data(_db_manager, artist_uuid, start_date_filter, end_date_filter, _all_dates):
    songs = get_all_songs_for_artist_from_db(_db_manager, artist_uuid)
    song_streams = []
    for song in songs:
        song_uuid = song['uuid']
        song_name = song.get('name', 'Unknown')
        data = get_song_audience_data(_db_manager, song_uuid, 'spotify', start_date_filter, end_date_filter)
        for entry in data:
            date_val = entry.get('date')
            plots = entry.get('plots', [])
            value = plots[0].get('value') if plots else None
            if date_val and value is not None:
                song_streams.append({'date': pd.to_datetime(date_val), 'song': song_name, 'cumulative': value})
    if not song_streams:
        return None
    df_songs = pd.DataFrame(song_streams)
    df_pivot = df_songs.pivot_table(index='date', columns='song', values='cumulative', fill_value=0)
    df_pivot = df_pivot.reindex(_all_dates, fill_value=0)
    return df_pivot


# Add this to streamlit_caching.py or wherever caching functions are defined

# Updated caching function to sort by date ascending (earliest first)
@st.cache_data
def get_playlist_tracklists(_db_manager, playlist_uuid: str):
    """
    Fetches all tracklist snapshots for a given playlist from the 'playlist_tracklists' collection,
    sorted by date in ascending order (earliest first).
    """
    if not playlist_uuid:
        return []
    
    if 'playlist_tracklists' in _db_manager.collections:
        return list(_db_manager.collections['playlist_tracklists'].find(
            {'playlist_uuid': playlist_uuid}
        ).sort('date', 1))  # Changed to ascending sort
    
    return []

@st.cache_data
def get_song_playlist_presence_intervals(_db_manager, song_uuid: str):
    """
    Aggregates presence intervals across all playlists for the song.
    """
    intervals = []
    playlist_entries = get_playlists_for_song(_db_manager, song_uuid)
    
    for entry in playlist_entries:
        playlist_uuid = entry.get('playlist', {}).get('uuid')
        if not playlist_uuid:
            continue
        
        tracklists = get_playlist_tracklists(_db_manager, playlist_uuid)
        
        history = []
        for tl in tracklists:
            date_str = tl.get('date')
            position = None
            for track in tl.get('tracks', []):
                if track.get('song', {}).get('uuid') == song_uuid:
                    position = track.get('position')
                    break
            history.append({'date': pd.to_datetime(date_str), 'position': position})
        
        start = None
        for h in history:
            if h['position'] is not None:
                if start is None:
                    start = h['date']
            else:
                if start is not None:
                    intervals.append({'start': start, 'end': h['date']})
                    start = None
        if start is not None:
            # Assume still present up to last snapshot
            intervals.append({'start': start, 'end': history[-1]['date']})
    
    # Merge overlapping intervals
    if intervals:
        intervals.sort(key=lambda i: i['start'])
        merged = [intervals[0]]
        for current in intervals[1:]:
            previous = merged[-1]
            if previous['end'] >= current['start']:
                previous['end'] = max(previous['end'], current['end'])
            else:
                merged.append(current)
        return merged
    return []

