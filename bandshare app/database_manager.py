# database_manager.py

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime

class DatabaseManager:
    """Manages all interactions with the MongoDB database."""

    def __init__(self, mongo_uri: str, db_name: str):
        """Initializes the database manager and connects to MongoDB."""
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ismaster')
            self.db = self.client[db_name]
            self.collections = {
                'artists': self.db['artists'],
                'albums': self.db['albums'],
                'tracklists': self.db['album_tracklist'],
                'songs': self.db['songs'],
                'playlists': self.db['playlists'],
                'audience': self.db['audience'],
                'popularity': self.db['popularity'],
                'streaming_audience': self.db['streaming_audience'],
                'songs_aggregate_audience': self.db['songs_aggregate_audience'],
                'demographic_followers': self.db['demographic_followers'],
                'local_streaming_history': self.db['local_streaming_history'],
                'album_audience': self.db['album_audience'],
                'song_audience': self.db['song_audience'],
            }
            # Create indexes for efficient queries
            self.collections['audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['popularity'].create_index([("artist_uuid", ASCENDING), ("source", ASCENDING)])
            self.collections['streaming_audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['demographic_followers'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['local_streaming_history'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['songs_aggregate_audience'].create_index([("song_uuid", ASCENDING)])
            self.collections['album_audience'].create_index([("album_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['song_audience'].create_index([("song_uuid", ASCENDING), ("platform", ASCENDING)])
        except ConnectionFailure as e:
            raise e

    def close_connection(self):
        """Closes the MongoDB connection."""
        self.client.close()
        
    def get_all_songs_for_artist(self, artist_uuid: str) -> List[Dict[str, str]]:
        """Retrieves all songs (UUID and name) for a given artist."""
        songs_cursor = self.collections['songs'].find(
            {'artist_uuid': artist_uuid},
            {'song_uuid': 1, 'object.name': 1, 'name': 1, '_id': 0}
        )
        songs = []
        for song in songs_cursor:
            # The song name can be in 'object.name' or 'name'
            name = song.get('object', {}).get('name') or song.get('name')
            if name and song.get('song_uuid'):
                songs.append({'song_uuid': song['song_uuid'], 'name': name})
        return songs

    def search_artist_by_name(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """Finds an artist by name in the local database."""
        if not artist_name: return None
        search_regex = re.compile(f"^{re.escape(artist_name)}$", re.IGNORECASE)
        return self.collections['artists'].find_one({'$or': [{'name': search_regex}, {'object.name': search_regex}]})

    def store_static_artist_data(self, artist_uuid: str, data: Dict[str, Any]) -> Dict[str, str]:
        """Upserts the main artist metadata document."""
        if 'error' in data: return {'status': 'error', 'message': data['error']}
        
        try:
            if 'metadata' in data:
                self.collections['artists'].update_one(
                    {'artist_uuid': artist_uuid},
                    {'$set': data['metadata']},
                    upsert=True
                )
            return {'status': 'success', 'message': 'Static data stored.'}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"DB error: {e}"}

    def store_secondary_artist_data(self, artist_uuid: str, data: Dict[str, Any]):
        """
        Stores or updates all secondary data, now only using aggregated song audience data.
        """
        try:
            if 'albums' in data and 'items' in data.get('albums', {}):
                all_album_metadata = data.get('album_metadata', {})
                for album_summary in data['albums']['items']:
                    if album_uuid_val := album_summary.get('uuid'):
                        combined_meta = album_summary | all_album_metadata.get(album_uuid_val, {})
                        album_doc = {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'album_metadata': combined_meta}
                        self.collections['albums'].update_one({'album_uuid': album_uuid_val}, {'$set': album_doc}, upsert=True)

            if 'tracklists' in data:
                for album_uuid_val, tracklist_data in data['tracklists'].items():
                    if 'error' not in tracklist_data:
                        doc_to_store = tracklist_data | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val}
                        self.collections['tracklists'].update_one({'album_uuid': album_uuid_val}, {'$set': doc_to_store}, upsert=True)
            
            if 'song_metadata' in data:
                for album_uuid_val, songs in data['song_metadata'].items():
                    for song_uuid, song_meta in songs.items():
                        if 'error' not in song_meta:
                            doc_to_store = song_meta | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'song_uuid': song_uuid}
                            self.collections['songs'].update_one({'song_uuid': song_uuid}, {'$set': doc_to_store}, upsert=True)

            if 'playlists' in data and 'items' in data.get('playlists', {}):
                for playlist_item in data['playlists']['items']:
                    if playlist_uuid := playlist_item.get('playlist', {}).get('uuid'):
                        document_to_store = playlist_item | {'artist_uuid': artist_uuid}
                        self.collections['playlists'].update_one({'artist_uuid': artist_uuid, 'playlist.uuid': playlist_uuid}, {'$set': document_to_store}, upsert=True)

            # The logic for 'song_streaming_data' which populated the old 'songs_audience' collection has been removed.

            if 'song_centric_streaming' in data:
                for song_data in data['song_centric_streaming']:
                    if song_uuid := song_data.get('song_uuid'):
                        doc_to_store = {
                            'artist_uuid': artist_uuid,
                            'song_uuid': song_uuid,
                            'history': song_data.get('history', []),
                            'playlists': song_data.get('playlists', []),
                            'last_updated': datetime.utcnow()
                        }
                        self.collections['songs_aggregate_audience'].update_one(
                            {'song_uuid': song_uuid},
                            {'$set': doc_to_store},
                            upsert=True
                        )

        except OperationFailure as e:
            print(f"Error storing secondary data: {e}")

    def store_demographic_data(self, artist_uuid: str, data: Dict[str, Any]):
        """Stores demographic data for followers."""
        try:
            if 'local_audience' in data and 'error' not in data['local_audience']:
                platform = data['local_audience'].get('platform', 'instagram')
                query_filter = {'artist_uuid': artist_uuid, 'platform': platform}
                self.collections['demographic_followers'].update_one(
                    query_filter,
                    {'$set': data['local_audience']},
                    upsert=True
                )
        except OperationFailure as e:
            print(f"Error storing demographic data: {e}")

    def store_album_audience_data(self, album_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for ALBUM-LEVEL audience.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            platform = data.get('platform', 'spotify')
            query_filter = {'album_uuid': album_uuid, 'platform': platform}
            self.append_timeseries_data('album_audience', query_filter, data['items'])

        except OperationFailure as e:
            print(f"Error storing album audience data: {e}")

    def store_song_audience_data(self, song_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for SONG-LEVEL audience.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            platform = data.get('platform', 'spotify')
            query_filter = {'song_uuid': song_uuid, 'platform': platform}
            self.append_timeseries_data('song_audience', query_filter, data['items'])

        except OperationFailure as e:
            print(f"Error storing song audience data: {e}")


    def store_timeseries_data(self, artist_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for ARTIST-LEVEL collections.
        """
        try:
            for coll_name, data_key, platform_key in [
                ('audience', 'audience', 'platform'), 
                ('popularity', 'popularity', 'source'), 
                ('streaming_audience', 'streaming_audience', 'platform'),
                ('local_streaming_history', 'local_streaming_audience', 'platform')
            ]:
                if data_key in data and 'items' in data.get(data_key, {}):
                    # --- Start of corrected block ---
                    # Robustly get the platform/source value, providing a fallback.
                    # This prevents 'None' from being used in the query filter.
                    platform_or_source_value = data[data_key].get(platform_key)
                    if not platform_or_source_value:
                        platform_or_source_value = 'spotify' # Default fallback
                    
                    query_filter = {'artist_uuid': artist_uuid, platform_key: platform_or_source_value}
                    # --- End of corrected block ---
                    self.append_timeseries_data(coll_name, query_filter, data[data_key]['items'])
        except OperationFailure as e:
            print(f"Error storing time-series data: {e}")

    def get_timeseries_data_range(self, collection_name: str, query_filter: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Finds the earliest and latest date for a given time-series document."""
        pipeline = [{'$match': query_filter}, {'$unwind': '$history'}, {'$group': {'_id': '$_id', 'minDate': {'$min': '$history.date'}, 'maxDate': {'$max': '$history.date'}}}]
        result = list(self.collections[collection_name].aggregate(pipeline))
        if not result: return None, None
        min_date = datetime.fromisoformat(result[0]['minDate'].replace('Z', '+00:00')) if result[0].get('minDate') else None
        max_date = datetime.fromisoformat(result[0]['maxDate'].replace('Z', '+00:00')) if result[0].get('maxDate') else None
        return min_date, max_date

    def append_timeseries_data(self, collection_name: str, query_filter: Dict, new_data_points: List[Dict]):
        """Appends new data points to a time-series document's history array."""
        if new_data_points:
            self.collections[collection_name].update_one(
                query_filter, 
                {'$addToSet': {'history': {'$each': new_data_points}}}, 
                upsert=True
            )

    def get_timeseries_data_for_display(self, collection_name: str, query_filter: Dict, start_date, end_date) -> List[Dict]:
        """Gets the final time-series data within a specific date range for display."""
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"
        pipeline = [{'$match': query_filter}, {'$project': {'history': {'$filter': {'input': '$history', 'as': 'item', 'cond': {'$and': [{'$gte': ['$$item.date', start_iso]}, {'$lte': ['$$item.date', end_iso]}]}}}}}]
        result = list(self.collections[collection_name].aggregate(pipeline))
        return result[0]['history'] if result and 'history' in result[0] else []
