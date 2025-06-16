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
                # ADDED: collection for streaming audience
                'streaming_audience': self.db['streaming_audience'] 
            }
            # Create indexes for efficient queries
            self.collections['audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['popularity'].create_index([("artist_uuid", ASCENDING), ("source", ASCENDING)])
            # ADDED: index for streaming audience
            self.collections['streaming_audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            print(f"Successfully connected to MongoDB database '{db_name}'")
        except ConnectionFailure as e:
            raise e

    def close_connection(self):
        """Closes the MongoDB connection."""
        self.client.close()

    def search_artist_by_name(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """Finds an artist by name in the local database."""
        if not artist_name: return None
        search_regex = re.compile(f"^{re.escape(artist_name)}$", re.IGNORECASE)
        return self.collections['artists'].find_one({'$or': [{'name': search_regex}, {'object.name': search_regex}]})

    def store_static_artist_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Upserts the main artist metadata document."""
        if 'error' in data: return {'status': 'error', 'message': data['error']}
        try:
            artist_uuid = data['artist_uuid']
            if 'metadata' in data:
                self.collections['artists'].update_one({'artist_uuid': artist_uuid}, {'$set': data['metadata']}, upsert=True)
            return {'status': 'success', 'message': 'Static data stored.'}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"DB error: {e}"}

    def store_secondary_artist_data(self, artist_uuid: str, data: Dict[str, Any]):
        """
        Stores or updates albums, playlists, and streaming data using an 'upsert' strategy.
        """
        try:
            # Upsert albums
            if 'albums' in data and 'items' in data.get('albums', {}):
                all_album_metadata = data.get('album_metadata', {})
                for album_summary in data['albums']['items']:
                    album_uuid_val = album_summary.get('uuid')
                    if not album_uuid_val: continue

                    combined_meta = album_summary.copy()
                    detailed_metadata = all_album_metadata.get(album_uuid_val, {})
                    combined_meta.update(detailed_metadata.get('object', detailed_metadata))

                    album_doc = {
                        'artist_uuid': artist_uuid,
                        'album_uuid': album_uuid_val,
                        'album_metadata': combined_meta
                    }
                    
                    self.collections['albums'].update_one(
                        {'album_uuid': album_uuid_val},
                        {'$set': album_doc},
                        upsert=True
                    )
                    
            # --- ADDED: Logic to correctly store song metadata ---
            if 'song_metadata' in data:
                for album_uuid, songs in data['song_metadata'].items():
                    for song_uuid, song_meta in songs.items():
                        if 'error' not in song_meta:
                            document_to_store = song_meta | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid, 'song_uuid': song_uuid}
                            self.collections['songs'].update_one(
                                {'song_uuid': song_uuid},
                                {'$set': document_to_store},
                                upsert=True
                            )

            
            # Upsert playlists
            if 'playlists' in data and 'items' in data.get('playlists', {}):
                for playlist_item in data['playlists']['items']:
                    playlist_doc = playlist_item.get('playlist', {})
                    playlist_uuid = playlist_doc.get('uuid')
                    if not playlist_uuid: continue
                    document_to_store = playlist_item | {'artist_uuid': artist_uuid}
                    self.collections['playlists'].update_one(
                        {'playlist.uuid': playlist_uuid, 'artist_uuid': artist_uuid},
                        {'$set': document_to_store},
                        upsert=True
                    )
            
            # --- ADDED: Logic to correctly store tracklist data ---
            if 'tracklists' in data:
                for album_uuid, tracklist_data in data['tracklists'].items():
                    if 'error' not in tracklist_data:
                        document_to_store = tracklist_data | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid}
                        self.collections['tracklists'].update_one(
                            {'album_uuid': album_uuid},
                            {'$set': document_to_store},
                            upsert=True
                        )
                        
            # --- ADDED: Upsert streaming audience data ---
            if 'streaming_audience' in data and 'items' in data.get('streaming_audience', {}):
                for item in data['streaming_audience']['items']:
                    # Use artist_uuid and date as a unique key for each data point
                    if item_date := item.get('date'):
                        document_to_store = item | {'artist_uuid': artist_uuid}
                        self.collections['streaming_audience'].update_one(
                            {'artist_uuid': artist_uuid, 'date': item_date},
                            {'$set': document_to_store},
                            upsert=True
                        )

        except OperationFailure as e:
            print(f"Error storing secondary data: {e}")

    def get_timeseries_data_range(self, collection_name: str, query_filter: Dict) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Finds the earliest and latest date for a given time-series document."""
        pipeline = [{'$match': query_filter}, {'$unwind': '$history'}, {'$group': {'_id': '$_id', 'minDate': {'$min': '$history.date'}, 'maxDate': {'$max': '$history.date'}}}]
        result = list(self.collections[collection_name].aggregate(pipeline))
        if not result: return None, None
        min_date_str = result[0].get('minDate')
        max_date_str = result[0].get('maxDate')
        min_date = datetime.fromisoformat(min_date_str.replace('Z', '+00:00')) if min_date_str else None
        max_date = datetime.fromisoformat(max_date_str.replace('Z', '+00:00')) if max_date_str else None
        return min_date, max_date

    def append_timeseries_data(self, collection_name: str, query_filter: Dict, new_data_points: List[Dict]):
        """Appends new data points to a time-series document's history array."""
        if new_data_points:
            self.collections[collection_name].update_one(query_filter, {'$addToSet': {'history': {'$each': new_data_points}}}, upsert=True)

    def get_timeseries_data_for_display(self, collection_name: str, query_filter: Dict, start_date, end_date) -> List[Dict]:
        """Gets the final time-series data within a specific date range for display."""
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"
        pipeline = [{'$match': query_filter}, {'$project': {'history': {'$filter': {'input': '$history', 'as': 'item', 'cond': {'$and': [{'$gte': ['$$item.date', start_iso]}, {'$lte': ['$$item.date', end_iso]}]}}}}}]
        result = list(self.collections[collection_name].aggregate(pipeline))
        return result[0]['history'] if result and 'history' in result[0] else []
