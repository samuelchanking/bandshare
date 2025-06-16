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
                'streaming_audience': self.db['streaming_audience'] 
            }
            # Create indexes for efficient queries
            self.collections['audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
            self.collections['popularity'].create_index([("artist_uuid", ASCENDING), ("source", ASCENDING)])
            self.collections['streaming_audience'].create_index([("artist_uuid", ASCENDING), ("platform", ASCENDING)])
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

    # --- UPDATED: The old store_artist_data function is now split into these two. ---

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
        Stores or updates albums, tracklists, songs, and playlists.
        """
        try:
            # Intelligent "ADD ONLY" logic for albums, tracklists, and songs
            new_album_uuids = {album.get('uuid') for album in data.get('albums', {}).get('items', []) if album.get('uuid')}
            existing_albums_cursor = self.collections['albums'].find({'artist_uuid': artist_uuid}, {'album_uuid': 1})
            existing_album_uuids = {album['album_uuid'] for album in existing_albums_cursor}
            
            missing_album_uuids = new_album_uuids - existing_album_uuids

            if missing_album_uuids:
                print(f"Found {len(missing_album_uuids)} new album(s) for artist {artist_uuid}. Adding them.")
                
                albums_to_insert, tracklists_to_insert, songs_to_insert = [], [], []
                all_album_metadata = data.get('album_metadata', {})
                all_tracklists = data.get('tracklists', {})
                all_song_metadata = data.get('song_metadata', {})
                missing_albums_data = [album for album in data.get('albums', {}).get('items', []) if album.get('uuid') in missing_album_uuids]

                for album_summary in missing_albums_data:
                    album_uuid_val = album_summary.get('uuid')
                    combined_meta = album_summary | all_album_metadata.get(album_uuid_val, {})
                    album_doc = {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'album_metadata': combined_meta}
                    albums_to_insert.append(album_doc)

                    if tracklist_data := all_tracklists.get(album_uuid_val):
                        if 'error' not in tracklist_data:
                            tracklists_to_insert.append(tracklist_data | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val})
                    
                    if songs := all_song_metadata.get(album_uuid_val):
                        for song_uuid, song_meta in songs.items():
                             if 'error' not in song_meta:
                                songs_to_insert.append(song_meta | {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'song_uuid': song_uuid})

                if albums_to_insert: self.collections['albums'].insert_many(albums_to_insert)
                if tracklists_to_insert: self.collections['tracklists'].insert_many(tracklists_to_insert)
                if songs_to_insert: self.collections['songs'].insert_many(songs_to_insert)
            else:
                print(f"Album list for artist {artist_uuid} is unchanged.")

            # "Upsert" logic for Playlists
            if 'playlists' in data and 'items' in data.get('playlists', {}):
                for playlist_item in data['playlists']['items']:
                    playlist_uuid = playlist_item.get('playlist', {}).get('uuid')
                    if not playlist_uuid: continue
                    document_to_store = playlist_item | {'artist_uuid': artist_uuid}
                    self.collections['playlists'].update_one({'artist_uuid': artist_uuid, 'playlist.uuid': playlist_uuid}, {'$set': document_to_store}, upsert=True)
        
        except OperationFailure as e:
            print(f"Error storing secondary data: {e}")


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
            self.collections[collection_name].update_one(query_filter, {'$addToSet': {'history': {'$each': new_data_points}}}, upsert=True)

    def get_timeseries_data_for_display(self, collection_name: str, query_filter: Dict, start_date, end_date) -> List[Dict]:
        """Gets the final time-series data within a specific date range for display."""
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"
        pipeline = [{'$match': query_filter}, {'$project': {'history': {'$filter': {'input': '$history', 'as': 'item', 'cond': {'$and': [{'$gte': ['$$item.date', start_iso]}, {'$lte': ['$$item.date', end_iso]}]}}}}}]
        result = list(self.collections[collection_name].aggregate(pipeline))
        return result[0]['history'] if result and 'history' in result[0] else []
