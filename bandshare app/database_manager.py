# database_manager.py

from pymongo import MongoClient
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
                'playlists': self.db['playlists'], # Handles playlists
            }
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
        Stores albums and playlists, using a delete-and-insert strategy for freshness.
        This ensures that if an album or playlist is removed, it is reflected in the DB.
        """
        try:
            # Store albums
            if 'albums' in data and 'items' in data.get('albums', {}):
                self.collections['albums'].delete_many({'artist_uuid': artist_uuid})
                albums_to_insert = [d | {'album_uuid': d['uuid'], 'artist_uuid': artist_uuid} for d in data['albums']['items']]
                if albums_to_insert:
                    self.collections['albums'].insert_many(albums_to_insert)
            
            # Store playlists
            if 'playlists' in data and 'items' in data.get('playlists', {}):
                self.collections['playlists'].delete_many({'artist_uuid': artist_uuid})
                playlists_to_insert = [p | {'artist_uuid': artist_uuid} for p in data['playlists']['items']]
                if playlists_to_insert:
                    self.collections['playlists'].insert_many(playlists_to_insert)

        except OperationFailure as e:
            print(f"Error storing secondary data: {e}")

    def delete_artist_data(self, artist_uuid: str) -> Dict[str, str]:
        """Deletes all documents related to a specific artist_uuid from all collections."""
        if not artist_uuid:
            return {'status': 'error', 'message': 'Artist UUID cannot be empty.'}
        try:
            query_filter = {'artist_uuid': artist_uuid}
            # Only delete from collections that exist in this version
            collections_to_delete_from = ['artists', 'albums', 'tracklists', 'songs', 'playlists']
            results = {name: self.collections[name].delete_many(query_filter).deleted_count for name in collections_to_delete_from}
            total = sum(results.values())
            print(f"Deletion report: {results}")
            return {'status': 'success', 'message': f"Successfully deleted {total} total documents for artist {artist_uuid}."}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"Database deletion failed: {e}"}

