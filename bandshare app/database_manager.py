# database_manager.py

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import Dict, Any
import re

class DatabaseManager:
    """Manages all interactions with the MongoDB database."""

    def __init__(self, mongo_uri: str, db_name: str):
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ismaster')
            self.db = self.client[db_name]
            self.collections = {
                'artists': self.db['artists'],
                'albums': self.db['albums'],
                'tracklists': self.db['album_tracklist'],
                'songs': self.db['songs'],
                'playlists': self.db['playlists'], # Added
            }
            print(f"Successfully connected to MongoDB database '{db_name}'")
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB at {mongo_uri}.")
            raise e

    def close_connection(self):
        self.client.close()
        print("MongoDB connection closed.")

    def store_artist_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        if 'error' in data: return {'status': 'error', 'message': data['error']}
        try:
            artist_uuid = data['artist_uuid']
            
            # Upsert artist
            if 'metadata' in data:
                self.collections['artists'].update_one({'artist_uuid': artist_uuid}, {'$set': data['metadata']}, upsert=True)

            # Upsert albums with merged metadata
            if 'albums' in data and 'items' in data.get('albums', {}):
                all_album_metadata = data.get('album_metadata', {})
                for album_summary in data['albums']['items']:
                    if album_uuid := album_summary.get('uuid'):
                        album_doc = album_summary.copy()
                        detailed_metadata = all_album_metadata.get(album_uuid, {})
                        album_doc.update(detailed_metadata.get('object', detailed_metadata))
                        album_doc.update({'album_uuid': album_uuid, 'artist_uuid': artist_uuid})
                        self.collections['albums'].update_one({'album_uuid': album_uuid}, {'$set': album_doc}, upsert=True)

            # Delete and insert for tracklists, songs, and playlists to ensure freshness
            sub_collections = {
                'tracklists': ('tracklists', 'album_uuid'),
                'songs': ('song_metadata', 'song_uuid'),
                'playlists': ('playlists', 'playlist.uuid') # Assuming playlist object has a uuid
            }

            for coll_name, (data_key, id_key_path) in sub_collections.items():
                self.collections[coll_name].delete_many({'artist_uuid': artist_uuid})
                docs_to_insert = []
                if data_key in data:
                    if coll_name == 'songs':
                        docs_to_insert = [meta | {'song_uuid': suuid, 'album_uuid': auuid, 'artist_uuid': artist_uuid} for auuid, songs in data[data_key].items() for suuid, meta in songs.items() if 'error' not in meta]
                    elif coll_name == 'tracklists':
                        docs_to_insert = [d | {'album_uuid': uuid, 'artist_uuid': artist_uuid} for uuid, d in data[data_key].items() if 'error' not in d]
                    elif coll_name == 'playlists':
                        docs_to_insert = [p | {'artist_uuid': artist_uuid} for p in data[data_key].get('items', [])]
                
                if docs_to_insert:
                    self.collections[coll_name].insert_many(docs_to_insert)
                    print(f"Stored {len(docs_to_insert)} documents in '{coll_name}'.")

            return {'status': 'success', 'message': f'Data for artist {artist_uuid} stored/updated.'}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"Database operation failed: {e}"}

    def delete_artist_data(self, artist_uuid: str) -> Dict[str, str]:
        if not artist_uuid: return {'status': 'error', 'message': 'Artist UUID cannot be empty.'}
        try:
            results = {name: col.delete_many({'artist_uuid': artist_uuid}).deleted_count for name, col in self.collections.items()}
            return {'status': 'success', 'message': f"Successfully deleted {sum(results.values())} documents."}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"Database deletion failed: {e}"}

    def search_artist_by_name(self, artist_name: str) -> Dict[str, Any] | None:
        if not artist_name: return None
        search_regex = re.compile(f"^{re.escape(artist_name)}$", re.IGNORECASE)
        query = {'$or': [{'name': search_regex}, {'object.name': search_regex}]}
        return self.collections['artists'].find_one(query)
