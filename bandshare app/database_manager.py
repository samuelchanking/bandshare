# database_manager.py

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime, date
from analysis_tools import adjust_cumulative_history

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
                'demographic_followers': self.db['demographic_followers'],
                'local_streaming_history': self.db['local_streaming_history'],
                'album_audience': self.db['album_audience'],
                'song_audience': self.db['song_audience'],
                'playlist_audience': self.db['playlist_audience'],
                'typed_playlists': self.db['typed_playlists'],
                'songs_playlists': self.db['songs_playlists'],
                'global_song': self.db['global_song'],
                'global_song_audience': self.db['global_song_audience'],
                'global_song_playlists': self.db['global_song_playlists'],
                'song_popularity': self.db['song_popularity'], 
                'events': self.db['events'], # Add this line


            }
            # Create indexes for efficient queries
            self.collections['songs_playlists'].create_index([("artist_uuid", ASCENDING)])
            self.collections['global_song'].create_index([("playlist_uuid", ASCENDING)])
            self.collections['global_song'].create_index([("song.uuid", ASCENDING)])
            self.collections['global_song_audience'].create_index([("song_uuid", ASCENDING)])
            self.collections['global_song_playlists'].create_index([("playlist.uuid", ASCENDING)])
            self.collections['song_popularity'].create_index([("song_uuid", ASCENDING)]) 
            self.collections['events'].create_index([("artist_uuid", ASCENDING)]) # Add this line
            self.collections['events'].create_index([("date", ASCENDING)]) # Add this line



        except ConnectionFailure as e:
            raise e
        
                
    def close_connection(self):
        """Closes the MongoDB connection."""
        self.client.close()
        
    # --- MODIFIED FUNCTION ---
    def get_all_songs_for_artist(self, artist_uuid: str) -> List[Dict[str, Any]]:
        """Retrieves all songs (UUID and name) for a given artist."""
        # Removed the complex projection to ensure the full document is retrieved
        songs_cursor = self.collections['songs'].find({'object.artists.uuid': artist_uuid})
        
        songs = []
        for song_doc in songs_cursor:
            # The full document is now available
            # Determine the source of metadata (top-level or nested 'object')
            meta_source = song_doc.get('object', song_doc)
            
            # Ensure we have the required fields before appending
            song_uuid = song_doc.get('song_uuid')
            name = meta_source.get('name')
            
            if song_uuid and name:
                songs.append({
                    'uuid': song_uuid,
                    'name': name,
                    'releaseDate': song_doc.get('releaseDate'),
                    'imageUrl': song_doc.get('imageUrl')
                })
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
                meta_source = data['metadata'].get('object', data['metadata'])
                filtered = {
                    'artist_uuid': artist_uuid,
                    'name': meta_source.get('name'),
                    'imageUrl': meta_source.get('imageUrl'),
                    'countryCode': meta_source.get('countryCode'),
                    'biography': meta_source.get('biography'),
                    'gender': meta_source.get('gender'),
                    'type': meta_source.get('type'),
                    'careerStage': meta_source.get('careerStage'),
                }
                genres_data = meta_source.get('genres', [])
                all_genres = []
                for g in genres_data:
                    root = g.get('root')
                    if root:
                        all_genres.append(root)
                    sub = g.get('sub', [])
                    all_genres.extend(sub)
                filtered['genres'] = all_genres
                self.collections['artists'].update_one(
                    {'artist_uuid': artist_uuid},
                    {'$set': filtered},
                    upsert=True
                )
            return {'status': 'success', 'message': 'Static data stored.'}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"DB error: {e}"}

    def store_secondary_artist_data(self, artist_uuid: str, data: Dict[str, Any]):
        """
        Stores or updates all secondary data with improved error handling.
        """
        try:
            if 'song_playlist_map' in data:
                all_song_uuids_in_batch = list(data['song_playlist_map'].keys())
                
                self.collections['songs_playlists'].delete_many({'artist_uuid': artist_uuid})

                docs_to_insert = []
                for song_uuid, entries in data['song_playlist_map'].items():
                    song_meta = data.get('song_metadata', {}).get(song_uuid, {})
                    for entry in entries:
                        doc = entry.copy()
                        doc['song'] = {
                            'uuid': song_uuid,
                            'name': song_meta.get('name')
                        }
                        doc['artist_uuid'] = artist_uuid 
                        docs_to_insert.append(doc)

                if docs_to_insert:
                    self.collections['songs_playlists'].insert_many(docs_to_insert)

            if 'albums' in data and 'items' in data.get('albums', {}):
                all_album_metadata = data.get('album_metadata', {})
                all_tracklists = data.get('tracklists', {})
                for album_summary in data['albums']['items']:
                    if isinstance(album_summary, dict) and (album_uuid_val := album_summary.get('uuid')):
                        album_meta = all_album_metadata.get(album_uuid_val, {})
                        tracklist_data = all_tracklists.get(album_uuid_val, {})
                        if isinstance(album_meta, dict):
                            combined_meta = album_summary.copy()
                            combined_meta.update(album_meta)
                            # Flatten the 'object' layer if present
                            if 'object' in combined_meta:
                                combined_meta.update(combined_meta.pop('object', {}))
                            # Build tracklist array
                            tracklist_array = []
                            if isinstance(tracklist_data, dict) and 'items' in tracklist_data:
                                for track in sorted(tracklist_data['items'], key=lambda t: (t.get('discNumber', 1), t.get('trackNumber', 0))):
                                    song_data = track.get('song', {})
                                    tracklist_array.append({
                                        'number': track.get('trackNumber', 1),  # Default to 1 for singles or if missing
                                        'name': song_data.get('name'),
                                        'uuid': song_data.get('uuid'),
                                        'imageUrl': song_data.get('imageUrl')
                                    })
                            # Filter to keep only specified fields
                            filtered_meta = {
                                k: combined_meta.get(k)
                                for k in ['name', 'creditName', 'releaseDate', 'uuid', 'type', 'upc', 'totalTracks', 'imageUrl']
                            }
                            filtered_meta['tracklist'] = tracklist_array
                            album_doc = {'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'album_metadata': filtered_meta}
                            self.collections['albums'].update_one({'album_uuid': album_uuid_val}, {'$set': album_doc}, upsert=True)

            if 'song_metadata' in data:
                for song_uuid, song_meta in data.get('song_metadata', {}).items():
                    if isinstance(song_meta, dict) and 'error' not in song_meta:
                        album_info = song_meta.get('album')
                        album_uuid_val = album_info.get('uuid') if isinstance(album_info, dict) else None
                        
                        doc_to_store = song_meta.copy()
                        doc_to_store.update({'artist_uuid': artist_uuid, 'album_uuid': album_uuid_val, 'song_uuid': song_uuid})
                        self.collections['songs'].update_one({'song_uuid': song_uuid}, {'$set': doc_to_store}, upsert=True)

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

    def store_playlist_audience_data(self, playlist_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for PLAYLIST-LEVEL audience
        using a single, atomic operation.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            query_filter = {'playlist_uuid': playlist_uuid}
            update_operation = {
                '$setOnInsert': {'playlist_uuid': playlist_uuid},
                '$addToSet': {'history': {'$each': data['items']}}
            }
            self.collections['playlist_audience'].update_one(
                query_filter,
                update_operation,
                upsert=True
            )
        except OperationFailure as e:
            print(f"Error storing playlist audience data: {e}")

    def store_album_audience_data(self, album_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for ALBUM-LEVEL audience
        using a single, atomic operation.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            platform = data.get('platform', 'spotify')
            query_filter = {'album_uuid': album_uuid, 'platform': platform}
            update_operation = {
                '$setOnInsert': {'album_uuid': album_uuid, 'platform': platform},
                '$addToSet': {'history': {'$each': data['items']}}
            }
            self.collections['album_audience'].update_one(
                query_filter,
                update_operation,
                upsert=True
            )
        except OperationFailure as e:
            print(f"Error storing album audience data: {e}")
            
    def store_song_metadata(self, song_uuid: str, metadata: Dict[str, Any]):
        """
        Stores or updates the metadata for a single song in the 'songs' collection.
        This is a more direct way to store song data.
        """
        try:
            if 'error' not in metadata and metadata:
                doc_to_store = metadata.copy()
                # Ensure essential UUIDs are at the top level
                doc_to_store['song_uuid'] = song_uuid
                if 'artists' in doc_to_store and doc_to_store['artists']:
                     doc_to_store['artist_uuid'] = doc_to_store['artists'][0].get('uuid')
                
                self.collections['songs'].update_one(
                    {'song_uuid': song_uuid},
                    {'$set': doc_to_store},
                    upsert=True
                )
        except OperationFailure as e:
            print(f"Error storing song metadata for {song_uuid}: {e}")


    def store_song_audience_data(self, song_uuid: str, data: Dict[str, Any]):
        """
        Stores pre-processed song audience data by overwriting any existing document.
        Assumes the passed 'history' is already cleaned, merged, and adjusted.
        """
        try:
            history = data.get('history', [])
            if not history:
                return

            platform = data.get('platform', 'spotify')
            query_filter = {'song_uuid': song_uuid, 'platform': platform}

            # Delete existing document to overwrite
            self.collections['song_audience'].delete_one(query_filter)

            # Insert new document with the provided history
            new_document = {
                'song_uuid': song_uuid,
                'platform': platform,
                'history': history
            }
            self.collections['song_audience'].insert_one(new_document)

        except OperationFailure as e:
            print(f"Error storing song audience data: {e}")   
            
            
            
    def store_song_popularity_data(self, song_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for SONG-LEVEL popularity
        using a single, atomic operation.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            platform = data.get('platform', 'spotify')
            query_filter = {'song_uuid': song_uuid, 'platform': platform}
            update_operation = {
                '$setOnInsert': {'song_uuid': song_uuid, 'platform': platform},
                '$addToSet': {'history': {'$each': data['items']}}
            }
            self.collections['song_popularity'].update_one(
                query_filter,
                update_operation,
                upsert=True
            )
        except OperationFailure as e:
            print(f"Error storing song popularity data: {e}")


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
                    platform_or_source_value = data[data_key].get(platform_key)
                    if not platform_or_source_value:
                        platform_or_source_value = 'spotify' 
                    
                    query_filter = {'artist_uuid': artist_uuid, platform_key: platform_or_source_value}
                    self.append_timeseries_data(coll_name, query_filter, data[data_key]['items'])
        except OperationFailure as e:
            print(f"Error storing time-series data: {e}")

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
            self.collections[collection_name].update_one(
                query_filter, 
                {'$addToSet': {'history': {'$each': new_data_points}}}, 
                upsert=True
            )
    
    def store_typed_playlists(self, playlists: List[Dict[str, Any]]):
        """
        Stores or updates playlists from the by-type endpoint.
        """
        try:
            for playlist in playlists:
                if playlist_uuid := playlist.get('uuid'):
                    self.collections['typed_playlists'].update_one(
                        {'uuid': playlist_uuid},
                        {'$set': playlist},
                        upsert=True
                    )
        except OperationFailure as e:
            print(f"Error storing typed playlists: {e}")

    def get_timeseries_data_for_display(self, collection_name: str, query_filter: Dict, start_date, end_date) -> List[Dict]:
        """Gets the final time-series data within a specific date range for display."""
        start_iso = datetime.combine(start_date, datetime.min.time()).isoformat() + "Z"
        end_iso = datetime.combine(end_date, datetime.max.time()).isoformat() + "Z"
        pipeline = [
            {'$match': query_filter},
            {'$project': {
                'history': {
                    '$filter': {
                        'input': '$history',
                        'as': 'item',
                        'cond': {
                            '$and': [
                                {'$gte': ['$$item.timestamp', start_iso]},
                                {'$lte': ['$$item.timestamp', end_iso]}
                            ]
                        }
                    }
                }
            }}
        ]
        result = list(self.collections[collection_name].aggregate(pipeline))
        return result[0]['history'] if result and 'history' in result[0] else []    
    
    
    def get_timeseries_value_for_date(self, collection_name: str, query_filter: Dict, target_date: datetime) -> Optional[int]:
        """
        Finds the value for a specific date within a time-series document's history
        by performing a robust BSON date comparison.
        """
        pipeline = [
            {'$match': query_filter},
            {'$project': {
                'matched_item': {
                    '$filter': {
                        'input': '$history',
                        'as': 'item',
                        'cond': {
                            # Use $dateFromString for robust comparison against a datetime object
                            '$eq': [
                                {'$dateFromString': {'dateString': '$$item.date'}},
                                target_date
                            ]
                        }
                    }
                }
            }},
            {'$unwind': '$matched_item'},
            {'$limit': 1}
        ]
        result = list(self.collections[collection_name].aggregate(pipeline))
        if not result:
            return None
        
        item = result[0].get('matched_item', {})
        value = item.get('followerCount')
        if value is None:
            value = item.get('value')
        
        return value
    
    def store_global_song_audience_data(self, song_uuid: str, data: Dict[str, Any]):
        """
        Appends new time-series data points to the database for a song's audience
        in the 'global_song_audience' collection.
        """
        try:
            if 'error' in data or 'items' not in data or not data['items']:
                return

            query_filter = {'song_uuid': song_uuid}
            update_operation = {
                '$setOnInsert': {'song_uuid': song_uuid},
                '$addToSet': {'history': {'$each': data['items']}}
            }
            self.collections['global_song_audience'].update_one(
                query_filter,
                update_operation,
                upsert=True
            )
        except OperationFailure as e:
            print(f"Error storing global song audience data for {song_uuid}: {e}")
            
    def store_artist_events(self, artist_uuid: str, events: List[Dict[str, Any]]):
        """
        Stores or updates the list of events for a given artist.
        Deletes existing events for the artist before inserting new ones to ensure freshness.
        """
        try:
            if not events or 'error' in events:
                return  # Do nothing if the new event list is empty or contains an error

            # Delete all existing events for this artist
            self.collections['events'].delete_many({'artist_uuid': artist_uuid})

            # Add artist_uuid to each event document for easy querying
            for event in events:
                event['artist_uuid'] = artist_uuid

            # Insert the new list of events
            self.collections['events'].insert_many(events)

        except OperationFailure as e:
            print(f"Error storing artist events for {artist_uuid}: {e}")

    def get_artist_events_from_db(self, artist_uuid: str) -> List[Dict[str, Any]]:
        """
        Retrieves all stored events for a given artist, sorted by date.
        """
        try:
            # Sort by date descending to show most recent/upcoming first
            return list(self.collections['events'].find({'artist_uuid': artist_uuid}).sort('date', -1))
        except OperationFailure as e:
            print(f"Error retrieving artist events for {artist_uuid}: {e}")
            return []
        
    def store_venue_metadata(self, venue_data: Dict[str, Any]) -> bool:
        """Stores or updates metadata for a single venue. Returns True on success."""
        try:
            # Handle nested 'object' structure if present
            inner_data = venue_data.get('object', venue_data)
            if 'error' not in inner_data and (venue_uuid := inner_data.get('uuid')):
                result = self.collections['venues'].update_one(
                    {'uuid': venue_uuid},
                    {'$set': inner_data},
                    upsert=True
                )
                return result.matched_count > 0 or result.modified_count > 0 or result.upserted_id is not None
        except OperationFailure as e:
            print(f"Error storing venue metadata for {venue_data.get('uuid')}: {e}")
        return False

    def store_festival_metadata(self, festival_data: Dict[str, Any]) -> bool:
        """Stores or updates metadata for a single festival. Returns True on success."""
        try:
            # Handle nested 'object' structure if present
            inner_data = festival_data.get('object', festival_data)
            if 'error' not in inner_data and (festival_uuid := inner_data.get('uuid')):
                result = self.collections['festivals'].update_one(
                    {'uuid': festival_uuid},
                    {'$set': inner_data},
                    upsert=True
                )
                return result.matched_count > 0 or result.modified_count > 0 or result.upserted_id is not None
        except OperationFailure as e:
            print(f"Error storing festival metadata for {festival_data.get('uuid')}: {e}")
        return False
