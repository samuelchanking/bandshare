import requests
import json
import urllib.parse
import re
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime, timedelta


HEADERS = {
    'x-app-id': 'MANCHESTER_696DCD6E',
    'x-api-key': '11ed17a6cf25afa4',
}


class SoundchartsAPI:
    def __init__(self, db_name='soundcharts7', mongo_uri='mongodb://localhost:27017/'):
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collections = {
                'artists': self.db['artists'],
                'albums': self.db['albums'],
                'tracklist': self.db['album_tracklist'],
                'songs': self.db['songs'],
            }
            print(f"Connected to MongoDB database '{db_name}'")
        except ConnectionError:
            print(f"Failed to connect to MongoDB at {mongo_uri}. Ensure MongoDB is running.")
            raise

    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()
        print("MongoDB connection closed")

    def fetch_artist_data(self, artist_name):
        """Fetch artist metadata, albums, tracklists, album metadata, and song metadata without storing in the database."""
        # Search for artist and get UUID
        search_params = {'offset': '0', 'limit': '1'}
        encoded_artist_name = urllib.parse.quote(artist_name)
        search_response = requests.get(
            f'https://customer.api.soundcharts.com/api/v2/artist/search/{encoded_artist_name}',
            params=search_params,
            headers=HEADERS,
        )
        if search_response.status_code != 200:
            return {'error': f"Error searching for artist: Status code {search_response.status_code}"}
        search_data = search_response.json()
        if 'items' not in search_data or not search_data['items']:
            return {'error': f"No artists found for '{artist_name}'"}
        artist_uuid = search_data['items'][0]['uuid']

        # Define endpoints for artist metadata, albums, and songs
        endpoints = [
            {'name': 'metadata', 'url': f'https://customer.api.soundcharts.com/api/v2.9/artist/{artist_uuid}', 'params': {}, 'key': 'metadata'},
            {'name': 'albums', 'url': f'https://customer.api.soundcharts.com/api/v2.34/artist/{artist_uuid}/albums', 'params': {'offset': '0', 'limit': '100'}, 'key': 'albums'},
            {'name': 'songs', 'url': f'https://customer.api.soundcharts.com/api/v2.21/artist/{artist_uuid}/songs', 'params': {'offset': '0', 'limit': '100'}, 'key': 'songs'},
        ]

        # Fetch data
        result = {'artist_uuid': artist_uuid, 'artist_name': artist_name}
        for endpoint in endpoints:
            response = requests.get(endpoint['url'], params=endpoint['params'], headers=HEADERS)
            if response.status_code == 200:
                result[endpoint['key']] = response.json()
            else:
                result[endpoint['key']] = {'error': f"Failed to fetch {endpoint['name']}: Status {response.status_code}"}

        # Fetch tracklists, album metadata, and song metadata for each album
        if 'albums' in result and 'items' in result['albums']:
            result['tracklists'] = {}
            result['album_metadata'] = {}
            result['song_metadata'] = {}
            for album in result['albums']['items']:
                album_uuid = album['uuid']
                # Fetch tracklist
                tracklist_response = requests.get(
                    f'https://customer.api.soundcharts.com/api/v2.26/album/{album_uuid}/tracks',
                    params={},
                    headers=HEADERS
                )
                if tracklist_response.status_code == 200:
                    result['tracklists'][album_uuid] = tracklist_response.json()
                else:
                    result['tracklists'][album_uuid] = {'error': f"Failed to fetch tracklist: Status {tracklist_response.status_code}"}

                # Fetch album metadata
                album_metadata_response = requests.get(
                    f'https://customer.api.soundcharts.com/api/v2.36/album/by-uuid/{album_uuid}',
                    params={},
                    headers=HEADERS
                )
                if album_metadata_response.status_code == 200:
                    result['album_metadata'][album_uuid] = album_metadata_response.json()
                else:
                    result['album_metadata'][album_uuid] = {'error': f"Failed to fetch album metadata: Status {album_metadata_response.status_code}"}

                # Fetch song metadata for each track in the tracklist
                if album_uuid in result['tracklists'] and 'items' in result['tracklists'][album_uuid]:
                    result['song_metadata'][album_uuid] = {}
                    for item in result['tracklists'][album_uuid]['items']:
                        song_uuid = item.get('song', {}).get('uuid')
                        if song_uuid:
                            song_metadata_response = requests.get(
                                f'https://customer.api.soundcharts.com/api/v2.25/song/{song_uuid}',
                                params={},
                                headers=HEADERS
                            )
                            if song_metadata_response.status_code == 200:
                                result['song_metadata'][album_uuid][song_uuid] = song_metadata_response.json()
                            else:
                                result['song_metadata'][album_uuid][song_uuid] = {'error': f"Failed to fetch song metadata: Status {song_metadata_response.status_code}"}

        return result

    def store_artist_data(self, data):
        """Store fetched artist data into MongoDB."""
        try:
            # Store artist metadata
            artist_metadata = data.get('metadata', {})
            artist_metadata['artist_uuid'] = data['artist_uuid']
            self.collections['artists'].update_one(
                {'artist_uuid': data['artist_uuid']},
                {'$set': artist_metadata},
                upsert=True
            )
            print(f"Stored artist metadata for {data['artist_uuid']}")

            # Store albums
            if 'albums' in data and 'items' in data['albums']:
                for album in data['albums']['items']:
                    album_metadata = album.copy()
                    album_metadata['album_uuid'] = album['uuid']
                    album_metadata['artist_uuid'] = data['artist_uuid']
                    self.collections['albums'].update_one(
                        {'album_uuid': album['uuid']},
                        {'$set': album_metadata},
                        upsert=True
                    )
                    print(f"Stored album {album['uuid']}")

                    # Store tracklist
                    if 'tracklists' in data and album['uuid'] in data['tracklists']:
                        tracklist_data = data['tracklists'][album['uuid']]
                        tracklist_data['album_uuid'] = album['uuid']
                        tracklist_data['artist_uuid'] = data['artist_uuid']
                        self.collections['tracklist'].update_one(
                            {'album_uuid': album['uuid']},
                            {'$set': tracklist_data},
                            upsert=True
                        )
                        print(f"Stored tracklist for album {album['uuid']}")

                    # Store album metadata
                    if 'album_metadata' in data and album['uuid'] in data['album_metadata']:
                        album_meta_data = data['album_metadata'][album['uuid']]
                        album_meta_data['album_uuid'] = album['uuid']
                        album_meta_data['artist_uuid'] = data['artist_uuid']
                        self.collections['albums'].update_one(
                            {'album_uuid': album['uuid']},
                            {'$set': {'metadata': album_meta_data}},
                            upsert=True
                        )
                        print(f"Stored album metadata for {album['uuid']}")

                    # Store song metadata
                    if 'song_metadata' in data and album['uuid'] in data['song_metadata']:
                        for song_uuid, song_meta_data in data['song_metadata'][album['uuid']].items():
                            song_meta_data['song_uuid'] = song_uuid
                            song_meta_data['album_uuid'] = album['uuid']
                            song_meta_data['artist_uuid'] = data['artist_uuid']
                            self.collections['tracklist'].update_one(
                                {'album_uuid': album['uuid'], 'items.song.uuid': song_uuid},
                                {'$set': {'items.$': {'song': {'uuid': song_uuid, 'metadata': song_meta_data}}}},
                                upsert=True
                            )
                            print(f"Stored song metadata for song {song_uuid} in album {album['uuid']}")

            return {'status': 'success', 'message': 'Data stored successfully'}
        except OperationFailure as e:
            return {'status': 'error', 'message': f"Failed to store data: {e}"}
