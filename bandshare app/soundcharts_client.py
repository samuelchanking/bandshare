# soundcharts_client.py

import requests
import urllib.parse
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

class SoundchartsAPIClient:
    """A client to interact with the Soundcharts API."""
    BASE_URL = "https://customer.api.soundcharts.com/api"

    def __init__(self, app_id: str, api_key: str):
        if not app_id or not api_key:
            raise ValueError("API App ID and Key cannot be empty.")
        self._headers = {'x-app-id': app_id, 'x-api-key': api_key}

    def _request(self, url: str, params: Dict = None) -> Dict[str, Any]:
        try:
            response = requests.get(url, params=params, headers=self._headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': f"API request failed: {e}"}

    def search_artist(self, artist_name: str) -> Dict[str, Any]:
        """Finds an artist by name and returns their UUID."""
        encoded_artist_name = urllib.parse.quote(artist_name)
        url = f"{self.BASE_URL}/v2/artist/search/{encoded_artist_name}"
        search_data = self._request(url, params={'limit': '1'})
        if 'error' in search_data: return search_data
        if not search_data.get('items'): return {'error': f"No artists found for '{artist_name}'"}
        return {'uuid': search_data['items'][0]['uuid']}

    def get_artist_metadata(self, artist_uuid: str) -> Dict[str, Any]:
        """Fetches the main metadata document for an artist."""
        url = f"{self.BASE_URL}/v2.9/artist/{artist_uuid}"
        return self._request(url)

    def get_artist_audience(self, artist_uuid: str, platform: str, start_date=None, end_date=None) -> Dict[str, Any]:
        """Fetches time-series audience data (listeners, followers)."""
        url = f"{self.BASE_URL}/v2/artist/{artist_uuid}/audience/{platform}/"
        params = {}
        if start_date: params['startDate'] = str(start_date)
        if end_date: params['endDate'] = str(end_date)
        return self._request(url, params=params)

    def get_artist_popularity(self, artist_uuid: str, source: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        """Fetches time-series popularity data."""
        url = f"{self.BASE_URL}/v2/artist/{artist_uuid}/popularity/{source}"
        params = {}
        if start_date: params['startDate'] = str(start_date)
        if end_date: params['endDate'] = str(end_date)
        return self._request(url, params=params)

    def get_artist_albums(self, artist_uuid: str) -> Dict[str, Any]:
        """Fetches the list of albums for an artist."""
        url = f"{self.BASE_URL}/v2.34/artist/{artist_uuid}/albums"
        return self._request(url, params={'offset': '0', 'limit': '100'})

    def get_album_metadata(self, album_uuid: str) -> Dict[str, Any]:
        """Fetches detailed metadata for a single album."""
        url = f"{self.BASE_URL}/v2.36/album/by-uuid/{album_uuid}"
        return self._request(url)

    def get_album_tracks(self, album_uuid: str) -> Dict[str, Any]:
        """Fetches the tracklist for a given album UUID."""
        url = f"{self.BASE_URL}/v2.26/album/{album_uuid}/tracks"
        return self._request(url)

    def get_song_metadata(self, song_uuid: str) -> Dict[str, Any]:
        """Fetches detailed metadata for a single song."""
        url = f"{self.BASE_URL}/v2.25/song/{song_uuid}"
        return self._request(url)

    def get_artist_playlists(self, artist_uuid: str, platform: str = "spotify") -> Dict[str, Any]:
        """Fetches the list of playlists an artist is currently featured on."""
        url = f"{self.BASE_URL}/v2.20/artist/{artist_uuid}/playlist/current/{platform}"
        return self._request(url)

    def get_artist_streaming_audience(self, artist_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2/artist/{artist_uuid}/streaming/{platform}/listening"
        params = {}
        if start_date: params['startDate'] = str(start_date)
        if end_date: params['endDate'] = str(end_date)
        return self._request(url, params=params)

    def get_song_streaming_audience(self, song_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        """
        Fetches the time-series streaming audience data for a specific song.
        """
        if not song_uuid:
            return {'error': 'Song UUID must be provided.'}

        url = f"{self.BASE_URL}/v2/song/{song_uuid}/audience/{platform}"
        params = {}
        if start_date:
            params['startDate'] = str(start_date)
        if end_date:
            params['endDate'] = str(end_date)

        return self._request(url, params=params)


    def fetch_static_artist_data(self, artist_name: str) -> Dict[str, Any]:
        """
        Finds an artist and fetches only their main metadata document.
        """
        artist_info = self.search_artist(artist_name)
        if 'error' in artist_info: return artist_info

        artist_uuid = artist_info['uuid']
        result = {'artist_uuid': artist_uuid, 'artist_name': artist_name}
        result['metadata'] = self.get_artist_metadata(artist_uuid)
        return result

    def fetch_secondary_artist_data(self, artist_uuid: str, start_date=None, end_date=None) -> Dict[str, Any]:
        """
        Fetches all song metadata from albums and fetches streaming data for EACH playlist entry.
        """
        result = {}
        result['albums'] = self.get_artist_albums(artist_uuid)
        result['playlists'] = self.get_artist_playlists(artist_uuid)

        # --- Fetch metadata for ALL songs on ALL albums ---
        result['album_metadata'] = {}
        result['tracklists'] = {}
        result['song_metadata'] = {}
        if 'albums' in result and 'items' in result.get('albums', {}):
            for album_summary in result['albums']['items']:
                album_uuid_val = album_summary.get('uuid')
                if album_uuid_val:
                    result['album_metadata'][album_uuid_val] = self.get_album_metadata(album_uuid_val)
                    tracklist_data = self.get_album_tracks(album_uuid_val)
                    result['tracklists'][album_uuid_val] = tracklist_data

                    if 'items' in tracklist_data:
                        result['song_metadata'][album_uuid_val] = {}
                        for item in tracklist_data['items']:
                            if song_uuid := item.get('song', {}).get('uuid'):
                                result['song_metadata'][album_uuid_val][song_uuid] = self.get_song_metadata(song_uuid)

        # --- MODIFIED: Prepare to fetch streaming data for EACH song on EACH playlist ---
        playlist_items = result.get('playlists', {}).get('items', [])
        tasks_to_run = []
        processed_playlist_uuids = set() #<-- ADD: Create a set to track unique playlists

        if playlist_items:
            for item in playlist_items:
                song_uuid = item.get('song', {}).get('uuid')
                playlist_uuid = item.get('playlist', {}).get('uuid')
                entry_date_str = item.get('entryDate')

                # --- MODIFICATION ---
                # Process this item only if we haven't seen its playlist_uuid before
                if playlist_uuid and playlist_uuid not in processed_playlist_uuids:
                    if all([song_uuid, entry_date_str]):
                        try:
                            entry_date = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date()
                            start_date_for_song = entry_date - timedelta(days=90)
                            end_date_for_song = entry_date

                            # Append a task for this unique playlist entry
                            tasks_to_run.append({
                                'song_uuid': song_uuid,
                                'playlist_uuid': playlist_uuid, 
                                'start': start_date_for_song,
                                'end': end_date_for_song
                            })
                            # ADD: Add the playlist_uuid to the set to prevent duplicates
                            processed_playlist_uuids.add(playlist_uuid)
                        except ValueError:
                            continue
        
        # --- MODIFIED: Execute parallel fetch for each unique playlist song entry ---
        song_streaming_data = [] 
        if tasks_to_run:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_task = {
                    executor.submit(self.get_song_streaming_audience, task['song_uuid'], "spotify", task['start'], task['end']): task
                    for task in tasks_to_run
                }
                
                # This print statement will now correctly show the count of unique playlists (e.g., 68)
                print(f"Fetching streaming data for {len(future_to_task)} unique playlist entries...")
                for future in as_completed(future_to_task):
                    task_info = future_to_task[future]
                    try:
                        data = future.result()
                        # Append a result dict containing all necessary info for storage
                        if 'error' not in data and 'items' in data:
                            song_streaming_data.append({
                                'song_uuid': task_info['song_uuid'],
                                'playlist_uuid': task_info['playlist_uuid'],
                                'data': data
                            })
                    except Exception as e:
                        print(f"Task {task_info} generated an exception during fetch: {e}")

        result['song_streaming_data'] = song_streaming_data
        return result
