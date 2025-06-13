# soundcharts_client.py

import requests
import urllib.parse
from typing import Dict, Any

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
        encoded_artist_name = urllib.parse.quote(artist_name)
        url = f"{self.BASE_URL}/v2/artist/search/{encoded_artist_name}"
        search_data = self._request(url, params={'limit': '1'})
        if 'error' in search_data: return search_data
        if not search_data.get('items'): return {'error': f"No artists found for '{artist_name}'"}
        return {'uuid': search_data['items'][0]['uuid']}

    def get_artist_metadata(self, artist_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.9/artist/{artist_uuid}"
        return self._request(url)

    def get_artist_albums(self, artist_uuid: str, start_date=None, end_date=None) -> Dict[str, Any]:
        """Fetches albums, optionally filtered by release date."""
        url = f"{self.BASE_URL}/v2.34/artist/{artist_uuid}/albums"
        params = {'offset': '0', 'limit': '100'}
        if start_date: params['releaseDateStart'] = str(start_date)
        if end_date: params['releaseDateEnd'] = str(end_date)
        return self._request(url, params=params)

    def get_album_tracks(self, album_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.26/album/{album_uuid}/tracks"
        return self._request(url)

    def get_album_metadata(self, album_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.36/album/by-uuid/{album_uuid}"
        return self._request(url)
        
    def get_song_metadata(self, song_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.25/song/{song_uuid}"
        return self._request(url)

    def get_artist_playlist_entries(self, artist_uuid: str, start_date=None, end_date=None, platform: str = "spotify") -> Dict[str, Any]:
        """Fetches playlist entries for an artist, filtered by date."""
        url = f"{self.BASE_URL}/v2.20/artist/{artist_uuid}/playlist/entries/{platform}"
        params = {}
        if start_date: params['startDate'] = str(start_date)
        if end_date: params['endDate'] = str(end_date)
        return self._request(url, params=params)

    def fetch_full_artist_data(self, artist_name: str, start_date=None, end_date=None) -> Dict[str, Any]:
        artist_info = self.search_artist(artist_name)
        if 'error' in artist_info: return artist_info
        
        artist_uuid = artist_info['uuid']
        result = {'artist_uuid': artist_uuid, 'artist_name': artist_name}

        result['metadata'] = self.get_artist_metadata(artist_uuid)
        # Pass dates to the album and playlist fetchers
        result['albums'] = self.get_artist_albums(artist_uuid, start_date, end_date)
        result['playlists'] = self.get_artist_playlist_entries(artist_uuid, start_date, end_date)

        if 'albums' in result and result.get('albums', {}).get('items'):
            result['tracklists'] = {}
            result['album_metadata'] = {}
            result['song_metadata'] = {}
            
            for album in result['albums']['items']:
                album_uuid = album['uuid']
                tracklist_data = self.get_album_tracks(album_uuid)
                result['tracklists'][album_uuid] = tracklist_data
                result['album_metadata'][album_uuid] = self.get_album_metadata(album_uuid)

                if 'items' in tracklist_data:
                    result['song_metadata'][album_uuid] = {}
                    for item in tracklist_data['items']:
                        if song_uuid := item.get('song', {}).get('uuid'):
                            result['song_metadata'][album_uuid][song_uuid] = self.get_song_metadata(song_uuid)
        return result
