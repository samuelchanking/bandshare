# soundcharts_client.py

import requests
import urllib.parse
from typing import Dict, Any
import json # Added for debugging

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
        
    # --- ADDED: The missing get_album_metadata function ---
    def get_album_metadata(self, album_uuid: str) -> Dict[str, Any]:
        """Fetches detailed metadata for a single album."""
        url = f"{self.BASE_URL}/v2.36/album/by-uuid/{album_uuid}"
        return self._request(url)

    def get_artist_playlists(self, artist_uuid: str, platform: str = "spotify") -> Dict[str, Any]:
        """Fetches the list of playlists an artist is currently featured on."""
        url = f"{self.BASE_URL}/v2.20/artist/{artist_uuid}/playlist/current/{platform}"
        return self._request(url)

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

    def fetch_secondary_artist_data(self, artist_uuid: str) -> Dict[str, Any]:
        """
        Fetches secondary data like albums and playlists, including detailed
        metadata for each album.
        """
        result = {}
        result['albums'] = self.get_artist_albums(artist_uuid)
        result['playlists'] = self.get_artist_playlists(artist_uuid)
        
        # --- ADDED: Fetch detailed metadata for each album ---
        if 'albums' in result and 'items' in result.get('albums', {}):
            result['album_metadata'] = {}
            for album_summary in result['albums']['items']:
                album_uuid_val = album_summary.get('uuid')
                if album_uuid_val:
                    result['album_metadata'][album_uuid_val] = self.get_album_metadata(album_uuid_val)
                    
        return result

    # --- Debugging function ---
    def debug_fetch_secondary_artist_data(self, artist_uuid: str, filename="debug_secondary_data_output.json"):
        """
        Calls fetch_secondary_artist_data and saves the raw JSON response to a file.
        This helps diagnose issues with the structure of album and playlist data.
        """
        print(f"--- [DEBUG] Fetching secondary data for artist UUID: {artist_uuid} ---")
        response_data = self.fetch_secondary_artist_data(artist_uuid)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)
            print(f"--- [SUCCESS] Raw JSON response saved to '{filename}' ---")
            
            if 'error' in response_data:
                print("--- [DIAGNOSIS] The API returned an error.")
            elif 'albums' in response_data and 'items' in response_data.get('albums', {}):
                print(f"--- [DIAGNOSIS] Found {len(response_data['albums']['items'])} album(s). Check the JSON file for their structure.")
                if 'album_metadata' in response_data:
                    print("--- [DIAGNOSIS] Successfully fetched detailed 'album_metadata'.")
                else:
                    print("--- [WARNING] Detailed 'album_metadata' key is missing.")
            else:
                 print("--- [DIAGNOSIS] The response structure is unexpected. Please review the JSON file's contents.")

        except Exception as e:
            print(f"--- [ERROR] Failed to save JSON file: {e} ---")
        
        return response_data

# --- Main execution block for direct debugging ---
if __name__ == '__main__':
    # This block allows you to run this file directly to debug a specific function.
    import config 

    # --- Configuration for Debugging ---
    # PASTE THE ARTIST UUID YOU WANT TO TEST HERE
    TEST_ARTIST_UUID = '11e81bd1-a865-34aa-b1fa-a0369fe50396'

    if not TEST_ARTIST_UUID or TEST_ARTIST_UUID == 'YOUR_ARTIST_UUID_HERE':
        print("\nERROR: Please open soundcharts_client.py and set the TEST_ARTIST_UUID at the bottom of the file.")
    else:
        print("\n--- Running Secondary Artist Data Debugger ---")
        client = SoundchartsAPIClient(app_id=config.APP_ID, api_key=config.API_KEY)
        client.debug_fetch_secondary_artist_data(TEST_ARTIST_UUID)
        print("--- Debug script finished. ---")
