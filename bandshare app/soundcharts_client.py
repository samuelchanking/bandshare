# soundcharts_client.py

import requests
import urllib.parse
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, date

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
            
    def _fetch_song_streaming_in_chunks(self, song_uuid: str, start_date: date, end_date: date) -> list:
        """Helper to fetch data for a date range > 90 days by splitting into chunks."""
        all_items = []
        current_start = start_date
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            while current_start <= end_date:
                current_end = current_start + timedelta(days=89)
                if current_end > end_date:
                    current_end = end_date
                
                futures.append(executor.submit(self.get_song_streaming_audience, song_uuid, "spotify", current_start, current_end))
                current_start = current_end + timedelta(days=1)

            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    if 'error' not in chunk_result and 'items' in chunk_result:
                        all_items.extend(chunk_result['items'])
                except Exception as e:
                    print(f"A chunk fetch failed: {e}")
        return all_items

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

    def get_local_audience(self, artist_uuid: str, platform: str = "instagram") -> Dict[str, Any]:
        """Fetches local audience data for a given platform. This is not a time-series."""
        url = f"{self.BASE_URL}/v2.37/artist/{artist_uuid}/social/{platform}/followers/"
        return self._request(url)

    def get_local_streaming_audience(self, artist_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        """Fetches local streaming audience data for a given platform."""
        url = f"{self.BASE_URL}/v2/artist/{artist_uuid}/streaming/{platform}"
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

    def fetch_secondary_artist_data(self, artist_uuid: str) -> Dict[str, Any]:
        """
        Fetches albums, playlists, and all related song metadata.
        NOW INCLUDES two types of song streaming data:
        1. song_streaming_data: Pre/post data for individual playlist entries (existing).
        2. song_centric_streaming: Aggregated data for unique songs across all their playlists.
        """
        result = {}
        result['albums'] = self.get_artist_albums(artist_uuid)
        result['playlists'] = self.get_artist_playlists(artist_uuid)

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

        playlist_items = result.get('playlists', {}).get('items', [])
        
        # --- BLOCK 1: Per-Playlist-Entry Data (Existing Logic) ---
        tasks_to_run_individual = []
        processed_playlist_uuids = set()

        if playlist_items:
            today = date.today()
            for item in playlist_items:
                song_uuid = item.get('song', {}).get('uuid')
                playlist_uuid = item.get('playlist', {}).get('uuid')
                
                if playlist_uuid and playlist_uuid not in processed_playlist_uuids:
                    if song_uuid and (entry_date_str := item.get('entryDate')):
                        try:
                            entry_date_dt = datetime.fromisoformat(entry_date_str.replace('Z', '+00:00')).date()
                            
                            pre_start_date = entry_date_dt - timedelta(days=90)
                            tasks_to_run_individual.append({'song_uuid': song_uuid, 'playlist_uuid': playlist_uuid, 'start': pre_start_date, 'end': entry_date_dt, 'type': 'pre_entry'})
                            
                            post_start_date = entry_date_dt + timedelta(days=1)
                            if post_start_date <= today:
                                post_end_date = post_start_date + timedelta(days=89)
                                if post_end_date > today: post_end_date = today
                                tasks_to_run_individual.append({'song_uuid': song_uuid, 'playlist_uuid': playlist_uuid, 'start': post_start_date, 'end': post_end_date, 'type': 'post_entry'})

                            processed_playlist_uuids.add(playlist_uuid)
                        except ValueError:
                            continue
        
        grouped_results_individual = {}
        if tasks_to_run_individual:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_task = {executor.submit(self.get_song_streaming_audience, task['song_uuid'], "spotify", task['start'], task['end']): task for task in tasks_to_run_individual}
                for future in as_completed(future_to_task):
                    task_info = future_to_task[future]
                    key = (task_info['song_uuid'], task_info['playlist_uuid'])
                    grouped_results_individual.setdefault(key, {'pre_entry_data': None, 'post_entry_data': None})
                    try:
                        data = future.result()
                        if 'error' not in data:
                            if task_info['type'] == 'pre_entry': grouped_results_individual[key]['pre_entry_data'] = data
                            elif task_info['type'] == 'post_entry': grouped_results_individual[key]['post_entry_data'] = data
                    except Exception as e:
                        print(f"Task {task_info} generated an exception during fetch: {e}")
        
        result['song_streaming_data'] = [{'song_uuid': k[0], 'playlist_uuid': k[1], **v} for k, v in grouped_results_individual.items()]

        # --- BLOCK 2: Song-Centric Aggregated Data (New Logic) ---
        songs_to_process = {}
        for item in playlist_items:
            song_uuid = item.get('song', {}).get('uuid')
            if not song_uuid: continue
            
            if song_uuid not in songs_to_process:
                songs_to_process[song_uuid] = {'entry_dates': [], 'playlists': []}
            
            try:
                entry_date = datetime.fromisoformat(item['entryDate'].replace('Z', '+00:00')).date()
                songs_to_process[song_uuid]['entry_dates'].append(entry_date)
                songs_to_process[song_uuid]['playlists'].append({
                    'name': item.get('playlist', {}).get('name', 'N/A'),
                    'entryDate': str(entry_date)
                })
            except (ValueError, KeyError):
                continue

        song_centric_results = []
        for song_uuid, data in songs_to_process.items():
            if not data['entry_dates']: continue

            earliest_entry = min(data['entry_dates'])
            latest_entry = max(data['entry_dates'])
            
            fetch_start_date = earliest_entry - timedelta(days=90)
            fetch_end_date = latest_entry + timedelta(days=90)
            
            if fetch_end_date > date.today():
                fetch_end_date = date.today()
            
            full_history = self._fetch_song_streaming_in_chunks(song_uuid, fetch_start_date, fetch_end_date)
            
            seen_dates = set()
            unique_history = []
            for item in sorted(full_history, key=lambda x: x['date']):
                if item['date'] not in seen_dates:
                    unique_history.append(item)
                    seen_dates.add(item['date'])
            
            song_centric_results.append({
                'song_uuid': song_uuid,
                'history': unique_history,
                'playlists': data['playlists']
            })
            
        result['song_centric_streaming'] = song_centric_results
        
        return result
