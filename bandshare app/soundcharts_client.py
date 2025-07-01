# soundcharts_client.py

import requests
import urllib.parse
from typing import Dict, Any, Callable, List
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

    def _fetch_paginated_data(self, url: str) -> List[Any]:
        """Helper to fetch all items from a paginated API endpoint."""
        all_items = []
        next_page_url = url
        while next_page_url:
            page_data = self._request(next_page_url)
            if 'error' in page_data:
                # Stop pagination on error
                break
            all_items.extend(page_data.get('items', []))
            next_page_url = page_data.get('next')
        return all_items

    def _fetch_timeseries_in_chunks(self, url_builder: Callable[[date, date], str], start_date: date, end_date: date) -> Dict[str, Any]:
        """
        Generic helper to fetch time-series data for a date range > 90 days by splitting it into chunks.
        It calls a provided function to generate the specific API endpoint URL for each chunk.
        """
        all_items = []
        platform_or_source = None  # To store the platform/source from the first successful response
        current_start = start_date

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            while current_start <= end_date:
                current_end = current_start + timedelta(days=89)
                if current_end > end_date:
                    current_end = end_date
                
                def fetch_chunk(start, end):
                    url = url_builder(start, end)
                    params = {'startDate': str(start), 'endDate': str(end)}
                    return self._request(url, params=params)

                futures.append(executor.submit(fetch_chunk, current_start, current_end))
                current_start = current_end + timedelta(days=1)

            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    if 'error' not in chunk_result and 'items' in chunk_result:
                        all_items.extend(chunk_result['items'])
                        if not platform_or_source:
                            if 'platform' in chunk_result:
                                platform_or_source = {'platform': chunk_result.get('platform')}
                            elif 'source' in chunk_result:
                                platform_or_source = {'source': chunk_result.get('source')}

                except Exception as e:
                    print(f"A chunk fetch failed: {e}")

        final_response = {'items': all_items}
        if platform_or_source:
            final_response.update(platform_or_source)
        return final_response

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

    def get_artist_songs(self, artist_uuid: str) -> List[Any]:
        """Fetches a complete list of all songs for an artist."""
        url = f"{self.BASE_URL}/v2.21/artist/{artist_uuid}/songs"
        return self._fetch_paginated_data(url)

    def get_artist_audience(self, artist_uuid: str, platform: str, start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/artist/{artist_uuid}/audience/{platform}/"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date))

    def get_album_audience(self, album_uuid: str, platform: str, start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/album/{album_uuid}/audience/{platform}"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date))

    def get_local_audience(self, artist_uuid: str, platform: str = "instagram") -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.37/artist/{artist_uuid}/social/{platform}/followers/"
        return self._request(url)

    def get_local_streaming_audience(self, artist_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/artist/{artist_uuid}/streaming/{platform}"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date))

    def get_artist_popularity(self, artist_uuid: str, source: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/artist/{artist_uuid}/popularity/{source}"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date))

    def get_artist_albums(self, artist_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.34/artist/{artist_uuid}/albums"
        return self._request(url, params={'offset': '0', 'limit': '100'})

    def get_album_metadata(self, album_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.36/album/by-uuid/{album_uuid}"
        return self._request(url)

    def get_album_tracks(self, album_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.26/album/{album_uuid}/tracks"
        return self._request(url)

    def get_song_metadata(self, song_uuid: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.25/song/{song_uuid}"
        return self._request(url)

    def get_artist_playlists(self, artist_uuid: str, platform: str = "spotify") -> Dict[str, Any]:
        url = f"{self.BASE_URL}/v2.20/artist/{artist_uuid}/playlist/current/{platform}"
        return self._request(url)
    
    # --- MODIFIED: This function now takes a limit and does not paginate. ---
    def get_playlists_by_type(self, platform: str, playlist_type: str, limit: int = 20) -> List[Any]:
        """
        Fetches a list of top playlists for a given type, sorted by audience size.
        """
        url = f"{self.BASE_URL}/v2.20/playlist/by-type/{platform}/{playlist_type}"
        
        # Set params to sort by audience in descending order per the documentation
        params = {
            'limit': str(limit),
            'sortBy': 'audience',
            'sortOrder': 'desc'
        }
        
        data = self._request(url, params=params)
        return data.get('items', []) if 'error' not in data else []


    def get_playlist_audience(self, playlist_uuid: str, start_date=None, end_date=None) -> Dict[str, Any]:
        """Fetches time-series audience data for a playlist. Now supports chunking."""
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2.20/playlist/{playlist_uuid}/audience"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        
        params = {}
        if start_date: params['startDate'] = str(start_date)
        if end_date: params['endDate'] = str(end_date)
        return self._request(url_builder(start_date, end_date), params=params)
    
    # --- MODIFIED: This function now accepts additional filtering parameters. ---
    def get_song_playlist_entries(self, song_uuid: str, platform: str = "spotify", playlist_type: str = None, sort_by: str = None, limit: int = None) -> List[Any]:
        """
        Fetches a list of current playlist entries for a specific song, with optional filters.
        Handles pagination. The API will respect the limit parameter within pagination.
        """
        url = f"{self.BASE_URL}/v2.20/song/{song_uuid}/playlist/current/{platform}"
        params = {}
        if playlist_type:
            params['type'] = playlist_type
        if sort_by:
            params['sortBy'] = sort_by
        if limit:
            params['limit'] = str(limit)

        # Encode params into the URL for the paginator
        if params:
            url += "?" + urllib.parse.urlencode(params)

        # _fetch_paginated_data will handle the API calls. If a limit is set, the API
        # will only return that many items and 'next' will be null, stopping pagination naturally.
        return self._fetch_paginated_data(url)


    def get_artist_streaming_audience(self, artist_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/artist/{artist_uuid}/streaming/{platform}/listening"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date))

    def get_song_streaming_audience(self, song_uuid: str, platform: str = "spotify", start_date=None, end_date=None) -> Dict[str, Any]:
        url_builder = lambda sd, ed: f"{self.BASE_URL}/v2/song/{song_uuid}/audience/{platform}"
        if start_date and end_date:
            return self._fetch_timeseries_in_chunks(url_builder, start_date, end_date)
        return self._request(url_builder(start_date, end_date), params={'startDate': str(start_date), 'endDate': str(end_date)})

    def fetch_static_artist_data(self, artist_name: str) -> Dict[str, Any]:
        artist_info = self.search_artist(artist_name)
        if 'error' in artist_info: return artist_info

        artist_uuid = artist_info['uuid']
        result = {'artist_uuid': artist_uuid, 'artist_name': artist_name}
        result['metadata'] = self.get_artist_metadata(artist_uuid)
        return result

    def _fetch_and_process_song_data(self, song_uuid: str, data: dict) -> dict:
        if not data['entry_dates']: return {}

        processed_playlists = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_playlist = {}
            for p in data['playlists']:
                if p.get('uuid') and p.get('entryDate'):
                    entry_date_obj = datetime.fromisoformat(p['entryDate']).date()
                    future = executor.submit(self.get_playlist_audience, p['uuid'], entry_date_obj, entry_date_obj)
                    future_to_playlist[future] = p

            for future in as_completed(future_to_playlist):
                playlist_item = future_to_playlist[future]
                try:
                    audience_data = future.result()
                    entry_followers = None
                    if 'error' not in audience_data and audience_data.get('items'):
                        entry_followers = audience_data['items'][0].get('followerCount')
                    
                    playlist_item['entrySubscribers'] = entry_followers
                    processed_playlists.append(playlist_item)
                except Exception as e:
                    print(f"Failed to fetch audience for playlist {playlist_item.get('uuid')}: {e}")
                    processed_playlists.append(playlist_item)

        earliest_entry = min(data['entry_dates'])
        latest_entry = max(data['entry_dates'])
        
        fetch_start_date = earliest_entry - timedelta(days=90)
        fetch_end_date = latest_entry + timedelta(days=90)
        
        if fetch_end_date > date.today():
            fetch_end_date = date.today()
        
        full_history_response = self.get_song_streaming_audience(song_uuid, "spotify", fetch_start_date, fetch_end_date)
        full_history = full_history_response.get('items', [])
        
        seen_dates = set()
        unique_history = []
        for item in sorted(full_history, key=lambda x: x['date'], reverse=True):
            if item['date'] not in seen_dates:
                unique_history.insert(0, item)
                seen_dates.add(item['date'])
        
        return {
            'song_uuid': song_uuid,
            'history': unique_history,
            'playlists': processed_playlists
        }
        
    def get_playlist_tracks(self, playlist_uuid: str) -> List[Any]:
        """
        Fetches a complete, paginated list of all tracks for a given playlist.
        """
        url = f"{self.BASE_URL}/v2.20/playlist/{playlist_uuid}/tracks/latest"
        return self._fetch_paginated_data(url)
        
    
    # --- MODIFIED: This is the main orchestration function ---
    def fetch_secondary_artist_data(self, artist_uuid: str, platform: str = "spotify") -> Dict[str, Any]:
        """
        Fetches all secondary artist data. Song discovery relies on the artist songs
        endpoint, and playlist discovery is now done for each individual song to ensure
        complete and accurate data.
        """
        result = {}
        all_song_uuids = set()

        with ThreadPoolExecutor(max_workers=20) as executor:
            # Step 1: Concurrently fetch albums and the definitive list of artist songs
            future_albums = executor.submit(self.get_artist_albums, artist_uuid)
            future_artist_songs = executor.submit(self.get_artist_songs, artist_uuid)

            # Process definitive song list
            artist_songs_list = future_artist_songs.result()
            for song_item in artist_songs_list:
                if song_uuid := song_item.get('uuid'):
                    all_song_uuids.add(song_uuid)

            # Step 2: For each unique song, concurrently fetch its playlist entries.
            song_playlist_futures = {
                executor.submit(self.get_song_playlist_entries, song_uuid, platform): song_uuid
                for song_uuid in all_song_uuids
            }

            # --- NEW: Store playlist entries in a clean map {song_uuid: [entries]} ---
            song_playlist_map = {}
            for future in as_completed(song_playlist_futures):
                song_uuid = song_playlist_futures[future]
                try:
                    playlist_entries = future.result()
                    if playlist_entries:
                        song_playlist_map[song_uuid] = playlist_entries
                except Exception as e:
                    print(f"Failed to fetch playlist entries for song {song_uuid}: {e}")
            
            # Add the map to the final result dictionary
            result['song_playlist_map'] = song_playlist_map

            # Step 3: Concurrently fetch metadata for albums and songs (no changes to this logic)
            result['albums'] = future_albums.result()
            # ... existing logic for album_metadata, tracklists, and song_metadata ...
            result['album_metadata'] = {}
            result['tracklists'] = {}
            if 'items' in result.get('albums', {}):
                album_meta_futures = {
                    executor.submit(self.get_album_metadata, album.get('uuid')): album.get('uuid')
                    for album in result['albums']['items'] if album.get('uuid')
                }
                tracklist_futures = {
                    executor.submit(self.get_album_tracks, album.get('uuid')): album.get('uuid')
                    for album in result['albums']['items'] if album.get('uuid')
                }

                for future in as_completed(album_meta_futures):
                    album_uuid = album_meta_futures[future]
                    result['album_metadata'][album_uuid] = future.result()
                
                for future in as_completed(tracklist_futures):
                    album_uuid = tracklist_futures[future]
                    result['tracklists'][album_uuid] = future.result()

            result['song_metadata'] = {}
            song_meta_futures = {
                executor.submit(self.get_song_metadata, song_uuid): song_uuid
                for song_uuid in all_song_uuids
            }
            for future in as_completed(song_meta_futures):
                song_uuid = song_meta_futures[future]
                song_meta = future.result()
                if 'error' not in song_meta:
                    result['song_metadata'][song_uuid] = song_meta
        
        return result
