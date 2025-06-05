import requests
import json
import urllib.parse
import re
import time

headers = {
    'x-app-id': 'MANCHESTER_696DCD6E',
    'x-api-key': '11ed17a6cf25afa4',
}

# Prompt for artist name
artist_name = input("Enter artist name to search (e.g., Billie Eilish): ")

# Sanitize artist name for filename (replace spaces and special chars with underscores)
sanitized_artist_name = re.sub(r'[^a-zA-Z0-9]', '_', artist_name).lower()

# Step 1: Search for artist and get the first result's UUID
search_params = {
    'offset': '0',
    'limit': '1',  # Only need the first result
}
encoded_artist_name = urllib.parse.quote(artist_name)
search_response = requests.get(
    f'https://customer.api.soundcharts.com/api/v2/artist/search/{encoded_artist_name}',
    params=search_params,
    headers=headers,
)

if search_response.status_code != 200:
    print(f"Error searching for artist: Status code {search_response.status_code}")
    try:
        print("Error details:", json.dumps(search_response.json(), indent=2))
    except ValueError:
        print("No additional error details available")
    exit()

search_data = search_response.json()
if 'items' not in search_data or not search_data['items']:
    print(f"No artists found for '{artist_name}'")
    exit()

artist_uuid = search_data['items'][0]['uuid']
print(f"Using artist UUID: {artist_uuid}")

# Step 2: Fetch data for multiple endpoints and combine
combined_data = {}
endpoints = [
    {
        'name': 'metadata',
        'url': f'https://customer.api.soundcharts.com/api/v2/artist/{artist_uuid}',
        'params': {},
        'file_key': 'metadata',
    },
    {
        'name': 'audience',
        'url': f'https://customer.api.soundcharts.com/api/v2/artist/{artist_uuid}/audience/spotify',
        'params': {'startDate': '2025-03-01', 'endDate': '2025-05-30'},  # Adjusted to past dates
        'file_key': 'audience',
    },
    {
        'name': 'albums',
        'url': f'https://customer.api.soundcharts.com/api/v2.34/artist/{artist_uuid}/albums',
        'params': {'offset': '0', 'limit': '100'},
        'file_key': 'albums',
    },
    {
        'name': 'songs',  # Tracks and songs use the same endpoint
        'url': f'https://customer.api.soundcharts.com/api/v2.21/artist/{artist_uuid}/songs',
        'params': {'offset': '0', 'limit': '100'},
        'file_key': 'songs',
    },
    {
        'name': 'popularity',
        'url': f'https://customer.api.soundcharts.com/api/v2/artist/{artist_uuid}/popularity/spotify',
        'params': {'startDate': '2025-03-01', 'endDate': '2025-05-30'},
        'file_key': 'popularity',
    },
]

for endpoint in endpoints:
    response = requests.get(
        endpoint['url'],
        params=endpoint['params'],
        headers=headers,
    )
    
    # Check response
    if response.status_code == 200:
        data = response.json()
        combined_data[endpoint['file_key']] = data
        print(f"{endpoint['name'].capitalize()} data fetched successfully")
        
        # Print sample of data (if applicable)
        try:
            if 'data' in data and isinstance(data['data'], list):
                print(f"Sample {endpoint['name']} data (first 5 entries):")
                for entry in data['data'][:5]:
                    print(json.dumps(entry, indent=2))
            else:
                print(f"{endpoint['name'].capitalize()} response structure:", json.dumps(data, indent=2))
        except KeyError:
            print(f"Unexpected {endpoint['name']} response structure:", json.dumps(data, indent=2))
    else:
        print(f"Error fetching {endpoint['name']}: Status code {response.status_code}")
        try:
            error_details = response.json()
            print("Error details:", json.dumps(error_details, indent=2))
        except ValueError:
            print("No additional error details available")
        combined_data[endpoint['file_key']] = {'error': f"Failed to fetch: Status {response.status_code}"}

# Step 3: Fetch album data if albums are available
combined_album_data = {}
if 'albums' in combined_data and 'items' in combined_data['albums']:
    album_uuids = [album['uuid'] for album in combined_data['albums']['items']]
    print(f"Found {len(album_uuids)} albums to process")

    album_endpoints = [
        {
            'name': 'metadata',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2.36/album/by-uuid/{uuid}',
            'params': {},
            'data_key': 'metadata',
        },
        {
            'name': 'streams',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2/album/{uuid}/audience/spotify',
            'params': {'startDate': '2025-03-01', 'endDate': '2025-05-30'},
            'data_key': 'streams',
        },
        {
            'name': 'tracklist',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2.26/album/{uuid}/tracks',
            'params': {},
            'data_key': 'tracklist',
        },
    ]

    for album_uuid in album_uuids:
        combined_album_data[album_uuid] = {}
        for endpoint in album_endpoints:
            response = requests.get(
                endpoint['url'](album_uuid),
                params=endpoint['params'],
                headers=headers,
            )
            
            if response.status_code == 200:
                data = response.json()
                combined_album_data[album_uuid][endpoint['data_key']] = data
                print(f"Album {album_uuid} {endpoint['name']} data fetched successfully")
                
                try:
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"Sample {endpoint['name']} data for album {album_uuid} (first 5 entries):")
                        for entry in data['data'][:5]:
                            print(json.dumps(entry, indent=2))
                    else:
                        print(f"{endpoint['name'].capitalize()} response structure for album {album_uuid}:", json.dumps(data, indent=2))
                except KeyError:
                    print(f"Unexpected {endpoint['name']} response structure for album {album_uuid}:", json.dumps(data, indent=2))
            else:
                print(f"Error fetching {endpoint['name']} for album {album_uuid}: Status code {response.status_code}")
                try:
                    error_details = response.json()
                    print("Error details:", json.dumps(error_details, indent=2))
                except ValueError:
                    print("No additional error details available")
                combined_album_data[album_uuid][endpoint['data_key']] = {'error': f"Failed to fetch: Status {response.status_code}"}

# Step 4: Fetch song data if songs are available
combined_song_data = {}
if 'songs' in combined_data and 'items' in combined_data['songs'] and isinstance(combined_data['songs']['items'], list):
    song_uuids = [song['uuid'] for song in combined_data['songs']['items'] if 'uuid' in song]
    print(f"Found {len(song_uuids)} songs to process")

    song_endpoints = [
        {
            'name': 'metadata',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2.25/song/{uuid}',
            'params': {},
            'data_key': 'metadata',
        },
        {
            'name': 'audience',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2/song/{uuid}/audience/spotify',
            'params': {'startDate': '2025-03-01', 'endDate': '2025-05-30'},
            'data_key': 'audience',
        },
        {
            'name': 'popularity',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2/song/{uuid}/spotify/identifier/popularity',
            'params': {'startDate': '2025-03-01', 'endDate': '2025-05-30'},
            'data_key': 'popularity',
        },
    ]

    for song_uuid in song_uuids:
        combined_song_data[song_uuid] = {}
        for endpoint in song_endpoints:
            response = requests.get(
                endpoint['url'](song_uuid),
                params=endpoint['params'],
                headers=headers,
            )
            time.sleep(1)  # Add delay to avoid rate limiting
            
            if response and response.status_code == 200:
                data = response.json()
                combined_song_data[song_uuid][endpoint['data_key']] = data
                print(f"Song {song_uuid} {endpoint['name']} data fetched successfully")
                
                try:
                    if 'data' in data and isinstance(data['data'], list):
                        print(f"Sample {endpoint['name']} data for song {song_uuid} (first 5 entries):")
                        for entry in data['data'][:5]:
                            print(json.dumps(entry, indent=2))
                    else:
                        print(f"{endpoint['name'].capitalize()} response structure for song {song_uuid}:", json.dumps(data, indent=2))
                except KeyError:
                    print(f"Unexpected {endpoint['name']} response structure for song {song_uuid}:", json.dumps(data, indent=2))
            else:
                print(f"Error fetching {endpoint['name']} for song {song_uuid}: Status code {response.status_code if response else 'N/A'}")
                try:
                    error_details = response.json()
                    print("Error details:", json.dumps(error_details, indent=2))
                except (ValueError, AttributeError):
                    print("No additional error details available")
                combined_song_data[song_uuid][endpoint['data_key']] = {'error': f"Failed to fetch: Status {response.status_code if response else 'N/A'}"}

# Step 5: Export combined data to separate JSON files
artist_output_file = f'artist_data_{sanitized_artist_name}.json'
with open(artist_output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=4)
print(f"Combined artist data exported to {artist_output_file}")

album_output_file = f'album_data_{sanitized_artist_name}.json'
with open(album_output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_album_data, f, ensure_ascii=False, indent=4)
print(f"Combined album data exported to {album_output_file}")

song_output_file = f'song_data_{sanitized_artist_name}.json'
with open(song_output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_song_data, f, ensure_ascii=False, indent=4)
print(f"Combined song data exported to {song_output_file}")
