import requests
import json
import urllib.parse
import re
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime, timedelta


headers = {
    'x-app-id': 'MANCHESTER_696DCD6E',
    'x-api-key': '11ed17a6cf25afa4',
}

# Connect to MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['soundcharts3']
    # Artist-related collections
    artists_collection = db['artists']
    audience_collection = db['audience']
    popularity_collection = db['popularity']
    # Album-related collections
    albums_collection = db['albums']
    streams_collection = db['streams']
    tracklist_collection = db['tracklist']
    # Song-related collections
    songs_collection = db['songs']
    songs_audience_collection = db['songs_audience']
    songs_popularity_collection = db['songs_popularity']
    print("Connected to MongoDB database 'soundcharts'")
except ConnectionError:
    print("Failed to connect to MongoDB. Ensure MongoDB is running on localhost:27017")
    exit()

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
    exit()

search_data = search_response.json()
if 'items' not in search_data or not search_data['items']:
    print(f"No artists found for '{artist_name}'")
    exit()

artist_uuid = search_data['items'][0]['uuid']
print(f"Using artist UUID: {artist_uuid}")

# Function to generate 90-day periods for 2 years
def get_date_periods(start_date, total_days=730):
    periods = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = current_date - timedelta(days=total_days)
    while current_date > end_date:
        period_end = current_date
        period_start = period_end - timedelta(days=89)  # 90 days including end date
        if period_start < end_date:
            period_start = end_date
        periods.append((period_start.strftime('%Y-%m-%d'), period_end.strftime('%Y-%m-%d')))
        current_date = period_start - timedelta(days=1)
    return periods[::-1]

# Generate periods starting from 2025-05-30 back to cover 2 years
date_periods = get_date_periods('2025-05-30', 730)
print(f"Fetching timestamp data for {len(date_periods)} periods covering 2 years")

# Step 2: Fetch data for artist endpoints
combined_data = {}
endpoints = [
    {
        'name': 'metadata',
        'url': f'https://customer.api.soundcharts.com/api/v2.9/artist/{artist_uuid}',
        'params': {},
        'file_key': 'metadata',
    },
    {
        'name': 'audience',
        'url': f'https://customer.api.soundcharts.com/api/v2/artist/{artist_uuid}/audience/spotify',
        'params': {'startDate': '', 'endDate': ''},  # Will be set dynamically
        'file_key': 'audience',
    },
    {
        'name': 'albums',
        'url': f'https://customer.api.soundcharts.com/api/v2.34/artist/{artist_uuid}/albums',
        'params': {'offset': '0', 'limit': '100'},
        'file_key': 'albums',
    },
    {
        'name': 'songs',
        'url': f'https://customer.api.soundcharts.com/api/v2.21/artist/{artist_uuid}/songs',
        'params': {'offset': '0', 'limit': '100'},
        'file_key': 'songs',
    },
    {
        'name': 'popularity',
        'url': f'https://customer.api.soundcharts.com/api/v2/artist/{artist_uuid}/popularity/spotify',
        'params': {'startDate': '', 'endDate': ''},  # Will be set dynamically
        'file_key': 'popularity',
    },
]

# Fetch non-timestamp data for artist (metadata, albums, songs) once
for endpoint in endpoints:
    if endpoint['file_key'] in ['metadata', 'albums', 'songs']:
        response = requests.get(
            endpoint['url'],
            params=endpoint['params'],
            headers=headers,
        )
        
        if response.status_code == 200:
            combined_data[endpoint['file_key']] = response.json()
            print(f"Fetched {endpoint['name']} data successfully")
        else:
            print(f"Error fetching {endpoint['name']}: Status code {response.status_code}")
            combined_data[endpoint['file_key']] = {'error': f"Failed to fetch: Status {response.status_code}"}

# Store artist metadata in the artists collection
artist_metadata = combined_data.get('metadata', {})
artist_metadata['artist_uuid'] = artist_uuid
try:
    if artists_collection.find_one({'artist_uuid': artist_uuid}):
        artists_collection.update_one(
            {'artist_uuid': artist_uuid},
            {'$set': artist_metadata}
        )
        print(f"Updated artist metadata for {artist_uuid} in 'artists' collection")
    else:
        artists_collection.insert_one(artist_metadata)
        print(f"Inserted artist metadata for {artist_uuid} into 'artists' collection")
except OperationFailure as e:
    print(f"Failed to store artist metadata in MongoDB: {e}")
    exit()

# Fetch and store timestamp data for artist (audience, popularity) for all periods
audience_data = {'artist_uuid': artist_uuid, 'data': []}
popularity_data = {'artist_uuid': artist_uuid, 'data': []}
for i, (start_date, end_date) in enumerate(date_periods, 1):
    for endpoint in endpoints:
        if endpoint['file_key'] in ['audience', 'popularity']:
            endpoint['params']['startDate'] = start_date
            endpoint['params']['endDate'] = end_date
            print(f"Requesting {endpoint['name']} for period {i}: {endpoint['url']} with params {endpoint['params']}")
            response = requests.get(
                endpoint['url'],
                params=endpoint['params'],
                headers=headers,
            )
            print(f"Response status: {response.status_code}, Body: {response.text}")
            if response.status_code == 200:
                data = response.json()
                target_data = audience_data if endpoint['file_key'] == 'audience' else popularity_data
                if 'data' in data and isinstance(data['data'], list):
                    target_data['data'].extend(data['data'])
                elif 'items' in data and isinstance(data['items'], list):  # Handle 'items' key
                    target_data['data'].extend(data['items'])
                else:
                    print(f"Unexpected response in period {i}: {data}")
                    target_data['error'] = f"Unexpected response structure in period {i}"
            elif response.status_code == 404:
                print(f"No data available for {endpoint['name']} in period {i} ({start_date} to {end_date})")
            else:
                print(f"Error fetching {endpoint['name']} for period {i} ({start_date} to {end_date}): Status code {response.status_code}")
                target_data['error'] = f"Failed to fetch period {i}: Status {response.status_code}"
            print(f"Fetched {endpoint['name']} data for period {i} ({start_date} to {end_date}) successfully")

# Store audience and popularity data
try:
    if audience_collection.find_one({'artist_uuid': artist_uuid}):
        audience_collection.update_one(
            {'artist_uuid': artist_uuid},
            {'$set': audience_data}
        )
        print(f"Updated audience data for {artist_uuid} in 'audience' collection")
    else:
        audience_collection.insert_one(audience_data)
        print(f"Inserted audience data for {artist_uuid} into 'audience' collection")
except OperationFailure as e:
    print(f"Failed to store audience data in MongoDB: {e}")
    exit()

try:
    if popularity_collection.find_one({'artist_uuid': artist_uuid}):
        popularity_collection.update_one(
            {'artist_uuid': artist_uuid},
            {'$set': popularity_data}
        )
        print(f"Updated popularity data for {artist_uuid} in 'popularity' collection")
    else:
        popularity_collection.insert_one(popularity_data)
        print(f"Inserted popularity data for {artist_uuid} into 'popularity' collection")
except OperationFailure as e:
    print(f"Failed to store popularity data in MongoDB: {e}")
    exit()

# Step 3: Fetch album data if albums are available
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
            'params': {'startDate': '', 'endDate': ''},  # Will be set dynamically
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
        album_metadata = {'album_uuid': album_uuid, 'artist_uuid': artist_uuid}
        tracklist_data = {'album_uuid': album_uuid, 'artist_uuid': artist_uuid}
        streams_data = {'album_uuid': album_uuid, 'artist_uuid': artist_uuid, 'data': []}

        # Fetch non-timestamp data first (metadata, tracklist)
        for endpoint in album_endpoints:
            if endpoint['data_key'] not in ['streams']:
                response = requests.get(
                    endpoint['url'](album_uuid),
                    params=endpoint['params'],
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if endpoint['data_key'] == 'metadata':
                        album_metadata.update(data)
                    elif endpoint['data_key'] == 'tracklist':
                        tracklist_data.update(data)
                    print(f"Fetched {endpoint['name']} data for album {album_uuid} successfully")
                else:
                    print(f"Error fetching {endpoint['name']} for album {album_uuid}: Status code {response.status_code}")
                    if endpoint['data_key'] == 'metadata':
                        album_metadata['error'] = f"Failed to fetch: Status {response.status_code}"
                    elif endpoint['data_key'] == 'tracklist':
                        tracklist_data['error'] = f"Failed to fetch: Status {response.status_code}"

        # Fetch timestamp data (streams) for all periods
        for i, (start_date, end_date) in enumerate(date_periods, 1):
            for endpoint in album_endpoints:
                if endpoint['data_key'] == 'streams':
                    endpoint['params']['startDate'] = start_date
                    endpoint['params']['endDate'] = end_date
                    print(f"Requesting {endpoint['name']} for period {i}: {endpoint['url'](album_uuid)} with params {endpoint['params']}")
                    response = requests.get(
                        endpoint['url'](album_uuid),
                        params=endpoint['params'],
                        headers=headers,
                    )
                    print(f"Response status: {response.status_code}, Body: {response.text}")
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data and isinstance(data['data'], list):
                            streams_data['data'].extend(data['data'])
                        elif 'items' in data and isinstance(data['items'], list):  # Handle 'items' key
                            streams_data['data'].extend(data['items'])
                        else:
                            print(f"Unexpected response in period {i}: {data}")
                            streams_data['error'] = f"Unexpected response structure in period {i}"
                    elif response.status_code == 404:
                        print(f"No data available for {endpoint['name']} in period {i} ({start_date} to {end_date})")
                    else:
                        print(f"Error fetching {endpoint['name']} for period {i} ({start_date} to {end_date}): Status code {response.status_code}")
                        streams_data['error'] = f"Failed to fetch period {i}: Status {response.status_code}"
                    print(f"Fetched {endpoint['name']} data for period {i} ({start_date} to {end_date}) successfully")

        # Store album-related data
        try:
            if albums_collection.find_one({'album_uuid': album_uuid}):
                albums_collection.update_one(
                    {'album_uuid': album_uuid},
                    {'$set': album_metadata}
                )
                print(f"Updated album metadata for {album_uuid} in 'albums' collection")
            else:
                albums_collection.insert_one(album_metadata)
                print(f"Inserted album metadata for {album_uuid} into 'albums' collection")
        except OperationFailure as e:
            print(f"Failed to store album metadata in MongoDB: {e}")
            exit()

        try:
            if streams_collection.find_one({'album_uuid': album_uuid}):
                streams_collection.update_one(
                    {'album_uuid': album_uuid},
                    {'$set': streams_data}
                )
                print(f"Updated streams data for album {album_uuid} in 'streams' collection")
            else:
                streams_collection.insert_one(streams_data)
                print(f"Inserted streams data for album {album_uuid} into 'streams' collection")
        except OperationFailure as e:
            print(f"Failed to store streams data in MongoDB: {e}")
            exit()

        try:
            if tracklist_collection.find_one({'album_uuid': album_uuid}):
                tracklist_collection.update_one(
                    {'album_uuid': album_uuid},
                    {'$set': tracklist_data}
                )
                print(f"Updated tracklist data for album {album_uuid} in 'tracklist' collection")
            else:
                tracklist_collection.insert_one(tracklist_data)
                print(f"Inserted tracklist data for album {album_uuid} into 'tracklist' collection")
        except OperationFailure as e:
            print(f"Failed to store tracklist data in MongoDB: {e}")
            exit()

# Step 4: Fetch song data if songs are available
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
            'params': {'startDate': '', 'endDate': ''},  # Will be set dynamically
            'data_key': 'audience',
        },
        {
            'name': 'popularity',
            'url': lambda uuid: f'https://customer.api.soundcharts.com/api/v2/song/{uuid}/spotify/identifier/popularity',
            'params': {'startDate': '', 'endDate': ''},  # Will be set dynamically
            'data_key': 'popularity',
        },
    ]

    for song_uuid in song_uuids:
        song_metadata = {'song_uuid': song_uuid, 'artist_uuid': artist_uuid}
        song_audience_data = {'song_uuid': song_uuid, 'artist_uuid': artist_uuid, 'data': []}
        song_popularity_data = {'song_uuid': song_uuid, 'artist_uuid': artist_uuid, 'data': []}

        # Fetch non-timestamp data first (metadata)
        for endpoint in song_endpoints:
            if endpoint['data_key'] not in ['audience', 'popularity']:
                response = requests.get(
                    endpoint['url'](song_uuid),
                    params=endpoint['params'],
                    headers=headers,
                )
                time.sleep(1)  # Add delay to avoid rate limiting
                
                if response.status_code == 200:
                    song_metadata.update(response.json())
                    print(f"Fetched {endpoint['name']} data for song {song_uuid} successfully")
                else:
                    print(f"Error fetching {endpoint['name']} for song {song_uuid}: Status code {response.status_code}")
                    song_metadata['error'] = f"Failed to fetch: Status {response.status_code}"

        # Fetch timestamp data (audience, popularity) for all periods
        for i, (start_date, end_date) in enumerate(date_periods, 1):
            for endpoint in song_endpoints:
                if endpoint['data_key'] in ['audience', 'popularity']:
                    endpoint['params']['startDate'] = start_date
                    endpoint['params']['endDate'] = end_date
                    print(f"Requesting {endpoint['name']} for period {i}: {endpoint['url'](song_uuid)} with params {endpoint['params']}")
                    response = requests.get(
                        endpoint['url'](song_uuid),
                        params=endpoint['params'],
                        headers=headers,
                    )
                    time.sleep(1)  # Add delay to avoid rate limiting
                    print(f"Response status: {response.status_code}, Body: {response.text}")
                    if response.status_code == 200:
                        data = response.json()
                        target_data = song_audience_data if endpoint['data_key'] == 'audience' else song_popularity_data
                        if 'data' in data and isinstance(data['data'], list):
                            target_data['data'].extend(data['data'])
                        elif 'items' in data and isinstance(data['items'], list):  # Handle 'items' key
                            target_data['data'].extend(data['items'])
                        else:
                            print(f"Unexpected response in period {i}: {data}")
                            target_data['error'] = f"Unexpected response structure in period {i}"
                    elif response.status_code == 404:
                        print(f"No data available for {endpoint['name']} in period {i} ({start_date} to {end_date})")
                    else:
                        print(f"Error fetching {endpoint['name']} for period {i} ({start_date} to {end_date}): Status code {response.status_code}")
                        target_data['error'] = f"Failed to fetch period {i}: Status {response.status_code}"
                    print(f"Fetched {endpoint['name']} data for period {i} ({start_date} to {end_date}) successfully")

        # Store song-related data
        try:
            if songs_collection.find_one({'song_uuid': song_uuid}):
                songs_collection.update_one(
                    {'song_uuid': song_uuid},
                    {'$set': song_metadata}
                )
                print(f"Updated song metadata for {song_uuid} in 'songs' collection")
            else:
                songs_collection.insert_one(song_metadata)
                print(f"Inserted song metadata for {song_uuid} into 'songs' collection")
        except OperationFailure as e:
            print(f"Failed to store song metadata in MongoDB: {e}")
            exit()

        try:
            if songs_audience_collection.find_one({'song_uuid': song_uuid}):
                songs_audience_collection.update_one(
                    {'song_uuid': song_uuid},
                    {'$set': song_audience_data}
                )
                print(f"Updated audience data for song {song_uuid} in 'songs_audience' collection")
            else:
                songs_audience_collection.insert_one(song_audience_data)
                print(f"Inserted audience data for song {song_uuid} into 'songs_audience' collection")
        except OperationFailure as e:
            print(f"Failed to store song audience data in MongoDB: {e}")
            exit()

        try:
            if songs_popularity_collection.find_one({'song_uuid': song_uuid}):
                songs_popularity_collection.update_one(
                    {'song_uuid': song_uuid},
                    {'$set': song_popularity_data}
                )
                print(f"Updated popularity data for song {song_uuid} in 'songs_popularity' collection")
            else:
                songs_popularity_collection.insert_one(song_popularity_data)
                print(f"Inserted popularity data for song {song_uuid} into 'songs_popularity' collection")
        except OperationFailure as e:
            print(f"Failed to store song popularity data in MongoDB: {e}")
            exit()

# Step 5: Close MongoDB connection
client.close()
print("MongoDB connection closed")
