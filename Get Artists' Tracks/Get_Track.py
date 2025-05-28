import json
import time
from requests import get, post
from sys import exit

# Chartmetric API base URL
HOST = 'https://api.chartmetric.com'
TOKEN = 'eLi0rCADQ5iyoOdy3VQYsamjPz6tUZ1AYcZYeLpLhH582xO5G6guF6E4CHZRYpqQ'  # Replace with your refresh token

# Step 1: Get a new access token
try:
    res = post(f'{HOST}/api/token', json={"refreshtoken": TOKEN})
    if res.status_code != 200:
        print(f"ERROR: Received a {res.status_code} instead of 200 from /api/token")
        print(f"Response text: {res.text}")
        exit(1)
    access_token = res.json()['token']
    print(f"New access token: {access_token}")
except Exception as e:
    print(f"ERROR: Failed to get access token: {str(e)}")
    exit(1)

# Step 2: Custom GET function with retry logic
def Get(uri, max_retries=3):
    retries = 0
    base_delay = 60  # Initial delay of 60 seconds
    while retries < max_retries:
        try:
            response = get(f'{HOST}{uri}', headers={'Authorization': f'Bearer {access_token}'})
            if response.status_code == 429:  # Rate limit hit
                wait_time = base_delay * (2 ** retries)  # Exponential backoff
                print(f"WARNING: Rate limit reached for {uri}. Retrying ({retries+1}/{max_retries}) after {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                continue
            elif response.status_code == 401:  # Token expired
                print("ERROR: Access token expired (401 Unauthorized). Please refresh your token.")
                exit(1)
            elif response.status_code != 200:
                print(f"WARNING: Non-200 status code {response.status_code} for {uri}. Response: {response.text}")
                return response
            return response
        except Exception as e:
            print(f"WARNING: Network or request error for {uri}: {str(e)}. Retrying ({retries+1}/{max_retries})...")
            time.sleep(base_delay * (2 ** retries))
            retries += 1
    print(f"ERROR: Max retries ({max_retries}) reached for {uri}. Exiting...")
    exit(1)

# Step 3: Prompt user for Chartmetric Artist ID
try:
    artist_cmid = input("Enter the Chartmetric Artist ID (CMID): ").strip()
    artist_cmid = int(artist_cmid)  # Ensure it's an integer
except ValueError:
    print("ERROR: Invalid CMID. Please enter a valid integer.")
    exit(1)

# Step 4: Fetch tracks for the artist with default parameters
limit = 10  # Default value from documentation
offset = 0  # Default value from documentation
artist_type = "main"  # Optional: default to main artist, can be "featured" or omitted for both
uri = f'/api/artist/{artist_cmid}/tracks?limit={limit}&offset={offset}'
if artist_type:
    uri += f'&artist_type={artist_type}'
print(f"Fetching tracks for artist ID {artist_cmid} (limit={limit}, offset={offset}, artist_type={artist_type})")
res = Get(uri)
if res.status_code == 404:
    print(f"ERROR: Received a 404 Not Found for artist ID {artist_cmid}. The CMID might be invalid or the artist has no tracks.")
    print(f"Response text: {res.text}")
    exit(1)
elif res.status_code != 200:
    print(f"ERROR: Received a {res.status_code} instead of 200 from {uri}")
    print(f"Response text: {res.text}")
    exit(1)

# Step 5: Process the track data
track_data = res.json()
if not track_data:
    print(f"WARNING: No tracks found for artist ID {artist_cmid} with the given parameters.")
    exit(0)

# Step 6: Save the track data to a JSON file
try:
    with open(f'artist_tracks_{artist_cmid}.json', 'w', encoding='utf-8') as f:
        json.dump(track_data, f, indent=4)
    print(f"Successfully saved track data to artist_tracks_{artist_cmid}.json")
except Exception as e:
    print(f"ERROR: Failed to save track data: {str(e)}")
    exit(1)

# Step 7: Display the track data
print("Track Data:")
print(json.dumps(track_data, indent=4))
