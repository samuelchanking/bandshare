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

# Step 3: Prompt user for Chartmetric Track ID
try:
    track_cmid = input("Enter the Chartmetric Track ID (CMID): ").strip()
    track_cmid = int(track_cmid)  # Ensure it's an integer
except ValueError:
    print("ERROR: Invalid CMID. Please enter a valid integer.")
    exit(1)

# Step 4: Fetch track stats with specified parameters
platform = "chartmetric"  # Required: default platform
mode = "highest-playcounts"  # Required: default mode
uri = f'/api/track/{track_cmid}/{platform}/stats/{mode}'
print(f"Fetching stats for track ID {track_cmid} (platform={platform}, mode={mode})")
res = Get(uri)
if res.status_code == 404:
    print(f"ERROR: Received a 404 Not Found for track ID {track_cmid}. The CMID might be invalid or no stats are available for {platform} in {mode} mode.")
    print(f"Response text: {res.text}")
    exit(1)
elif res.status_code != 200:
    print(f"ERROR: Received a {res.status_code} instead of 200 from {uri}")
    print(f"Response text: {res.text}")
    exit(1)

# Step 5: Process the track stats data
print(f"Raw response text: {res.text}")  # Log raw response for debugging
try:
    stats_data = res.json()
except json.JSONDecodeError:
    print(f"ERROR: Failed to parse response as JSON. Response: {res.text}")
    exit(1)

if not stats_data:
    print(f"ERROR: No stats data returned for track ID {track_cmid} on {platform} in {mode} mode.")
    exit(1)

# Step 6: Save the track stats to a JSON file
try:
    with open(f'track_stats_{track_cmid}_{platform}_{mode}.json', 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=4)
    print(f"Successfully saved track stats to track_stats_{track_cmid}_{platform}_{mode}.json")
except Exception as e:
    print(f"ERROR: Failed to save track stats: {str(e)}")
    exit(1)

# Step 7: Display the track stats
print("Track Stats:")
print(json.dumps(stats_data, indent=4))