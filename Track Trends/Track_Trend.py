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

# Step 2: Custom GET function with retry logic and 1-second interval between requests
def Get(uri, max_retries=5):
    retries = 0
    base_delay = 120  # Initial delay of 120 seconds for server recovery (for 429/504 retries)
    while retries < max_retries:
        try:
            response = get(f'{HOST}{uri}', headers={'Authorization': f'Bearer {access_token}'})
            # Add 1-second delay after each request attempt
            time.sleep(1)
            if response.status_code in [429, 504]:  # Rate limit or Gateway Timeout
                wait_time = base_delay * (2 ** retries)  # Exponential backoff
                print(f"WARNING: {response.status_code} error for {uri}. Retrying ({retries+1}/{max_retries}) after {wait_time} seconds...")
                # Sleep for the retry delay minus the 1-second already slept
                time.sleep(max(0, wait_time - 1))
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
            # Add 1-second delay after failed request attempt
            time.sleep(1)
            # Sleep for the retry delay minus the 1-second already slept
            time.sleep(max(0, base_delay * (2 ** retries) - 1))
            retries += 1
    print(f"ERROR: Max retries ({max_retries}) reached for {uri}. The server may be down or overloaded. Please try again later.")
    exit(1)


# Step 3: Prompt user for Chartmetric Track ID
try:
    track_cmid = input("Enter the Chartmetric Track ID (CMID): ").strip()
    track_cmid = int(track_cmid)  # Ensure it's an integer
except ValueError:
    print("ERROR: Invalid CMID. Please enter a valid integer.")
    exit(1)

# Step 4: Fetch track stats with specified parameters
platform = "spotify"  # Fixed platform as per request
mode = "highest-playcounts"  # Mode for trend data
since = "2025-01-01"  # Start date as per request
until = "2025-05-28"  # Today's date (07:05 PM HKT, May 28, 2025)
type_param = "streams"  # Type of stat as per request
uri = f'/api/track/{track_cmid}/{platform}/stats/{mode}?since={since}&until={until}&type={type_param}'
print(f"Fetching trend stats for track ID {track_cmid} (platform={platform}, mode={mode}, since={since}, until={until}, type={type_param})")
res = Get(uri)
if res.status_code == 404:
    print(f"ERROR: Received a 404 Not Found for track ID {track_cmid}. The CMID might be invalid or no trend stats are available for {platform}.")
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
    print(f"ERROR: No trend stats data returned for track ID {track_cmid} on {platform}.")
    exit(1)

# Step 6: Save the track stats to a JSON file
try:
    with open(f'track_trend_stats_{track_cmid}_{platform}.json', 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=4)
    print(f"Successfully saved trend stats to track_trend_stats_{track_cmid}_{platform}.json")
except Exception as e:
    print(f"ERROR: Failed to save track stats: {str(e)}")
    exit(1)

# Step 7: Display the track stats
print("Track Trend Stats:")
print(json.dumps(stats_data, indent=4))
