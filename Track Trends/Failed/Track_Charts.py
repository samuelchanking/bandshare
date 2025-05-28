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

# Step 3: Set parameters
try:
    track_cmid = 29894487  # Fixed from user input
    start_date = "2020-03-01"  # Adjusted to historical range
    end_date = "2025-05-20"    # Adjusted to historical range
    chart_type = "spotify_top_daily"  # Chart type for Spotify daily charts
except ValueError:
    print("ERROR: Invalid CMID. Please ensure CMID is a valid integer.")
    exit(1)

# Step 4: Fetch Spotify chart data
try:
    uri = f"/api/track/{track_cmid}/{chart_type}/charts?since={start_date}&until={end_date}"
    response = Get(uri)
    
    # Log raw response for debugging
    print(f"Raw API response: {response.text}")
    
    # Check if response is JSON-parsable
    try:
        response_json = response.json()
        print(f"Parsed JSON response: {json.dumps(response_json, indent=2)}")
        
        # Adjust parsing based on actual structure
        if "obj" in response_json:
            chart_data = response_json["obj"].get("data", [])
        else:
            chart_data = response_json.get("data", [])
        
        # Filter for global data if region field exists
        #chart_data = [entry for entry in chart_data if entry.get("code2", "").lower() == "global"]
        
    except ValueError:
        print(f"ERROR: Invalid JSON response from API: {response.text}")
        exit(1)
    
    if not chart_data:
        print(f"No Spotify chart data found for Track ID {track_cmid} between {start_date} and {end_date}.")
    else:
        # Count entries per date for debugging
        date_counts = {}
        for entry in chart_data:
            date = entry.get("added_at", entry.get("date", "N/A"))  # Try both timestamp and date
            date_counts[date] = date_counts.get(date, 0) + 1
        
        print(f"\nNumber of entries per date:")
        for date, count in date_counts.items():
            print(f"Date: {date}, Number of entries: {count}")
        
        print(f"\nSpotify Chart Data for Track ID {track_cmid}:")
        for entry in chart_data:
            # Try both timestamp and date fields
            date = entry.get("added_at", entry.get("date", "N/A"))
            position = entry.get("code2s", "N/A")
            streams = entry.get("chart_type", "N/A")
            region = entry.get("code2", entry.get("country", "N/A"))  # Try both region and country
            print(f"Date: {date}, Region: {region}, Chart Position: {position}, Streams: {streams}")
        
        # Save data to JSON file
        with open(f"spotify_chart_data_{track_cmid}.json", "w") as f:
            json.dump(chart_data, f, indent=4)
        print(f"\nData saved to 'spotify_chart_data_{track_cmid}.json'.")
except Exception as e:
    print(f"ERROR: Failed to fetch Spotify chart data: {str(e)}")
    exit(1)
