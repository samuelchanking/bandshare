from soundcharts.client import SoundchartsClient
import json

sc = SoundchartsClient(app_id="MANCHESTER_696DCD6E", api_key="11ed17a6cf25afa4")

# Example with Billie Eilish's UUID
billie_audience = sc.artist.get_audience("11e83fed-c5cd-e380-ba73-a0369fe50396")

# Export to JSON file
with open('billie_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(billie_audience, f, ensure_ascii=False, indent=4)
