Trace the number of streams and followers across all major platforms for an artist. Does it have steady behaviour?
 If yes,  we just produce an output,  if No, we try to dig deeper.
2.0) Which tracks or album generated the spike?
2.01) Playlists for spiked track. Are there any fresh playlists?
2.1) Check the growth of streams vs the growth of followers. 
2.2) Check geography of streams for suspected tracks
2.3) ...

What API calls to use:


-- 2.01)  headers = {
    'x-app-id': 'soundcharts',
    'x-api-key': 'soundcharts',
}

params = {
    'type': 'all',
    'offset': '0',
    'limit': '100',
    'sortBy': 'entryDate',
    'sortOrder': 'desc',
}

response = requests.get(
    'https://customer.api.soundcharts.com/api/v2.20/song/7d534228-5165-11e9-9375-549f35161576/playlist/current/spotify',
    params=params,
    headers=headers,
)

