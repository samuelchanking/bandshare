1. Trace the number of streams and followers across all major platforms for an artist. Does it have steady behaviour?
 If yes,  we just produce an output,  if No, we try to dig deeper.
 
2.0) Which tracks or album generated the spike?

2.01) Playlists for spiked track. Are there any fresh playlists?

2.1) Check the growth of streams vs the growth of followers. 

2.2) Check geography of streams for suspected tracks

2.3) ...

What API calls to use:


1. __Streams, followers__

   GET /api/v2/artist/{uuid}/current/stats
(gives rating and it's absolute and % change across all platforms, including social ones)



__2.01) Playlists__  
def get_playlists(platform, offset=0, limit=100, body=None, print_progress=False):
        """
        You can sort playlists in our database using specific parameters such as the number of followers, 28-day adds, track count, or last updated date. Apply filters based on attributes like genre, type, country, owner, track age, percentage of adds over the last 28 days, or performance metrics.
        Please note that you can only retrieve the playlists for one platform at a time.

        :param platform: A playlist Chart platform code. Default: spotify.
        :param offset: Pagination offset. Default: 0.
        :param limit: Number of results to retrieve. None: no limit (warning: can take up to 100,000 calls - you may want to use parallel processing). Default: 100.
        :param body: JSON Payload. If none, the default sorting will apply (by metric for the platforms who have one, by 28DayAdds for others) and there will be no filters.
        :param print_progress: Prints an estimated progress percentage (default: False).
        :return: JSON response or an empty dictionary.
        """
        
headers = {
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

