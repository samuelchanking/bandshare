import streamlit as st
import uuid
from soundcharts_api2 import SoundchartsAPI
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Soundcharts Dashboard",
    layout="wide"
)

# --- Database Connection & Caching ---
@st.cache_resource
def get_api_client():
    return SoundchartsAPI(db_name='soundcharts7')

api = get_api_client()

# --- Caching Functions ---
@st.cache_data
def get_all_artists():
    """Fetches all artists, including database-stored and temporarily fetched artists."""
    # Fetch artists from database
    db_artists = list(api.collections['artists'].find(
        {'$or': [{'object.name': {'$exists': True}}, {'name': {'$exists': True}}], 'artist_uuid': {'$exists': True}},
        {'object.name': 1, 'name': 1, 'artist_uuid': 1, '_id': 0}
    ))

    # Add temporarily fetched artist if available
    all_artists = db_artists.copy()
    if (st.session_state.get('fetched_data') and 
        'artist_uuid' in st.session_state['fetched_data'] and 
        'error' not in st.session_state['fetched_data']):
        fetched_metadata = st.session_state['fetched_data'].get('metadata', {})
        fetched_name = fetched_metadata.get('object', {}).get('name', fetched_metadata.get('name', 'Unknown Artist'))
        temp_artist = {
            'artist_uuid': st.session_state['fetched_data']['artist_uuid'],
            'name': fetched_name,
            'object': {'name': f"{fetched_name} (Fetched)"}
        }
        all_artists.append(temp_artist)

    return all_artists

@st.cache_data
def get_artist_details(artist_uuid):
    """Fetches artist details from the database or session state."""
    if not artist_uuid:
        return None

    # Check if it's a temporarily fetched artist
    if (st.session_state.get('fetched_data') and 
        st.session_state['fetched_data'].get('artist_uuid') == artist_uuid):
        return {
            "metadata": st.session_state['fetched_data'].get('metadata', {}),
            "albums": st.session_state['fetched_data'].get('albums', {}).get('items', [])
        }

    # Fetch from database
    artist_metadata = api.collections['artists'].find_one({'artist_uuid': artist_uuid})
    albums = list(api.collections['albums'].find({'artist_uuid': artist_uuid}))
    return {
        "metadata": artist_metadata,
        "albums": albums
    }

@st.cache_data
def get_album_details(album_uuid):
    """Fetches album tracklist from the database, including song metadata."""
    if not album_uuid:
        return None
    tracklist = api.collections['tracklist'].find_one({'album_uuid': album_uuid})
    return {
        "tracklist": tracklist
    }

@st.cache_data
def get_song_details(song_uuid):
    """Fetches song metadata from session state or database."""
    if not song_uuid:
        return None

    # Check temporarily fetched data
    if (st.session_state.get('fetched_data') and 
        'song_metadata' in st.session_state['fetched_data']):
        album_uuid = st.session_state['fetched_data'].get('current_album_uuid', '')
        if album_uuid and song_uuid in st.session_state['fetched_data']['song_metadata'].get(album_uuid, {}):
            return st.session_state['fetched_data']['song_metadata'][album_uuid][song_uuid]

    # Fetch from database
    if 'songs' in api.collections:
        song = api.collections['songs'].find_one({'song_uuid': song_uuid})  # Use song_uuid
        if not song:
            song = api.collections['songs'].find_one({'uuid': song_uuid})  # Fallback to uuid
        if song:
            return song
    st.warning(f"No metadata found for song_uuid: {song_uuid}")
    return {}

# --- Helper Functions for Display ---
def display_artist_metadata(metadata):
    """Displays artist's metadata."""
    if not metadata:
        st.warning("No artist metadata available.")
        return

    # Handle both 'object' nested structure and direct metadata
    metadata = metadata.get('object', metadata)

    image_url = metadata.get("imageUrl")
    if image_url:
        st.image(image_url, width=150, caption=metadata.get("name", "Unknown Artist"))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Country", value=metadata.get("countryCode", "N/A"))
    with col2:
        st.metric(label="Type", value=metadata.get("type", "N/A").capitalize())
    with col3:
        st.metric(label="Career Stage", value=metadata.get("careerStage", "N/A").replace("_", " ").title())

    genres = metadata.get("genres", [])
    if genres:
        sub_genres = [g.get('sub', []) for g in genres]
        flat_genres = [item for sublist in sub_genres for item in sublist]
        st.write("**Genres:**")
        st.write(", ".join(g.capitalize() for g in flat_genres))

    biography = metadata.get("biography")
    if biography:
        with st.expander("Show Biography"):
            st.markdown(biography)

    app_url = metadata.get("appUrl")
    if app_url:
        st.markdown(f"[View on Soundcharts]({app_url})", unsafe_allow_html=True)

def display_album_metadata(album_data, tracklist_data, album_name):
    """Displays album metadata and tracklist with button to toggle song metadata."""
    if not album_data:
        st.warning("No album metadata available.")
        return

    # Handle nested 'object' structure and merge with album_metadata
    album_uuid = album_data.get('uuid', album_data.get('album_uuid', ''))
    metadata = album_data.get('object', {})
    if (st.session_state.fetched_data and 
        'album_metadata' in st.session_state.fetched_data and 
        album_uuid in st.session_state.fetched_data['album_metadata']):
        metadata.update(st.session_state.fetched_data['album_metadata'][album_uuid].get('object', {}))

    st.subheader(album_name)

    # Get number of tracks from totalTracks
    track_count = metadata.get('totalTracks', 0)

    # Get distributor from distributor field
    distributor = metadata.get('distributor', 'N/A')

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Number of Tracks", value=track_count)
    with col2:
        st.metric(label="Distributor", value=distributor)

    # Parse release date to show only the date part
    release_date = metadata.get('releaseDate', 'N/A')
    if isinstance(release_date, str) and len(release_date) > 10:
        release_date = release_date[:10]  # Extract YYYY-MM-DD

    st.write(f"**Release Date:** {release_date}")
    st.write(f"**UPC:** `{metadata.get('upc', 'N/A')}`")

    st.markdown("---")
    st.write("**Tracklist:**")

    if tracklist_data:
        data_object = tracklist_data.get('object', tracklist_data)
        items_list = data_object.get('items', [])
        if items_list and isinstance(items_list, list):
            display_data = [
                {
                    "Track #": item.get("number", "-"),
                    "Name": item.get("song", {}).get("name", "Unknown")
                }
                for item in items_list
            ]
            st.dataframe(display_data, use_container_width=True)

            # Use buttons to toggle song metadata visibility
            for item in items_list:
                song_uuid = item.get('song', {}).get('uuid')
                if song_uuid:
                    song_name = item.get('song', {}).get('name', 'Unknown Track')
                    key = f"btn_{album_uuid}_{song_uuid}_{song_name}"  # Unique key with album_uuid
                    if 'show_metadata' not in st.session_state:
                        st.session_state['show_metadata'] = {}
                    if st.button(f"Show Metadata for {song_name}", key=key):
                        st.session_state['show_metadata'][song_uuid] = not st.session_state['show_metadata'].get(song_uuid, False)
                    if st.session_state['show_metadata'].get(song_uuid, False):
                        song_metadata = get_song_details(song_uuid)
                        if song_metadata:
                            st.write(f"### {song_name} Metadata")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{song_metadata.get('object', {}).get('duration', 'N/A')}s")
                            with col2:
                                release_date = song_metadata.get('releaseDate', 'N/A')
                                if isinstance(release_date, str) and len(release_date) > 10:
                                    release_date = release_date[:10]
                                st.metric("Release Date", release_date)
                            with col3:
                                genres = song_metadata.get('object', {}).get('genres', [{}])
                                root_genre = genres[0].get('root', 'N/A') if genres else 'N/A'
                                st.metric("Genre", root_genre)
                                sub_genres = genres[0].get('sub', []) if genres and isinstance(genres[0], dict) else []
                                if sub_genres:
                                    st.write("**Sub Genres:**")
                                    for sub in sub_genres:
                                        st.write(f"- {sub}")
                            st.write(f"**Composers:** {', '.join(song_metadata.get('composers', ['N/A']))}")
                            st.write(f"**Producers:** {', '.join(song_metadata.get('producers', ['N/A']))}")
                            st.markdown("---")
                        else:
                            st.write(f"No metadata available for {song_name}")
        else:
            st.info("No 'items' field found in the tracklist record.")
    else:
        st.info("No tracklist document found.")

# --- Main App Layout ---
st.title("Soundcharts Interactive Dashboard")

# Initialize session state
for key in ['selected_artist_uuid', 'fetched_data']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- SECTION 1: DATA FETCHER ---
with st.expander("Fetch Data for a New Artist"):
    artist_name_input = st.text_input("Artist Name:", placeholder="Enter artist name (e.g., Dirty Freud)")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Fetch Data"):
            try:
                with st.spinner(f"Fetching data for {artist_name_input}..."):
                    st.session_state.fetched_data = api.fetch_artist_data(artist_name_input)
                if 'error' in st.session_state.fetched_data:
                    st.error(st.session_state.fetched_data['error'])
                else:
                    st.success(f"Data fetched for {artist_name_input}!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with col2:
        if st.button("Store Fetched Data"):
            if not st.session_state.fetched_data:
                st.error("No fetched data to store. Fetch data first.")
            else:
                try:
                    with st.spinner("Storing data..."):
                        result = api.store_artist_data(st.session_state.fetched_data)
                    if result['status'] == 'success':
                        st.success("Data stored successfully!")
                        get_all_artists.clear()
                        get_artist_details.clear()
                        get_album_details.clear()
                        get_song_details.clear()  # Clear song details cache after storage
                        st.session_state.fetched_data = None  # Clear fetched data after storage
                        st.rerun()
                    else:
                        st.error(result['message'])
                except Exception as e:
                    st.error(f"Failed to store data: {e}")

# --- SECTION 2: ARTIST SELECTION ---
st.header("Select an Artist to Analyze")
artists_data = get_all_artists()
if not artists_data:
    st.warning("No valid artists found. Fetch data for an artist or check database.")
else:
    artist_map = {
        f"{artist.get('object', {}).get('name', artist.get('name', 'Unknown Artist'))} ({artist['artist_uuid']})": artist['artist_uuid']
        for artist in artists_data
    }
    # Use placeholder to avoid pre-selecting an artist
    selected_artist_name = st.selectbox(
        "Choose an artist from the database:",
        options=["Select an artist"] + list(artist_map.keys()),
        index=0,  # Default to "Select an artist"
        key="artist_selectbox"
    )
    if selected_artist_name != "Select an artist":
        st.session_state.selected_artist_uuid = artist_map[selected_artist_name]
    else:
        st.session_state.selected_artist_uuid = None

# --- SECTION 3: DISPLAY DATA ---
left_col, right_col = st.columns(2)

if st.session_state.fetched_data and 'error' not in st.session_state.fetched_data:
    with left_col:
        st.header("Fetched Artist Data (Not Stored)")
        display_artist_metadata(st.session_state.fetched_data.get('metadata', {}))

    with right_col:
        st.header("Fetched Albums")
        albums = st.session_state.fetched_data.get('albums', {}).get('items', [])
        if not albums:
            st.info("No albums found for this artist.")
        else:
            for album in albums:
                album_uuid = album.get("uuid")
                album_name = album.get("name", f"Album ID: {album_uuid}")

                with st.expander(f"💿 {album_name}"):
                    tracklist_data = st.session_state.fetched_data.get("tracklists", {}).get(album_uuid, {})
                    display_album_metadata(album, tracklist_data, album_name)

if st.session_state.selected_artist_uuid:
    artist_details = get_artist_details(st.session_state.selected_artist_uuid)
    if artist_details and artist_details.get("metadata"):
        artist_name = artist_details["metadata"].get("object", artist_details["metadata"]).get("name", "Unknown Artist")
        with left_col:
            st.header(f"Analytics for {artist_name} (From Database)")
            display_artist_metadata(artist_details["metadata"])
        with right_col:
            st.header(f"Albums by {artist_name} (From Database)")
            albums = artist_details.get("albums", [])
            if not albums:
                st.info("No albums found for this artist.")
            else:
                for album in albums:
                    album_uuid = album.get("album_uuid")
                    album_name = album.get("object", album).get("name", f"Album ID: {album_uuid}")
                    with st.expander(f"💿 {album_name}"):
                        album_full_details = get_album_details(album_uuid)
                        tracklist_data = album_full_details.get("tracklist")
                        display_album_metadata(album, tracklist_data, album_name)
else:
    left_col.info("Select an artist from the dropdown to see their analytics.")
