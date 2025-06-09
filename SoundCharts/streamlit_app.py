import streamlit as st
import pandas as pd
import plotly.express as px
from soundcharts_api import SoundchartsAPI

# --- Page Configuration ---
st.set_page_config(
    page_title="Soundcharts Dashboard",
    layout="wide"
)

# --- Database Connection & Caching ---

@st.cache_resource
def get_api_client():
    # Connect to the database. Using the name from your previous debug logs.
    return SoundchartsAPI(db_name='soundcharts7')

api = get_api_client()

# --- Caching Functions (Updated for robustness) ---

@st.cache_data
def get_all_artists():
    """Fetches all artists, looking for the name inside the nested 'object' field."""
    artist_records = list(api.collections['artists'].find(
        {'object.name': {'$exists': True}, 'artist_uuid': {'$exists': True}},
        {'object.name': 1, 'artist_uuid': 1, '_id': 0}
    ))
    return artist_records

@st.cache_data
def get_artist_details(artist_uuid):
    """
    Fetches artist details and flattens the metadata.
    """
    if not artist_uuid: return None
    
    full_document = api.collections['artists'].find_one({'artist_uuid': artist_uuid})
    
    artist_metadata = full_document.get('object', {}) if full_document else {}

    details = {
        "metadata": artist_metadata,
        "audience": api.collections['audience'].find_one({'artist_uuid': artist_uuid}),
        "popularity": api.collections['popularity'].find_one({'artist_uuid': artist_uuid}),
        "albums": list(api.collections['albums'].find({'artist_uuid': artist_uuid}))
    }
    return details

@st.cache_data
def get_album_details(album_uuid):
    if not album_uuid: return None
    return {
        "tracklist": api.collections['tracklist'].find_one({'album_uuid': album_uuid}),
        "streams": api.collections['streams'].find_one({'album_uuid': album_uuid})
    }

@st.cache_data
def get_song_details(song_uuid):
    if not song_uuid: return None
    return {
        "audience": api.collections['songs_audience'].find_one({'song_uuid': song_uuid})
    }

# --- Helper Functions for Plotting and Display ---

def plot_timeseries(data, title):
    """Generic function to plot time-series data from collections."""
    if not data or not isinstance(data.get('data'), list) or not data['data']:
        st.warning(f"No time-series data found for '{title}'.")
        return

    processed_data = []
    
    # Use the title to determine which data structure to parse
    is_artist_audience = "Artist Spotify Audience" in title
    is_artist_popularity = "Artist Popularity" in title

    for entry in data['data']:
        date = entry.get('date')
        value = None

        if is_artist_audience:
            # Case 1: Direct 'followerCount' key (for artist audience)
            value = entry.get('followerCount')
        elif is_artist_popularity:
            # Case 2: Direct 'value' key (for artist popularity)
            value = entry.get('value')
        else:
            # Case 3: Nested 'value' inside 'plots' array (for all other metrics)
            if 'plots' in entry and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
                value = entry['plots'][0].get('value')

        if date and value is not None:
            processed_data.append({'date': date, 'value': value})

    if not processed_data:
        st.warning(f"Could not find valid plot data for '{title}'.")
        return

    df = pd.DataFrame(processed_data)
    
    if 'date' not in df.columns or 'value' not in df.columns:
        st.error(f"Processed data for '{title}' is missing 'date' or 'value' fields.")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    fig = px.line(df, x='date', y='value', title=title, labels={'value': 'Value', 'date': 'Date'})
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def display_artist_metadata(metadata):
    """Creates a well-designed display for the artist's metadata."""
    if not metadata:
        st.warning("No artist metadata available.")
        return

    image_url = metadata.get("imageUrl")
    if image_url:
        st.image(image_url, width=150, caption=metadata.get("name"))

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
    """Creates a well-designed display for the album's metadata."""
    if not album_data:
        st.warning("No album metadata available.")
        return
    
    metadata = album_data.get('object', album_data)
    
    st.subheader(album_name)

    track_count = metadata.get('totalTracks', 0)
    
    distributor_info = metadata.get('distributor')
    if isinstance(distributor_info, dict):
        distributor_name = distributor_info.get('name', 'N/A')
    elif isinstance(distributor_info, str):
        distributor_name = distributor_info
    else:
        distributor_name = 'N/A'

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Number of Tracks", value=track_count)
    with col2:
        st.metric(label="Distributor", value=distributor_name)

    st.write(f"**Release Date:** {metadata.get('releaseDate', 'N/A')}")
    st.write(f"**UPC:** `{metadata.get('upc', 'N/A')}`")
    
    app_url = metadata.get("appUrl")
    if app_url:
        st.markdown(f"[View Album on Soundcharts]({app_url})", unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Raw Tracklist Items (from DB):**")
    
    if tracklist_data:
        data_object = tracklist_data.get('object', tracklist_data)
        items_list = data_object.get('items', [])
        
        if items_list and isinstance(items_list, list):
            display_data = [
                {
                    "Track #": item.get("number", "-"),
                    "Name": item.get("song", {}).get("name", "Unknown"),
                    "UUID": item.get("song", {}).get("uuid", "N/A")
                }
                for item in items_list
            ]
            st.dataframe(display_data, use_container_width=True)
        else:
             st.info("No 'items' field found in the tracklist record.")

    else:
        st.info("No tracklist document found.")

def display_track_info(track_data, album_uuid):
    """Creates a well-designed display for a single track."""
    track_number = track_data.get("number", "-")
    
    song_info = track_data.get("song", {})
    track_name = song_info.get("name", "Unknown Track")
    song_uuid = song_info.get("uuid")
    
    if not song_uuid:
        return

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 4, 2])
    
    with col1:
        st.metric(label="#", value=track_number)

    with col2:
        st.write(f"**{track_name}**")
        st.caption(f"UUID: {song_uuid}")

    with col3:
        if st.button("Analyze Track", key=f"song_{album_uuid}_{song_uuid}"):
            st.session_state.selected_song_uuid = song_uuid
            st.session_state.selected_album_uuid = None
            st.rerun()

# --- Main App Layout ---
st.title("Soundcharts Interactive Dashboard")

for key in ['selected_artist_uuid', 'selected_album_uuid', 'selected_song_uuid']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- SECTION 1: DATA FETCHER ---
with st.expander("Fetch Data for a New Artist"):
    artist_name_input = st.text_input("Artist Name:", "Dirty Freud")
    start_date_input = st.text_input("Start Date (YYYY-MM-DD):", "2024-01-01")
    end_date_input = st.text_input("End Date (YYYY-MM-DD):", "2025-05-30")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Fetch and Store Data"):
            try:
                with st.spinner(f"Fetching data for {artist_name_input}..."):
                    api.fetch_and_store_artist_data(artist_name_input, start_date_input, end_date_input)
                st.success(f"Data fetched for {artist_name_input}!")
                get_all_artists.clear()
                get_artist_details.clear()
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with col2:
        if st.button("Fetch for Taylor Swift"):
            try:
                with st.spinner("Fetching data for Taylor Swift..."):
                    api.fetch_and_store_artist_data("Taylor Swift", start_date_input, end_date_input)
                st.success("Data fetched for Taylor Swift!")
                get_all_artists.clear()
                get_artist_details.clear()
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")


# --- SECTION 2: ARTIST SELECTION ---
st.header("Select an Artist to Analyze")
artists_data = get_all_artists()
if not artists_data:
    st.warning("No valid artists found. Fetch data for an artist or check database.")
else:
    artist_map = {
        f"{artist['object']['name']} ({artist['artist_uuid']})": artist['artist_uuid']
        for artist in artists_data
    }
    
    current_selection_name = next((name for name, uuid in artist_map.items() if uuid == st.session_state.selected_artist_uuid), None)
    
    selected_artist_name = st.selectbox(
        "Choose an artist from the database:",
        options=list(artist_map.keys()),
        index=list(artist_map.keys()).index(current_selection_name) if current_selection_name else 0,
    )
    
    if selected_artist_name and artist_map[selected_artist_name] != st.session_state.selected_artist_uuid:
        st.session_state.selected_artist_uuid = artist_map[selected_artist_name]
        st.session_state.selected_album_uuid = None
        st.session_state.selected_song_uuid = None
        st.rerun()

# --- SECTION 3: DISPLAY DATA ---
left_col, right_col = st.columns(2)

if st.session_state.selected_artist_uuid:
    artist_details = get_artist_details(st.session_state.selected_artist_uuid)
    
    if artist_details and artist_details.get("metadata"):
        artist_name = artist_details["metadata"].get("name", "Unknown Artist")

        # --- LEFT PANEL (ARTIST PLOTS & METADATA) ---
        with left_col:
            st.header(f"Analytics for {artist_name}")
            
            st.subheader("Artist Details")
            display_artist_metadata(artist_details["metadata"])
            
            st.subheader("Spotify Audience")
            plot_timeseries(artist_details.get('audience'), "Artist Spotify Audience")
            
            with st.expander("Show Raw Audience Data (from DB)"):
                st.json(artist_details.get('audience', "No audience data found."))

            st.subheader("Spotify Popularity")
            plot_timeseries(artist_details.get('popularity'), "Artist Popularity")

        # --- RIGHT PANEL (ALBUMS AND TRACKS) ---
        with right_col:
            st.header(f"Albums by {artist_name}")
            albums = artist_details.get("albums", [])
            if not albums:
                st.info("No albums found for this artist.")
            else:
                for album in albums:
                    album_uuid = album.get("album_uuid")
                    album_name_data = album.get('object', album)
                    album_name = album_name_data.get("name", f"Album ID: {album_uuid}")
                    
                    if not album_uuid: continue

                    with st.expander(f"💿 {album_name}"):
                        album_full_details = get_album_details(album_uuid)
                        tracklist_data = album_full_details.get("tracklist")
                        
                        display_album_metadata(album, tracklist_data, album_name)
                        
                        if st.button("Show Album Streams", key=f"album_{album_uuid}"):
                            st.session_state.selected_album_uuid = album_uuid
                            st.session_state.selected_song_uuid = None
                            st.rerun()
                        
                        st.markdown("---") 
                        
                        tracklist_items = []
                        if tracklist_data:
                            tracklist_items = tracklist_data.get('object', tracklist_data).get('items', [])
                        
                        if tracklist_items:
                            st.write("**Tracks:**")
                            for track in tracklist_items:
                                display_track_info(track, album_uuid)
                        else:
                            total_tracks = album_name_data.get('totalTracks', 0)
                            if total_tracks > 0:
                                st.info("Track names are not available for this album.")
                            else:
                                st.info("This release does not contain any tracks.")


            # --- PLOTTING AREA (DYNAMIC BASED ON SELECTION) ---
            st.divider()
            
            if st.session_state.selected_album_uuid:
                st.header("Album Stream Analytics")
                album_details_for_plot = get_album_details(st.session_state.selected_album_uuid)
                plot_timeseries(album_details_for_plot.get("streams"), "Album Spotify Streams")

            if st.session_state.selected_song_uuid:
                st.header("Song Analytics")
                song_details = get_song_details(st.session_state.selected_song_uuid)
                plot_timeseries(song_details.get("audience"), "Song Spotify Audience")
    else:
        left_col.info("Select an artist from the dropdown to see their analytics.")
