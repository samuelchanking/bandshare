# streamlit_ui.py

import streamlit as st
import pandas as pd
from streamlit_caching import get_song_details
from datetime import datetime, date
import random

def display_artist_metadata(metadata):
    """Displays artist's metadata in a structured layout."""
    if not metadata:
        st.warning("No artist metadata available.")
        return
    metadata_obj = metadata.get('object', metadata)
    
    st.subheader(f"Data for: {metadata_obj.get('name', 'Unknown Artist')}")
    if image_url := metadata_obj.get("imageUrl"):
        st.image(image_url, width=150)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Country", metadata_obj.get("countryCode", "N/A"))
    col2.metric("Type", metadata_obj.get("type", "N/A").capitalize())
    col3.metric("Career Stage", metadata_obj.get("careerStage", "N/A").replace("_", " ").title())

def display_playlists(playlist_items):
    """Displays a list of playlists in a user-friendly format."""
    st.subheader("Featured On Playlists (Spotify)")
    if not playlist_items:
        st.info("No playlist data available for this artist.")
        return

    display_data = []
    for item in playlist_items:
        playlist = item.get('playlist', {})
        song = item.get('song', {})
        display_data.append({
            "Song Name": song.get('name', 'N/A'),
            "Playlist Name": playlist.get('name', 'N/A'),
            "Type": playlist.get('type', 'N/A').capitalize(),
            "Tracks": playlist.get('latestTrackCount', 0),
            "Position": item.get('position'),
            "Peak": item.get('peakPosition'),
            "Subscribers": playlist.get('latestSubscriberCount', 0)
        })
    
    st.dataframe(
        display_data,
        column_config={ "Subscribers": st.column_config.NumberColumn(format="%d") },
        use_container_width=True, hide_index=True
    )

def display_album_and_tracks(db_manager, album_data, tracklist_data):
    """
    Displays album and tracklist data from the database, reading from the corrected structure.
    """
    if not album_data:
        st.warning("No album data available."); return

    # FIX: Correctly reads the album details from the nested 'album_metadata' field,
    # which matches how the database_manager stores the document.
    unified_meta = album_data.get('album_metadata', {})
    album_uuid = unified_meta.get('album_uuid', unified_meta.get('uuid'))
    
    # Display Album Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Release Date", str(unified_meta.get('releaseDate', 'N/A'))[:10])
    col2.metric("Tracks", unified_meta.get('totalTracks', 'N/A'))
    col3.metric("Distributor", unified_meta.get('distributor', 'N/A'))
    st.write(f"**UPC:** `{unified_meta.get('upc', 'N/A')}`")
    st.markdown("---")

    # Display Interactive Tracklist
    st.write("**Tracklist:**")
    if not tracklist_data:
        st.info("Tracklist not available."); return

    items_list = tracklist_data.get('object', tracklist_data).get('items', [])
    if not items_list:
        st.info("Tracklist is empty."); return

    for i, item in enumerate(items_list):
        song = item.get('song', {})
        song_uuid = song.get('uuid')
        song_name = song.get('name', 'Unknown Track')
        
        track_col, button_col = st.columns([4, 1])
        track_col.write(f"**{item.get('number', '#')}.** {song_name}")

        if song_uuid:
            button_key = f"btn_{album_uuid or 'unknown'}_{song_uuid}_{i}"
            
            if button_col.button("Details", key=button_key, use_container_width=True):
                 session_key = f"show_{album_uuid}_{song_uuid}"
                 st.session_state[session_key] = not st.session_state.get(session_key, False)

            if st.session_state.get(f"show_{album_uuid}_{song_uuid}", False):
                song_metadata = get_song_details(db_manager, song_uuid)
                if song_metadata:
                    with st.container(border=True):
                        # RESTORED: Full song metadata display
                        meta_obj = song_metadata.get('object', song_metadata)
                        st.write(f"##### Details for **{meta_obj.get('name', song_name)}**")
                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("Duration", f"{meta_obj.get('duration', 'N/A')}s")
                        rel_date = meta_obj.get('releaseDate', 'N/A')
                        sc2.metric("Release Date", str(rel_date)[:10])
                        genres = meta_obj.get('genres', [])
                        root_genre = genres[0].get('root', 'N/A') if genres else 'N/A'
                        sc3.metric("Genre", root_genre)
                        st.write(f"**Composers:** {', '.join(meta_obj.get('composers', ['N/A']))}")
                        st.write(f"**Producers:** {', '.join(meta_obj.get('producers', ['N/A']))}")
                else:
                    st.warning(f"No metadata found for {song_name}")

def display_audience_chart(audience_data):
    """Displays processed audience chart data as a table for debugging."""
    st.subheader("Audience on Spotify")
    if not audience_data:
        st.info("No audience data available for the selected period.")
        return
        
    # Process the nested JSON to flatten the data for charting
    chart_data = []
    for entry in audience_data:
        chart_data.append({
            'date': entry.get('date'),
            'followerCount': entry.get('followerCount')
        })

    df = pd.DataFrame(chart_data)
    
    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Audience data is missing valid 'date' information.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Plot the 'followerCount' column
    if 'followerCount' in df.columns:
        st.line_chart(df[['followerCount']])
    else:
        st.warning("Could not find 'followerCount' data to plot.")


def display_popularity_chart(popularity_data):
    """Displays processed popularity chart data as a table for debugging."""
    st.subheader("Popularity on Spotify")
    if not popularity_data:
        st.info("No popularity data available for the selected period.")
        return
        
    # Process the nested JSON to flatten the data for charting
    chart_data = []
    for entry in popularity_data:
        chart_data.append({
            'date': entry.get('date'),
            'value': entry.get('value')
        })
        
    df = pd.DataFrame(chart_data)
    

    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Popularity data is missing the required 'date' information.")
        return
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Plot the 'value' column
    if 'value' in df.columns:
        st.line_chart(df['value'])
    else:
        st.warning("Could not find 'value' data to plot.")
