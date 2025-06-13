# streamlit_ui.py

import streamlit as st
from streamlit_caching import get_song_details
from datetime import datetime, date

def display_artist_metadata(metadata):
    """Displays artist's metadata in a structured layout."""
    if not metadata:
        st.warning("No artist metadata available.")
        return
    metadata_obj = metadata.get('object', metadata)
    
    if image_url := metadata_obj.get("imageUrl"):
        st.image(image_url, width=150, caption=metadata_obj.get("name", "Unknown Artist"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Country", metadata_obj.get("countryCode", "N/A"))
    col2.metric("Type", metadata_obj.get("type", "N/A").capitalize())
    col3.metric("Career Stage", metadata_obj.get("careerStage", "N/A").replace("_", " ").title())

    if genres := metadata_obj.get("genres", []):
        sub_genres = [g.get('sub', []) for g in genres]
        flat_genres = [item for sublist in sub_genres for item in sublist]
        st.write("**Genres:** " + ", ".join(g.capitalize() for g in flat_genres))

    if biography := metadata_obj.get("biography"):
        with st.expander("Show Biography"):
            st.markdown(biography)

def display_playlists(playlist_items):
    """Displays a list of playlists in a user-friendly format."""
    st.subheader("Featured On Playlists (Spotify)")
    
    if not playlist_items:
        st.info("No playlist entries found for this artist.")
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
        column_config={
            "Subscribers": st.column_config.NumberColumn(format="%d"),
            "Tracks": st.column_config.NumberColumn(),
            "Position": st.column_config.NumberColumn(),
            "Peak": st.column_config.NumberColumn(),
        },
        use_container_width=True,
        hide_index=True
    )

def display_album_and_tracks(db_manager, album_data, tracklist_data):
    """
    Displays album and tracklist data from the database.
    This version correctly handles the stored data structure.
    """
    if not album_data:
        st.warning("No album data available."); return

    # FIX: Logic is now simplified and robust. It looks for metadata in the nested 
    # 'object' field but falls back to the main document. This correctly handles
    # the structure of your stored album documents.
    unified_meta = album_data.get('object', album_data)
    album_uuid = unified_meta.get('album_uuid', unified_meta.get('uuid'))
    
    # --- Display Album Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Release Date", str(unified_meta.get('releaseDate', 'N/A'))[:10])
    col2.metric("Tracks", unified_meta.get('totalTracks', 'N/A'))
    col3.metric("Distributor", unified_meta.get('distributor', 'N/A'))
    st.write(f"**UPC:** `{unified_meta.get('upc', 'N/A')}`")
    st.markdown("---")

    # --- Display Interactive Tracklist ---
    st.write("**Tracklist:**")
    if not tracklist_data:
        st.info("Tracklist not available."); return

    items_list = tracklist_data.get('object', tracklist_data).get('items', [])
    if not items_list:
        st.info("Tracklist is empty."); return

    for item in items_list:
        song = item.get('song', {})
        song_uuid = song.get('uuid')
        song_name = song.get('name', 'Unknown Track')
        
        track_col, button_col = st.columns([4, 1])
        track_col.write(f"**{item.get('number', '#')}.** {song_name}")

        if song_uuid:
            if button_col.button("Details", key=f"btn_{album_uuid}_{song_uuid}", use_container_width=True):
                 st.session_state[f"show_{album_uuid}_{song_uuid}"] = not st.session_state.get(f"show_{album_uuid}_{song_uuid}", False)

            if st.session_state.get(f"show_{album_uuid}_{song_uuid}", False):
                song_metadata = get_song_details(db_manager, song_uuid)
                if song_metadata:
                    with st.container(border=True):
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
