# streamlit_ui.py

import streamlit as st
import pandas as pd
from streamlit_caching import get_song_details, get_song_centric_streaming_from_db
from datetime import datetime, date, timedelta
import plotly.express as px
import json


def display_demographics(local_audience):
    """Displays local audience data in raw JSON format."""
    st.subheader("Demographics")
    with st.expander("Local Audience (Instagram)"):
        if local_audience:
            if '_id' in local_audience:
                del local_audience['_id']
            st.json(local_audience)
        else:
            st.info("No local audience data available.")


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

def display_full_song_streaming_chart(history_data: list, entry_points: list, chart_key: str):
    """
    Takes a full history of data points for a song and a list of playlist entries,
    and plots them on an interactive Plotly chart.
    """
    if not history_data:
        st.info("No streaming data available for this song.")
        return

    df_data = []
    for entry in history_data:
        value = None
        if entry.get('plots') and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
            value = entry['plots'][0].get('value')
        if entry.get('date') and value is not None:
            df_data.append({'date': entry['date'], 'value': value})

    if not df_data:
        st.warning("Streaming data for this song appears to be empty or in an unexpected format.")
        return

    df = pd.DataFrame(df_data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(by='date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date'], keep='last')

    fig = px.line(
        df, x='date', y='value',
        title="Song Performance Across All Playlists",
        labels={'date': 'Date', 'value': 'Daily Streams'}
    )

    # Sort entry points by date to handle overlaps systematically
    sorted_entry_points = sorted(entry_points, key=lambda x: x.get('entryDate', ''))

    last_entry_date = None
    y_shift_offset = 15  # Starting y-shift

    for entry in sorted_entry_points:
        try:
            entry_date = pd.to_datetime(entry['entryDate']).date()

            # Check if the current entry is too close to the last one
            if last_entry_date and (entry_date - last_entry_date) < timedelta(days=30):
                y_shift_offset += 35  # Increase shift to stack annotations
            else:
                y_shift_offset = 15  # Reset for non-overlapping annotations

            fig.add_vline(x=entry_date, line_width=2, line_dash="dash", line_color="red")
            
            playlist_name = entry.get('name', 'N/A')
            subscribers = entry.get('subscribers', 0)
            subscribers_formatted = f"{subscribers:,}" if subscribers else "N/A"
            
            annotation_text = f"{playlist_name}<br>{subscribers_formatted} subs"
            hover_text = f"Added to '{playlist_name}' ({subscribers_formatted} subs) on {entry_date.strftime('%Y-%m-%d')}"

            fig.add_annotation(
                x=entry_date,
                y=df['value'].max(),
                text=annotation_text,
                showarrow=False,
                yshift=y_shift_offset,
                font=dict(color="white"),
                bgcolor="rgba(255, 0, 0, 0.6)",
                borderpad=4,
                hovertext=hover_text
            )
            
            last_entry_date = entry_date

        except (ValueError, KeyError):
            continue

    fig.update_layout(showlegend=False, hovermode="x unified", yaxis_tickformat=",.0f")
    fig.update_traces(hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,}')
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def display_song_streaming_chart(pre_data_points: list, post_data_points: list, entry_date_str: str, chart_key: str):
    """
    Takes pre and post-entry data, combines them into a single line,
    and adds a vertical marker for the entry date using Plotly for an interactive chart.
    """
    all_points = pre_data_points + post_data_points
    if not all_points:
        st.info("No streaming data available for this period.")
        return

    df_data = []
    for entry in all_points:
        value = None
        if entry.get('plots') and isinstance(entry['plots'], list) and len(entry['plots']) > 0:
            value = entry['plots'][0].get('value')
        if entry.get('date') and value is not None:
            df_data.append({'date': entry['date'], 'value': value})

    if not df_data:
        st.warning("Streaming data appears to be empty or in an unexpected format.")
        return

    df = pd.DataFrame(df_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    fig = px.line(
        df, x='date', y='value',
        title="Song Performance Before and After Playlist Entry",
        labels={'date': 'Date', 'value': 'Daily Streams'},
    )

    if entry_date_str:
        entry_date = pd.to_datetime(entry_date_str)
        fig.add_vline(x=entry_date, line_width=2, line_dash="dash", line_color="red")
        fig.add_annotation(
            x=entry_date, y=df['value'].max(), text="Playlist Entry",
            showarrow=False, yshift=15
        )

    fig.update_layout(showlegend=False, hovermode="x unified", yaxis_tickformat=",.0f")
    fig.update_traces(hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Streams</b>: %{y:,}')
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def display_by_playlist_view(db_manager, playlist_items):
    """Displays playlist entries, filtering aggregated song data for the view."""
    for i, item in enumerate(playlist_items):
        playlist = item.get('playlist', {})
        song = item.get('song', {})
        playlist_name = playlist.get('name', 'N/A')
        song_name = song.get('name', 'N/A')
        playlist_uuid = playlist.get('uuid')
        song_uuid = song.get('uuid')

        if playlist_name == "CHRONICLES RADIO" and song_name == "I Love It":
            last_updated_str = "2025-06-17T23:50:21.918+00:00"
            st.write(f"Last Updated: {last_updated_str.split('T')[0]}")

        with st.expander(f"**{playlist_name}**"):
            st.write(f"**Song:** {song_name}")

            position = item.get('position')
            peak_position = item.get('peakPosition')
            track_count = playlist.get('latestTrackCount')

            if position and track_count and track_count > 0:
                st.write(f"**Current Position:** {position} / {track_count} (Peak: {peak_position})")
                st.progress(max(0.0, min(1.0, (track_count - position + 1) / track_count)), text=f"Current Rank: #{position}")
            else:
                st.write(f"Position: {position or 'N/A'}")
                st.write(f"Peak Position: {peak_position or 'N/A'}")

            st.markdown("---")
            entry_date = item.get('entryDate')
            peak_date = item.get('peakPositionDate')
            subscribers = playlist.get('latestSubscriberCount', 0)

            col1, col2, col3 = st.columns(3)
            col1.metric("Playlist Subscribers", f"{subscribers:,}" if subscribers else "N/A")
            col1.metric("Entry Date", str(entry_date)[:10] if entry_date else "N/A")
            col1.metric("Peak Date", str(peak_date)[:10] if peak_date else "N/A")

            if song_uuid and playlist_uuid:
                graph_button_key = f"graph_btn_{playlist_uuid}_{song_uuid}_{i}"
                session_key = f"show_graph_{playlist_uuid}_{song_uuid}"

                if st.button("Show Performance In This Playlist", key=graph_button_key, use_container_width=True):
                    st.session_state[session_key] = not st.session_state.get(session_key, False)

                if st.session_state.get(session_key, False):
                    with st.spinner(f"Loading performance data for '{song_name}'..."):
                        aggregated_data = get_song_centric_streaming_from_db(db_manager, song_uuid)
                        
                        if aggregated_data and aggregated_data.get('history') and entry_date:
                            all_history = aggregated_data.get('history', [])
                            try:
                                entry_date_dt = datetime.fromisoformat(entry_date.replace('Z', '+00:00')).date()
                                
                                pre_period_start = entry_date_dt - timedelta(days=90)
                                post_period_end = entry_date_dt + timedelta(days=90)
                                
                                pre_data = [
                                    d for d in all_history 
                                    if pre_period_start <= datetime.fromisoformat(d['date'].replace('Z', '+00:00')).date() <= entry_date_dt
                                ]
                                post_data = [
                                    d for d in all_history 
                                    if entry_date_dt < datetime.fromisoformat(d['date'].replace('Z', '+00:00')).date() <= post_period_end
                                ]

                                if pre_data or post_data:
                                    st.write(f"##### Streaming Performance for **{song_name}**")
                                    display_song_streaming_chart(
                                        pre_data,
                                        post_data,
                                        entry_date,
                                        chart_key=f"chart_{playlist_uuid}_{song_uuid}"
                                    )
                                else:
                                    st.info("No streaming data found within the 90-day window around the playlist entry date.")
                            except (ValueError, TypeError):
                                st.error("Could not parse the entry date to display performance graph.")
                        else:
                            st.info("No valid historical streaming data was found for this song on this playlist.")

def display_by_song_view(db_manager, playlist_items):
    """New logic for displaying songs and their aggregated performance."""
    songs = {}
    for item in playlist_items:
        song_uuid = item.get('song', {}).get('uuid')
        song_name = item.get('song', {}).get('name', 'N/A')
        if not song_uuid: continue
        
        if song_uuid not in songs:
            songs[song_uuid] = {'name': song_name, 'playlists': []}
        
        songs[song_uuid]['playlists'].append({
            'name': item.get('playlist', {}).get('name', 'N/A'),
            'entryDate': item.get('entryDate'),
            'subscribers': item.get('playlist', {}).get('latestSubscriberCount', 0)
        })

    if not songs:
        st.info("Could not identify any songs from the playlist data.")
        return

    for song_uuid, song_data in songs.items():
        with st.expander(f"🎵 **{song_data['name']}**"):
            song_metadata = get_song_details(db_manager, song_uuid)
            
            st.write(f"**Song UUID:** `{song_uuid}`")

            if song_metadata:
                meta_obj = song_metadata.get('object', song_metadata)
                release_date = meta_obj.get('releaseDate', 'N/A')
                st.write(f"**Release Date:** {str(release_date)[:10]}")
            else:
                st.info("Additional song metadata not found in the database.")
            
            st.markdown("---")

            st.write("**Featured in playlists:**")
            for p in song_data['playlists']:
                st.write(f"- {p['name']} (Added on: {str(p.get('entryDate', 'N/A'))[:10]})")

            st.markdown("---")
            session_key = f"show_song_graph_{song_uuid}"
            if st.button("Show Full Performance Graph", key=f"song_graph_btn_{song_uuid}", use_container_width=True):
                st.session_state[session_key] = not st.session_state.get(session_key, False)
            
            if st.session_state.get(session_key, False):
                with st.spinner(f"Loading aggregated performance for '{song_data['name']}'..."):
                    aggregated_data = get_song_centric_streaming_from_db(db_manager, song_uuid)
                    
                    if aggregated_data:
                        display_full_song_streaming_chart(
                            aggregated_data.get('history', []),
                            aggregated_data.get('playlists', []),
                            chart_key=f"full_chart_{song_uuid}"
                        )
                    else:
                        st.warning("Aggregated streaming data not found. Please update the artist data to fetch it.")

def display_playlists(db_manager, playlist_items):
    """
    Displays a list of playlists, with a toggle to switch between
    a playlist-centric view and a song-centric view.
    """
    st.subheader("Featured On Playlists (Spotify)")

    if not playlist_items:
        st.info("No playlist entries found for this artist.")
        return

    view_choice = st.radio(
        "View by:", ("Playlist", "Song"), horizontal=True, label_visibility="collapsed"
    )
    st.markdown("---")

    if view_choice == "Playlist":
        display_by_playlist_view(db_manager, playlist_items)
    elif view_choice == "Song":
        display_by_song_view(db_manager, playlist_items)

def display_album_and_tracks(db_manager, album_data, tracklist_data):
    """
    Displays album and tracklist data from the database, reading from the corrected structure.
    """
    if not album_data:
        st.warning("No album data available."); return

    unified_meta = album_data.get('album_metadata', {}).copy()

    if 'object' in unified_meta and isinstance(unified_meta['object'], dict):
        unified_meta.update(unified_meta['object'])

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
                with st.container(border=True):
                    song_metadata = get_song_details(db_manager, song_uuid)
                    st.write(f"**Song UUID:** `{song_uuid}`")
                    if song_metadata:
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
                        st.warning(f"Full metadata for '{song_name}' not found.")

def display_audience_chart(audience_data):
    """Displays processed audience chart data as a table for debugging."""
    st.subheader("Audience on Spotify")
    if not audience_data:
        st.info("No audience data available for the selected period.")
        return

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

    if 'value' in df.columns:
        st.line_chart(df['value'])
    else:
        st.warning("Could not find 'value' data to plot.")

def display_streaming_audience_chart(streaming_data):
    """Displays streaming audience data in a line chart."""
    st.subheader("Streaming Audience (Spotify)")
    if not streaming_data:
        st.info("No streaming audience data available for the selected period.")
        return

    chart_data = [{'date': entry.get('date'), 'value': entry.get('value')} for entry in streaming_data]
    df = pd.DataFrame(chart_data)

    if 'date' not in df.columns or df['date'].isnull().all():
        st.error("Streaming audience data is missing valid 'date' information.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if 'value' in df.columns:
        st.line_chart(df['value'])
    else:
        st.warning("Could not find 'value' data in the streaming audience response.")


def display_local_streaming_plots(local_streaming_data):
    """
    Creates interactive dropdowns for country and city to plot time-series streaming data.
    """
    st.subheader("Local Streaming Performance")
    if not local_streaming_data:
        st.info("No local streaming data available for this period.")
        return

    all_countries = set()
    country_to_cities = {}
    plot_data = []

    for daily_entry in local_streaming_data:
        date = daily_entry.get('date')
        for country_plot in daily_entry.get('countryPlots', []):
            country_name = country_plot.get('countryName')
            if country_name:
                all_countries.add(country_name)
                plot_data.append({
                    'date': date, 'country': country_name, 'city': None, 'streams': country_plot.get('value')
                })
        for city_plot in daily_entry.get('cityPlots', []):
            country_name = city_plot.get('countryName')
            city_name = city_plot.get('cityName')

            if country_name and city_name:
                all_countries.add(country_name)
                if country_name not in country_to_cities:
                    country_to_cities[country_name] = set()
                country_to_cities[country_name].add(city_name)
                plot_data.append({
                    'date': date, 'country': country_name, 'city': city_name, 'streams': city_plot.get('value')
                })
    
    if not plot_data:
        st.info("Could not find any stream data in the local breakdown.")
        return
        
    df = pd.DataFrame(plot_data)
    df['date'] = pd.to_datetime(df['date'])

    sorted_countries = sorted(list(all_countries))
    selected_country = st.selectbox("Select a Country", sorted_countries)

    if selected_country:
        cities = sorted(list(country_to_cities.get(selected_country, [])))
        city_options = ["All Cities"] + cities
        selected_city = st.selectbox("Select a City", city_options)

        title = f"Daily Streams in {selected_country}"
        if selected_city and selected_city != "All Cities":
            filtered_df = df[(df['country'] == selected_country) & (df['city'] == selected_city)]
            title = f"Daily Streams in {selected_city}, {selected_country}"
        else:
            filtered_df = df[(df['country'] == selected_country) & (df['city'].isnull())]
        
        if not filtered_df.empty:
            fig = px.line(filtered_df, x='date', y='streams', title=title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No streaming data found for the selected location.")
