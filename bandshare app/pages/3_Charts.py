# pages/3_Charts.py

import streamlit as st
import config
from pymongo.errors import ConnectionFailure
from streamlit_caching import (
    initialize_clients,
    get_audience_data, get_popularity_data,
    get_streaming_audience_from_db,
    get_local_streaming_history_from_db
)
from streamlit_ui import (
    display_audience_chart,
    display_popularity_chart,
    display_streaming_audience_chart,
    display_local_streaming_plots
)
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Initialization ---
st.set_page_config(page_title="Artist Charts", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Helper: Update Logic ---
def update_timeseries_data(artist_uuid, start_date_filter, end_date_filter):
    try:
        with st.spinner("Checking and updating artist time-series data..."):
            tasks = {
                'audience': (lambda start, end: api_client.get_artist_audience(artist_uuid, 'spotify', start, end), 'audience'),
                'popularity': (lambda start, end: api_client.get_artist_popularity(artist_uuid, 'spotify', start, end), 'popularity'),
                'streaming_audience': (lambda start, end: api_client.get_artist_streaming_audience(artist_uuid, 'spotify', start, end), 'streaming_audience'),
                'local_streaming_audience': (lambda start, end: api_client.get_local_streaming_audience(artist_uuid, 'spotify', start, end), 'local_streaming_history')
            }
            
            full_ts_data = {}
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_name = {}
                for name, (func, coll_name) in tasks.items():
                    query_filter = {'artist_uuid': artist_uuid, ('source' if name == 'popularity' else 'platform'): 'spotify'}
                    min_db_date, max_db_date = db_manager.get_timeseries_data_range(coll_name, query_filter)

                    if min_db_date and start_date_filter < min_db_date.date():
                        backfill_end_date = min_db_date.date() - timedelta(days=1)
                        future_to_name[executor.submit(func, start_date_filter, backfill_end_date)] = name

                    forward_start_date = max_db_date.date() + timedelta(days=1) if max_db_date else start_date_filter
                    if forward_start_date <= end_date_filter:
                         future_to_name[executor.submit(func, forward_start_date, end_date_filter)] = name
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        data = future.result()
                        if 'error' not in data and 'items' in data and data['items']:
                            if name not in full_ts_data:
                                platform_or_source = 'source' if name == 'popularity' else 'platform'
                                full_ts_data[name] = {'items': [], platform_or_source: data.get(platform_or_source)}
                            full_ts_data[name]['items'].extend(data['items'])
                    except Exception as exc:
                        st.warning(f'{name} data fetching generated an exception: {exc}')

            if full_ts_data:
                db_manager.store_timeseries_data(artist_uuid, full_ts_data)
                st.cache_data.clear()
                st.success("Artist chart data updated successfully.")
                st.rerun()
            else:
                st.info("No new artist data to update for the selected range.")

    except Exception as e:
        st.error(f"An unexpected error occurred during the update process: {e}")

# --- Page Content ---
if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to view their charts.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')
st.header(f"Time-Series Charts for {artist_name}")

# --- Filters and Update Button ---
st.markdown("### Chart Data")
start_date_filter = st.date_input("Chart Start Date", date.today() - timedelta(days=365))
end_date_filter = st.date_input("Chart End Date", date.today())

if st.button("Get/Update Chart Data", use_container_width=True, type="primary"):
    update_timeseries_data(artist_uuid, start_date_filter, end_date_filter)

# --- Fetch and Display Charts ---
audience_data = get_audience_data(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
popularity_data = get_popularity_data(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
streaming_data = get_streaming_audience_from_db(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)
local_streaming_data = get_local_streaming_history_from_db(db_manager, artist_uuid, "spotify", start_date_filter, end_date_filter)

display_audience_chart(audience_data)
display_popularity_chart(popularity_data)
display_streaming_audience_chart(streaming_data)
display_local_streaming_plots(local_streaming_data)