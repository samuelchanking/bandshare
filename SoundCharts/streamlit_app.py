from soundcharts_api import SoundchartsAPI
import streamlit as st
import pandas as pd
from pymongo import MongoClient

# Initialize the Soundcharts API client
api = SoundchartsAPI(db_name='soundcharts3')

# Streamlit App Layout
st.title("Soundcharts Data Fetcher")

# Section 1: Display Table of Artists
st.subheader("List of Artists in Database")
try:
    # Fetch artists from the 'artists' collection
    artists_data = list(api.collections['artists'].find({}, {'name': 1, 'artist_uuid': 1, '_id': 0}))
    if artists_data:
        # Convert to DataFrame for display
        df = pd.DataFrame(artists_data)
        st.dataframe(df)
    else:
        st.write("No artists found in the database.")
except Exception as e:
    st.error(f"Error fetching artists: {e}")

# Section 2: Input Fields and Confirmation Button
st.subheader("Fetch Data for a New Artist")
artist_name = st.text_input("Artist Name (e.g., Billie Eilish):", "Billie Eilish")
start_date = st.text_input("Start Date (YYYY-MM-DD, e.g., 2023-05-31):", "2023-05-31")
end_date = st.text_input("End Date (YYYY-MM-DD, e.g., 2025-05-30):", "2025-05-30")

if st.button("Fetch and Store Data"):
    try:
        with st.spinner("Fetching and storing data..."):
            api.fetch_and_store_artist_data(artist_name, start_date, end_date)
        st.success(f"Data fetched and stored for {artist_name} from {start_date} to {end_date}!")
        # Refresh the table after adding new data
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        # Do not close connection here to allow further interactions
        pass

# Add a note for console logs
st.write("Check the console for detailed logs of the data fetching process.")

# Close the connection when the app shuts down (optional)
# Note: Streamlit doesn't have a clean shutdown hook, so this is informational
st.write("Note: Close the app to terminate the MongoDB connection manually.")
