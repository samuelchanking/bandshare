# pages/5_Reports.py

import streamlit as st

import pandas as pd

from datetime import date, timedelta

import json

import config

from client_setup import initialize_clients

from pymongo.errors import ConnectionFailure

# --- Page Configuration & Initialization ---

st.set_page_config(page_title="Spike Analysis Reports", layout="wide")

try:
    api_client, db_manager = initialize_clients(config)
except (ConnectionFailure, ValueError) as e:
    st.error(f"Failed to initialize clients: {e}")
    st.stop()

# --- Page Content ---

if not st.session_state.get('artist_uuid'):
    st.info("Please search for an artist on the Home page to generate a report.")
    st.stop()

artist_uuid = st.session_state.artist_uuid
artist_name = st.session_state.get('artist_name', 'the selected artist')

# --- Modified Header with Centered Artist Name and Trust Score ---
# Use markdown with CSS to center the header
st.markdown(
    f"""
    <h1 style='text-align: center;'>{artist_name}</h1>
    """,
    unsafe_allow_html=True
)

# Trust Score (default 100) displayed as circular progress with number in the middle
trust_score = 100  # Default value as per request
st.subheader("Artist Trust Score")

# Custom HTML/CSS for circular progress bar
circular_progress_html = f"""
<div style="position: relative; width: 150px; height: 150px; margin: auto;">
    <svg viewBox="0 0 36 36" style="position: absolute; width: 100%; height: 100%;">
        <path d="M18 2.0845
                 a 15.9155 15.9155 0 0 1 0 31.831
                 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke="#eee"
              stroke-width="3.8"
              stroke-dasharray="100, 100"></path>
        <path d="M18 2.0845
                 a 15.9155 15.9155 0 0 1 0 31.831
                 a 15.9155 15.9155 0 0 1 0 -31.831"
              fill="none"
              stroke="#4caf50"
              stroke-width="3.8"
              stroke-dasharray="{trust_score}, 100"></path>
    </svg>
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 24px; font-weight: bold;">
        {trust_score}%
    </div>
</div>
"""
st.markdown(circular_progress_html, unsafe_allow_html=True)

# --- Report Parameters ---

st.markdown("### Report Parameters")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=1095))
with col2:
    end_date = st.date_input("End Date", date.today())
