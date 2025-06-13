# main.py

from soundcharts_client import SoundchartsAPIClient
from database_manager import DatabaseManager
import config  # Import your configuration

def main():
    """Main function to run the data fetching and storing process."""
    
    # --- Configuration ---
    ARTIST_TO_FETCH = "Daft Punk"  # Example artist

    # --- Initialization ---
    try:
        api_client = SoundchartsAPIClient(app_id=config.APP_ID, api_key=config.API_KEY)
        db_manager = DatabaseManager(mongo_uri=config.MONGO_URI, db_name=config.DB_NAME)
    except (ValueError, ConnectionFailure) as e:
        print(f"Initialization failed: {e}")
        return # Exit if clients can't be initialized

    # --- Execution ---
    try:
        print(f"--- Starting data fetch for '{ARTIST_TO_FETCH}' ---")
        artist_data = api_client.fetch_full_artist_data(ARTIST_TO_FETCH)
        
        if 'error' in artist_data:
            print(f"Error fetching data: {artist_data['error']}")
        else:
            print(f"--- Successfully fetched data for artist UUID: {artist_data['artist_uuid']} ---")
            print(f"--- Starting data storage process ---")
            storage_result = db_manager.store_artist_data(artist_data)
            print(f"Storage status: {storage_result['message']}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- Cleanup ---
        # Ensure the database connection is always closed
        if 'db_manager' in locals() and db_manager.client:
            db_manager.close_connection()
        print("--- Process finished ---")


if __name__ == "__main__":
    main()

