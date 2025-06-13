# delete_artist.py

import config # Your config file with database settings
from database_manager import DatabaseManager

# --- IMPORTANT ---
# Paste the exact artist_uuid of the artist you wish to delete from the database.
# You can get this UUID from the dropdown menu in the Streamlit application.
# Example: '0115b868-29a8-4501-8276-374649585641'
ARTIST_UUID_TO_DELETE = '11e83fe6-0e33-2c46-a628-aa1c026db3d8'


def main():
    """
    Main function to connect to the database and delete an artist's data.
    """
    if not ARTIST_UUID_TO_DELETE or ARTIST_UUID_TO_DELETE == 'PASTE_ARTIST_UUID_TO_DELETE_HERE':
        print("\nERROR: Please open delete_artist.py and set the ARTIST_UUID_TO_DELETE variable.")
        return

    db_manager = None
    try:
        # Initialize the database manager
        db_manager = DatabaseManager(mongo_uri=config.MONGO_URI, db_name=config.DB_NAME)

        # Confirm with the user before deleting
        # This is a safety measure to prevent accidental data loss.
        confirm = input(f"Are you sure you want to delete all data for artist {ARTIST_UUID_TO_DELETE}? This cannot be undone. (yes/no): ")
        
        if confirm.lower() == 'yes':
            # Call the new delete function
            result = db_manager.delete_artist_data(ARTIST_UUID_TO_DELETE)
            print(f"\nResult: {result['message']}")
        else:
            print("\nDeletion cancelled by user.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Ensure the database connection is always closed
        if db_manager:
            db_manager.close_connection()

if __name__ == "__main__":
    main()
