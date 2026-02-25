-- First, extract the nested Playlist object into its own table
CREATE TABLE playlists (
    playlist_uuid UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    platform VARCHAR(100),
    type VARCHAR(100),
    identifier VARCHAR(255),
    country_code VARCHAR(10),
    image_url TEXT,
    latest_subscriber_count INTEGER,
    latest_track_count INTEGER,
    latest_crawl_date DATE
);

-- Then, create the relational table linking Songs to Playlists
CREATE TABLE song_playlist_placements (
    id SERIAL PRIMARY KEY,
    song_uuid UUID NOT NULL REFERENCES songs(song_uuid) ON DELETE CASCADE,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    playlist_uuid UUID NOT NULL REFERENCES playlists(playlist_uuid) ON DELETE CASCADE,
    
    -- Added fields for joining and exiting the playlist
    join_date DATE, 
    exit_date DATE, 
    
    peak_position INTEGER,
    peak_position_date DATE,
    current_position INTEGER,
    position_date DATE,
    
    -- Array of all dates the song was present on the playlist
    tracklist_dates DATE[],
    
    -- Ensure a song only has one active performance record per playlist
    UNIQUE(song_uuid, playlist_uuid)
);
