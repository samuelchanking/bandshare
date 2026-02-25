CREATE TABLE playlist_track_history (
    id SERIAL PRIMARY KEY,
    playlist_uuid UUID NOT NULL,
    artist_uuid UUID REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    song_uuid UUID REFERENCES songs(song_uuid) ON DELETE CASCADE,
    record_date DATE NOT NULL,
    track_position INTEGER NOT NULL,
    
    UNIQUE(playlist_uuid, song_uuid, record_date)
);
