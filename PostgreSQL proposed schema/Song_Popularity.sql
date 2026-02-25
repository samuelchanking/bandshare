CREATE TABLE song_popularity (
    id SERIAL PRIMARY KEY,
    song_uuid UUID NOT NULL REFERENCES songs(song_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    record_date DATE NOT NULL,
    popularity_value INTEGER NOT NULL,
    
    -- Prevent duplicate entries for the same day and platform
    UNIQUE(song_uuid, platform, record_date)
);
