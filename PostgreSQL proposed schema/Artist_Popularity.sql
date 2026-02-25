CREATE TABLE artist_popularity (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    source VARCHAR(100) NOT NULL, -- e.g., 'spotify', 'apple_music'
    record_date DATE NOT NULL,
    popularity_value INTEGER NOT NULL,
    
    -- Ensure we don't accidentally insert duplicates for the same day and source
    UNIQUE(artist_uuid, source, record_date) 
);
