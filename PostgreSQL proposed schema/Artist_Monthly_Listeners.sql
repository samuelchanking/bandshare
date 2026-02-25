CREATE TABLE streaming_audience (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    record_date DATE NOT NULL,
    audience_value INTEGER NOT NULL,
    
    -- Prevent duplicate entries for the same day and platform
    UNIQUE(artist_uuid, platform, record_date)
);
