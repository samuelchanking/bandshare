CREATE TABLE song_audience (
    id SERIAL PRIMARY KEY,
    song_uuid UUID NOT NULL REFERENCES songs(song_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    identifier VARCHAR(255),
    record_date DATE NOT NULL,
    audience_value INTEGER NOT NULL,
    
    -- Prevent duplicate daily entries for a specific song on a specific platform
    UNIQUE(song_uuid, platform, record_date)
);
