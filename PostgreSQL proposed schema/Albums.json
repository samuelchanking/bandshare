CREATE TABLE albums (
    album_uuid UUID PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    credit_name VARCHAR(255),
    image_url TEXT,
    release_date DATE,
    total_tracks INTEGER,
    type VARCHAR(100),
    upc VARCHAR(100)
);
