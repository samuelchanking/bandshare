CREATE TABLE songs (
    song_uuid UUID PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    album_uuid UUID REFERENCES albums(album_uuid) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    track_number INTEGER, -- Pulled from the album tracklist concept
    duration INTEGER,
    distributor VARCHAR(255),
    image_url TEXT,
    isrc VARCHAR(100),
    release_date DATE,
    composers TEXT[],
    producers TEXT[],
    genres TEXT[]
);
