CREATE TABLE artist_audience (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL, -- e.g., 'instagram', 'tiktok', 'youtube'
    record_date DATE NOT NULL,
    follower_count BIGINT NOT NULL,
    following_count INTEGER,
    like_count BIGINT,
    post_count INTEGER,
    view_count BIGINT,
    
    UNIQUE(artist_uuid, platform, record_date)
);
