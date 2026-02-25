-- Global Daily Total
CREATE TABLE local_streaming_summary (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    record_date DATE NOT NULL,
    total_streams INTEGER,
    UNIQUE(artist_uuid, platform, record_date)
);

-- Country-Level Breakdown
CREATE TABLE local_streaming_countries (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    record_date DATE NOT NULL,
    country_code VARCHAR(10) NOT NULL,
    country_name VARCHAR(255),
    streams INTEGER NOT NULL,
    UNIQUE(artist_uuid, platform, record_date, country_code)
);

-- City-Level Breakdown
CREATE TABLE local_streaming_cities (
    id SERIAL PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    platform VARCHAR(100) NOT NULL,
    record_date DATE NOT NULL,
    city_name VARCHAR(255) NOT NULL,
    country_code VARCHAR(10) NOT NULL,
    region VARCHAR(255),
    streams INTEGER NOT NULL,
    UNIQUE(artist_uuid, platform, record_date, city_name, country_code)
);
