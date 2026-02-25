CREATE TABLE events (
    event_uuid UUID PRIMARY KEY,
    artist_uuid UUID NOT NULL REFERENCES artists(artist_uuid) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100),
    status VARCHAR(100),
    capacity INTEGER,
    event_date DATE,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    
    -- Flattened Price Object
    price_currency VARCHAR(10),
    price_min INTEGER,
    price_max INTEGER,
    
    -- Flattened Venue Object
    venue_uuid UUID,
    venue_name VARCHAR(255),
    venue_city_name VARCHAR(255),
    venue_country_code VARCHAR(10),
    venue_region VARCHAR(255),
    
    -- Flattened Festival Object
    festival_uuid UUID,
    festival_name VARCHAR(255)
);
