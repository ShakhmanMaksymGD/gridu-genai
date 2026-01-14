#!/bin/bash
# Database initialization script

set -e

# Create additional databases or extensions if needed
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create extensions that might be useful
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    
    -- Create a schema for temporary tables if needed
    CREATE SCHEMA IF NOT EXISTS temp_data;
    
    -- Grant permissions
    GRANT ALL PRIVILEGES ON SCHEMA temp_data TO $POSTGRES_USER;
    
    -- Log initialization
    SELECT 'Database initialized successfully!' AS status;
EOSQL

echo "Database initialization completed."