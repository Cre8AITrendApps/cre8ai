#!/bin/bash
set -e

# Fix permissions for mounted volumes
echo "Fixing permissions for /app/db, /app/data, and /app/cache..."
chown -R appuser:appuser /app/db /app/data /app/cache

# Remove the chroma_db file if it's a file instead of a directory
if [ -f "/app/db/chroma_db" ]; then
    echo "Removing invalid chroma_db file..."
    rm "/app/db/chroma_db"
fi

# Execute the main command as appuser
exec gosu appuser "$@"
