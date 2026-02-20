#!/bin/bash
# Script to start Neo4j Docker container with persistent data and crash recovery.

set -eu

CONTAINER_NAME="neo4j-phido"
NEO4J_IMAGE="neo4j:5.15-enterprise"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# Persistent directories — all on the host so nothing is lost on crash/restart
DATA_DIR="$(pwd)/neo4j_data"
LOGS_DIR="$(pwd)/neo4j_logs"
PLUGINS_DIR="$(pwd)/neo4j_plugins"

ensure_dirs() {
    mkdir -p "$DATA_DIR" "$LOGS_DIR" "$PLUGINS_DIR"
}

# ---------------------------------------------------------------------------
# Recovery: remove stale Neo4j lock files that block restart after a crash
# ---------------------------------------------------------------------------
recover_lock_files() {
    echo "Checking for stale lock files after ungraceful shutdown..."
    local found=0
    # Neo4j uses store_lock files inside each database directory
    for lock_file in "$DATA_DIR"/databases/*/store_lock; do
        if [ -f "$lock_file" ]; then
            echo "  Removing stale lock: $lock_file"
            sudo rm -f "$lock_file"
            found=1
        fi
    done
    if [ "$found" -eq 1 ]; then
        echo "  ✓ Stale lock files removed — Neo4j should recover on next start."
    else
        echo "  No stale lock files found."
    fi
}

wait_for_neo4j() {
    local max_wait=${1:-60}
    local elapsed=0
    echo "Waiting for Neo4j to become ready (up to ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        # Check HTTP endpoint (returns 200 when ready)
        if curl -sf http://localhost:7474 >/dev/null 2>&1; then
            echo "  ✓ Neo4j is ready (took ${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "  ⚠ Neo4j did not become ready within ${max_wait}s"
    echo "  Check logs: sudo docker logs $CONTAINER_NAME"
    return 1
}

print_connection_info() {
    echo "  Browser:  http://localhost:7474"
    echo "  Bolt:     bolt://localhost:7687"
    echo "  Username: neo4j"
    echo "  Password: $NEO4J_PASSWORD"
}

# ===== Main Logic =====

echo "=== Neo4j Docker Manager ==="
ensure_dirs

# --- Case 1: Container already exists ---
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container '$CONTAINER_NAME' exists."

    if sudo docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "✓ Neo4j container is already running."
        print_connection_info
        exit 0
    fi

    # Container exists but stopped — likely a crash or system reboot
    echo "Container is stopped. Attempting recovery restart..."
    recover_lock_files

    if sudo docker start "$CONTAINER_NAME"; then
        if wait_for_neo4j 60; then
            echo "✓ Neo4j container restarted successfully."
            print_connection_info
            exit 0
        fi
    fi

    # If restart still fails, recreate the container (data survives in volumes)
    echo "⚠ Restart failed. Recreating container (data is preserved in $DATA_DIR)..."
    sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    recover_lock_files
    # Fall through to creation below
fi

# --- Case 2: Create new container ---
echo "Creating new container '$CONTAINER_NAME'..."

sudo docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p 7474:7474 -p 7687:7687 \
    -v "$DATA_DIR:/data" \
    -v "$LOGS_DIR:/logs" \
    -v "$PLUGINS_DIR:/plugins" \
    -e NEO4J_AUTH=neo4j/"$NEO4J_PASSWORD" \
    -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
    -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    -e NEO4J_server_memory_heap_initial__size=512m \
    -e NEO4J_server_memory_heap_max__size=1G \
    -e NEO4J_db_tx__log_rotation_retention__policy="2 days" \
    "$NEO4J_IMAGE"

if wait_for_neo4j 90; then
    echo "✓ Neo4j container created and started successfully."
    print_connection_info
    exit 0
else
    echo "✗ Failed to create/start container."
    echo "Check logs: sudo docker logs $CONTAINER_NAME"
    exit 1
fi
