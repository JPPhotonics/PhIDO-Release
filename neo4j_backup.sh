#!/bin/bash
# Neo4j Knowledge Base Backup & Restore
#
# Usage:
#   Backup:  ./neo4j_backup.sh backup              → creates timestamped dump in backups/
#   Restore: ./neo4j_backup.sh restore <dump_file>  → restores from a previous dump
#   List:    ./neo4j_backup.sh list                  → show available backups
#
# This uses `neo4j-admin database dump/load` which captures the full graph
# (nodes, relationships, indexes, constraints) in a single portable file.

set -euo pipefail

CONTAINER_NAME="neo4j-phido"
BACKUP_DIR="$(pwd)/backups"
DATABASE="neo4j"

mkdir -p "$BACKUP_DIR"

usage() {
    echo "Usage: $0 {backup|restore <file>|list}"
    echo ""
    echo "  backup              Create a timestamped dump of the Neo4j KB"
    echo "  restore <file>      Restore from a backup dump file"
    echo "  list                List available backups"
    exit 1
}

do_backup() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local dump_file="neo4j_kb_${timestamp}.dump"

    echo "=== Neo4j KB Backup ==="
    echo "Stopping the database for a consistent dump..."

    # Stop the database inside the container (neo4j-admin requires it)
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database stop "$DATABASE" 2>/dev/null || true

    echo "Creating dump..."
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database dump "$DATABASE" \
        --to-path=/data/backups/ --overwrite-destination=true

    # Copy dump out of container to host backup dir
    sudo docker cp "$CONTAINER_NAME:/data/backups/${DATABASE}.dump" "$BACKUP_DIR/$dump_file"

    # Restart the database
    echo "Restarting database..."
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database start "$DATABASE" 2>/dev/null || \
        sudo docker restart "$CONTAINER_NAME"

    echo "✓ Backup saved to: $BACKUP_DIR/$dump_file"
    echo "  Size: $(du -h "$BACKUP_DIR/$dump_file" | cut -f1)"
}

do_restore() {
    local dump_file="$1"

    if [ ! -f "$dump_file" ]; then
        # Check if it's a relative name in backup dir
        if [ -f "$BACKUP_DIR/$dump_file" ]; then
            dump_file="$BACKUP_DIR/$dump_file"
        else
            echo "✗ Dump file not found: $dump_file"
            exit 1
        fi
    fi

    echo "=== Neo4j KB Restore ==="
    echo "Source: $dump_file"
    echo ""
    echo "⚠ WARNING: This will REPLACE the current '$DATABASE' database."
    read -rp "Continue? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    # Copy dump into container
    sudo docker cp "$dump_file" "$CONTAINER_NAME:/data/restore.dump"

    # Stop database and load
    echo "Stopping database..."
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database stop "$DATABASE" 2>/dev/null || true

    echo "Loading dump..."
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database load "$DATABASE" \
        --from-path=/data/ --overwrite-destination=true

    # Restart
    echo "Restarting..."
    sudo docker exec "$CONTAINER_NAME" neo4j-admin database start "$DATABASE" 2>/dev/null || \
        sudo docker restart "$CONTAINER_NAME"

    echo "✓ Database restored from $dump_file"
}

do_list() {
    echo "=== Available Backups ==="
    if [ -d "$BACKUP_DIR" ] && ls "$BACKUP_DIR"/*.dump 1>/dev/null 2>&1; then
        ls -lh "$BACKUP_DIR"/*.dump
    else
        echo "  No backups found in $BACKUP_DIR/"
    fi
}

# --- Main ---
case "${1:-}" in
    backup)  do_backup ;;
    restore)
        if [ -z "${2:-}" ]; then
            echo "✗ Please specify a dump file to restore."
            usage
        fi
        do_restore "$2"
        ;;
    list)    do_list ;;
    *)       usage ;;
esac

