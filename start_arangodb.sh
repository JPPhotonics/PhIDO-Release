#!/bin/bash
# Script to start ArangoDB Docker container

echo "Checking ArangoDB Docker container status..."

# Check if container exists
if sudo docker ps -a | grep -q arangodb; then
    echo "Container 'arangodb' exists"
    
    # Check if it's running
    if sudo docker ps | grep -q arangodb; then
        echo "✓ ArangoDB container is already running"
        echo "  Access at: http://localhost:8529"
        exit 0
    else
        echo "Container exists but is not running. Starting it..."
        sudo docker start arangodb
        sleep 2
        if sudo docker ps | grep -q arangodb; then
            echo "✓ ArangoDB container started successfully"
            echo "  Access at: http://localhost:8529"
            exit 0
        else
            echo "✗ Failed to start container"
            exit 1
        fi
    fi
else
    echo "Container 'arangodb' does not exist. Creating new container..."
    sudo docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=my_password arangodb:latest
    sleep 3
    if sudo docker ps | grep -q arangodb; then
        echo "✓ ArangoDB container created and started successfully"
        echo "  Access at: http://localhost:8529"
        echo "  Username: root"
        echo "  Password: my_password"
        exit 0
    else
        echo "✗ Failed to create/start container"
        echo "Check logs with: sudo docker logs arangodb"
        exit 1
    fi
fi

