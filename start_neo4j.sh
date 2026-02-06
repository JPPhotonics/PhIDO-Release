#!/bin/bash
# Script to start Neo4j Docker container

echo "Checking Neo4j Docker container status..."

CONTAINER_NAME="neo4j-phido"
NEO4J_IMAGE="neo4j:5.15-enterprise"
NEO4J_PASSWORD="password"

# Check if container exists
if sudo docker ps -a | grep -q $CONTAINER_NAME; then
    echo "Container '$CONTAINER_NAME' exists"
    
    # Check if it's running
    if sudo docker ps | grep -q $CONTAINER_NAME; then
        echo "✓ Neo4j container is already running"
        echo "  Browser: http://localhost:7474"
        echo "  Bolt:    bolt://localhost:7687"
        exit 0
    else
        echo "Container exists but is not running. Starting it..."
        sudo docker start $CONTAINER_NAME
        sleep 10 # Neo4j takes a bit longer to be ready
        if sudo docker ps | grep -q $CONTAINER_NAME; then
            echo "✓ Neo4j container started successfully"
            echo "  Browser: http://localhost:7474"
            echo "  Bolt:    bolt://localhost:7687"
            exit 0
        else
            echo "✗ Failed to start container. It might be in a bad state."
            echo "  Recommendation: Run 'sudo docker rm -f $CONTAINER_NAME' and rerun this script."
            exit 1
        fi
    fi
else
    echo "Container '$CONTAINER_NAME' does not exist. Creating new container..."
    # Ensure data directory exists
    mkdir -p neo4j_data
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        -p 7474:7474 -p 7687:7687 \
        -v "$(pwd)/neo4j_data:/data" \
        -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD \
        -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes \
        -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
        $NEO4J_IMAGE
        
    echo "Waiting for Neo4j to initialize (this may take 20-30 seconds)..."
    sleep 20
    if sudo docker ps | grep -q $CONTAINER_NAME; then
        echo "✓ Neo4j container created and started successfully"
        echo "  Browser: http://localhost:7474"
        echo "  Bolt:    bolt://localhost:7687"
        echo "  Username: neo4j"
        echo "  Password: $NEO4J_PASSWORD"
        exit 0
    else
        echo "✗ Failed to create/start container"
        echo "Check logs with: sudo docker logs $CONTAINER_NAME"
        exit 1
    fi
fi
