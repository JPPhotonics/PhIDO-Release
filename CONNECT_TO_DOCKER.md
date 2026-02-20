# Connecting PPC Agent Tests to Docker ArangoDB

## Quick Setup

Your Docker container is already running. To connect the PPC Agent tests, you need to set the `ARANGO_PASSWORD` environment variable to match the password you used when starting the Docker container.

### Option 1: Set Environment Variable (Recommended)

```bash
# Set the password to match your Docker container
export ARANGO_PASSWORD=my_password

# Then run the test
python test_ppc_agent.py
```

### Option 2: Use Default (Already Configured)

The test script now defaults to `ARANGO_PASSWORD=my_password` if not set, which matches the Docker command:
```bash
docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=my_password arangodb:latest
```

So you can just run:
```bash
python test_ppc_agent.py
```

### Option 3: Create .env File

Create a `.env` file in the project root:
```bash
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
ARANGO_PASSWORD=my_password
ARANGO_DATABASE=photonics_kb
```

Then load it before running:
```bash
export $(cat .env | xargs)
python test_ppc_agent.py
```

## Verify Docker Container is Running

```bash
# Check if container is running
sudo docker ps | grep arangodb

# If not running, start it
sudo docker start arangodb

# Or if you need to create a new one (after removing old one)
sudo docker rm arangodb
sudo docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=my_password arangodb:latest
```

## Connection Settings

The test script uses these defaults (can be overridden with environment variables):
- **Host**: `localhost`
- **Port**: `8529`
- **Username**: `root`
- **Password**: `my_password` (must match `ARANGO_ROOT_PASSWORD` in Docker)
- **Database**: `photonics_kb`

## Troubleshooting

### Connection Refused
- Make sure Docker container is running: `sudo docker ps`
- Check if port 8529 is accessible: `curl http://localhost:8529/_api/version`

### Authentication Failed
- Verify `ARANGO_PASSWORD` matches `ARANGO_ROOT_PASSWORD` from Docker command
- Check Docker logs: `sudo docker logs arangodb`

### Database Not Found
- The database will be created automatically on first connection
- Or initialize it manually: `python -c "from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient; c = KnowledgeBaseClient(); c.connect(); c.initialize()"`

