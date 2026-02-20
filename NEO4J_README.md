# Neo4j Knowledge Base — Docker Operations

This document covers how to start, stop, persist, back up, and restore the Neo4j graph database used by PhIDO's Knowledge Base.

---

## Prerequisites

- Docker Engine installed (`sudo docker --version`)
- Docker Compose V2 (`sudo docker compose version`) — optional but recommended
- `curl` — used by health checks

## Quick Start

There are two ways to run Neo4j. Pick one and stick with it.

### Option A: Shell Script (simple)

```bash
# Start (or restart) the container
./start_neo4j.sh

# The script will:
#   1. Create the container if it doesn't exist
#   2. Recover from stale lock files if it crashed
#   3. Wait until Neo4j is actually ready before returning
```

### Option B: Docker Compose (recommended)

```bash
sudo docker compose up -d        # start
sudo docker compose logs -f      # tail logs
sudo docker compose down          # stop (data persists)
```

Both options configure:
- `--restart unless-stopped` so the container auto-restarts after crashes or reboots
- Persistent volumes for `/data`, `/logs`, and `/plugins`
- APOC and Graph Data Science plugins
- A health check on the Neo4j HTTP endpoint

## Connection Details

| Setting  | Value                          |
|----------|--------------------------------|
| Browser  | http://localhost:7474          |
| Bolt URI | bolt://localhost:7687          |
| Username | `neo4j`                        |
| Password | `password` (or `$NEO4J_PASSWORD`) |

To use a custom password, set the environment variable before starting:

```bash
export NEO4J_PASSWORD=my_secure_password
./start_neo4j.sh
# or
sudo NEO4J_PASSWORD=my_secure_password docker compose up -d
```

The Python code reads the same variable via `Neo4jConfig` in
`PhotonicsAI/KnowledgeBase/Neo4j/config.py`.

## Data Persistence

### Where data lives

| Method         | Host Location                                  | Destroyed By                  |
|----------------|------------------------------------------------|-------------------------------|
| Shell script   | `./neo4j_data/`, `./neo4j_logs/`, `./neo4j_plugins/` (bind mounts) | Manual `rm -rf`               |
| Docker Compose | Docker named volumes (`neo4j_data`, `neo4j_logs`, `neo4j_plugins`) | `docker compose down -v` only |

In both cases, `docker rm -f neo4j-phido` removes the **container** but **not** the data.

### Crash recovery

If the container stops due to a crash or power loss:

1. **Auto-restart**: the `unless-stopped` restart policy tells Docker to bring it back automatically.
2. **Lock file cleanup**: `start_neo4j.sh` detects and removes stale `store_lock` files that Neo4j leaves behind after an ungraceful shutdown. If you use Docker Compose, Neo4j's built-in recovery typically handles this, but you can also run the script as a fallback.
3. **Recreate as last resort**: if the container is in a bad state, removing and recreating it is safe — the volumes survive.

```bash
# Manual recovery if auto-restart fails
sudo docker rm -f neo4j-phido
./start_neo4j.sh          # recreates container, reuses existing data directory
```

## Populating the Knowledge Base

### 1. Seed with foundational ontology

```bash
python reinitialize_neo4j_kb.py
```

This imports the YAML primitives from `PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives/` into Neo4j, creating the base nodes and relationships. It runs `client.initialize(reset=True)` so it **wipes and rebuilds** from scratch — only run this when you want a clean slate.

### 2. Ingest papers

```bash
# Place PDF files in the papers/ directory, then:
python process_papers.py
```

This runs each PDF through the PPC → VSA → DIA pipeline, adding extracted entities and relationships to the graph. Output reports are saved to `output/<paper_name>/`.

## Backup & Restore

After ingesting papers (which can take significant time and LLM cost), snapshot the database so you don't have to re-process if something goes wrong.

### Create a backup

```bash
./neo4j_backup.sh backup
# → backups/neo4j_kb_20260216_210000.dump
```

This stops the database briefly, creates a full dump (nodes, relationships, indexes, constraints), and restarts it.

### List available backups

```bash
./neo4j_backup.sh list
```

### Restore from a backup

```bash
./neo4j_backup.sh restore backups/neo4j_kb_20260216_210000.dump
```

You will be prompted to confirm since this **replaces** the current database.

### Recommended backup schedule

- **After `reinitialize_neo4j_kb.py`** — snapshot the clean foundational KB
- **After each batch of `process_papers.py`** — snapshot the enriched KB
- **Before any schema migration or experiment** — safety net

## Migrating Between Shell Script and Docker Compose

If you started with `./start_neo4j.sh` (bind mounts in `./neo4j_data/`) and want to switch to Docker Compose (named volumes):

```bash
# 1. Backup current data
./neo4j_backup.sh backup

# 2. Stop and remove old container
sudo docker rm -f neo4j-phido

# 3. Start via Compose (creates fresh named volumes)
sudo docker compose up -d

# 4. Restore your data
./neo4j_backup.sh restore backups/neo4j_kb_<timestamp>.dump
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container keeps restarting | Corrupted transaction log or lock file | `./start_neo4j.sh` handles this automatically; or manually: `sudo rm neo4j_data/databases/*/store_lock` then restart |
| `Neo4j did not become ready within 90s` | Slow first-time plugin download or low memory | Check `sudo docker logs neo4j-phido`; increase heap in compose/script env vars |
| `database already in use` | Stale lock file after crash | `./start_neo4j.sh` cleans this; or `sudo rm neo4j_data/databases/neo4j/store_lock` |
| Port 7474/7687 already in use | Another Neo4j or service on those ports | `sudo lsof -i :7474` to find it; change ports in compose/script |
| `docker compose` not found | Docker Compose V1 (`docker-compose`) installed | Use `sudo docker-compose up -d` or install Compose V2 |

## File Reference

| File | Purpose |
|------|---------|
| `start_neo4j.sh` | Start/restart Neo4j container with crash recovery |
| `docker-compose.yml` | Declarative Neo4j service definition with named volumes |
| `neo4j_backup.sh` | Backup and restore the graph database |
| `reinitialize_neo4j_kb.py` | Seed the KB with foundational YAML ontology |
| `process_papers.py` | Ingest PDFs through the PPC → VSA → DIA pipeline |
| `PhotonicsAI/KnowledgeBase/Neo4j/config.py` | Python connection config (`Neo4jConfig`) |

