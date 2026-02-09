# Knowledge Graph Generation Pipeline -- Setup Guide

This guide walks through setting up the PhIDO agentic Knowledge Graph (KG) generation pipeline from scratch on a clean local environment. The pipeline processes photonics research PDFs through three agents (PPC, VSA, DIA) and an optional Schema Evolution Agent (SEA) to populate a Neo4j graph database.

---

## Prerequisites

- **OS**: Linux (tested on Ubuntu/WSL2) or macOS
- **Python**: 3.11 or 3.12
- **Docker**: Docker Engine (for Neo4j)
- **System packages**: `graphviz`, `libgraphviz-dev`, `pkg-config`, `build-essential`, `python3-dev`, `swig`
- **LLM API key**: At minimum, a Google Gemini API key (`GOOGLEGENAI_API_KEY`)

---

## 1. Install System Dependencies

### Ubuntu / WSL2

```bash
sudo apt update
sudo apt install -y graphviz libgraphviz-dev pkg-config build-essential python3-dev swig
```

### macOS

```bash
brew install graphviz pkg-config swig
```

### Docker

Follow the official Docker installation guide for your platform:
- Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- macOS: https://docs.docker.com/desktop/install/mac-install/

Verify Docker is running:

```bash
docker --version
```

---

## 2. Clone and Install Python Environment

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repo-url> PhIDO-Release
cd PhIDO-Release

# Install uv (if not already installed) and create the virtual environment
make install
```

This runs:
1. Installs `uv` if not present
2. Creates a Python 3.12 virtual environment
3. Installs all dependencies from `pyproject.toml` and `uv.lock`

### Manual alternative (without Make)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and sync dependencies
uv venv --python 3.12
uv sync
```

### Activate the environment

```bash
source .venv/bin/activate
```

### Additional KG pipeline dependencies

Some packages used by the KG pipeline are not in the base `pyproject.toml` (they are in `requirements.txt` or installed ad-hoc). Install them inside the activated venv:

```bash
pip install neo4j python-dotenv sentence-transformers pydantic pyvis scikit-learn PyPDF2
```

---

## 3. Start Neo4j via Docker

The project includes a convenience script to run Neo4j 5.15 Enterprise with APOC and Graph Data Science plugins pre-installed.

```bash
chmod +x start_neo4j.sh
./start_neo4j.sh
```

This creates a Docker container named `neo4j-phido` with:

| Setting | Value |
|---|---|
| HTTP Browser | http://localhost:7474 |
| Bolt endpoint | bolt://localhost:7687 |
| Username | `neo4j` |
| Password | `password` |
| Plugins | APOC, Graph Data Science (GDS) |
| Data volume | `./neo4j_data` (persisted on host) |

Wait ~20 seconds for Neo4j to finish initializing. Verify the connection by visiting http://localhost:7474 in a browser.

### Troubleshooting

If the container is in a bad state:

```bash
sudo docker rm -f neo4j-phido
./start_neo4j.sh
```

---

## 4. Configure Environment Variables

Copy and customize the `.env` file in the project root. The defaults match the Docker container above:

```bash
# .env (already present in the repo -- edit as needed)

# --- Neo4j ---
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# --- Embedding Model ---
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# --- LLM API Keys (set at least one) ---
# Google Gemini (REQUIRED -- the pipeline defaults to gemini-2.5-pro / gemini-3-pro-preview)
GOOGLEGENAI_API_KEY=<your-google-api-key>

# Optional: additional providers
# OPENAI_API_KEY=<your-openai-key>
# ANTHROPIC_API_KEY=<your-anthropic-key>

# --- MCP (optional, for PDF-to-markdown) ---
# AXIOMATIC_API_KEY=<your-axiomatic-key>
```

The `GOOGLEGENAI_API_KEY` is the most important -- the PPC agent uses `gemini-3-pro-preview` and the VSA/DIA/SEA agents use `gemini-2.5-pro` by default. Set the key via either the `.env` file or a shell `export`:

```bash
export GOOGLEGENAI_API_KEY="your-key-here"
```

---

## 5. Initialize the Knowledge Base

### 5a. Seed the ontology

The KG starts from a YAML-based photonics ontology containing seed components, properties, design functions, and physical principles. Load it into Neo4j:

```bash
python reinitialize_neo4j_kb.py
```

This script:
1. Connects to Neo4j
2. Creates vector indexes and uniqueness constraints
3. Imports YAML ontology files from `PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives/`
4. Generates a verification visualization (`neo4j_kb_graph.html`)

Expected output:

```
Initializing Neo4j KB Client...
Connected to Neo4j at bolt://localhost:7687
Initializing Vector Indexes and Constraints...
Importing YAML data from PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives...
Success! Imported XX nodes and XX edges.
```

### 5b. Verify in browser

Open http://localhost:7474, log in with `neo4j` / `password`, and run:

```cypher
MATCH (n) RETURN labels(n) AS label, count(*) AS count
```

You should see counts for Component, Architecture, Property, Design_Function, and Physical_Principle nodes.

---

## 6. Process Papers

### 6a. Add PDFs

Place your research paper PDFs in the `papers/` directory:

```bash
mkdir -p papers
cp /path/to/your/papers/*.pdf papers/
```

### 6b. Run the pipeline

```bash
python process_papers.py
```

This runs each PDF through the full pipeline:

1. **PPC Agent** -- Extracts and normalizes entities from the PDF
2. **VSA Agent** -- Validates entities, infers relationships, creates a transaction manifest
3. **DIA Agent** -- Commits nodes/edges to Neo4j, runs global inference
4. **SEA Agent** -- (post-batch) Analyses novel relationship observations, promotes types that meet statistical thresholds

### 6c. Pipeline outputs

After completion, the `output/` directory contains:

```
output/
  <paper_name>/
    ppc_result.json          # Extracted entities (known + new)
    ppc_report.html          # Color-coded entity matching report
    vsa_manifest.json        # Transaction manifest (nodes + edges)
    vsa_graph.html           # Interactive PyVis graph of the manifest
    dia_report.json          # Integration metrics
    dia_report.html          # DIA performance report
  schema_evolution_report.json   # SEA promotion/recategorization results
  full_llm_trace.json            # Every LLM call across all stages
  full_kb_graph.html             # Complete knowledge graph visualization
```

---

## 7. Review Queue (Optional)

Items that lack sufficient evidence or fail verification are queued for human review. A Streamlit app is provided:

```bash
streamlit run review_queue_app.py
```

This opens a browser UI where you can inspect, approve, reject, or re-commit queued nodes and edges.

---

## Quick Reference

| Task | Command |
|---|---|
| Start Neo4j | `./start_neo4j.sh` |
| Stop Neo4j | `sudo docker stop neo4j-phido` |
| Initialize / reset KB | `python reinitialize_neo4j_kb.py` |
| Process papers | `python process_papers.py` |
| Review queue UI | `streamlit run review_queue_app.py` |
| Run tests | `make test` or `pytest -s` |
| Neo4j browser | http://localhost:7474 |

---

## Pipeline Architecture

```
PDF papers
    |
    v
[PPC Agent]  -- entity extraction, acronym resolution, KB grounding
    |
    v
[VSA Agent]  -- architecture validation, edge inference, knowledge merge
    |
    v
[DIA Agent]  -- graph commit, global inference (semantic + PageRank)
    |
    v
[SEA Agent]  -- (post-batch) schema evolution, relationship type promotion
    |
    v
Neo4j Knowledge Graph
```

---

## Troubleshooting

### "Failed to connect to Neo4j"
- Verify the container is running: `sudo docker ps | grep neo4j-phido`
- Check Neo4j logs: `sudo docker logs neo4j-phido`
- Ensure `.env` credentials match the Docker container (`password` by default)

### "Vector search failed (index ... might be missing)"
- Run `python reinitialize_neo4j_kb.py` to recreate indexes

### Embedding model download hangs
- The first run downloads `Qwen/Qwen3-Embedding-0.6B` (~1.2 GB). Ensure you have a stable internet connection. The model is cached in `~/.cache/huggingface/` for subsequent runs.

### "sklearn not available" warning from SEA
- Install scikit-learn: `pip install scikit-learn`

### LLM API errors
- Verify your `GOOGLEGENAI_API_KEY` is set and valid
- Check rate limits on your Gemini API plan
