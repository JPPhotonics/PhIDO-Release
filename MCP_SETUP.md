# Setting Up MCP Tools Access in Python

To make MCP tools accessible in your Python code, you need to install the MCP Python SDK and configure it to connect to your MCP server.

## Installation

Install the MCP Python SDK:

```bash
pip install mcp
```

## How It Works

The MCP client (`mcp_client.py`) connects to the axiomatic-mcp server using stdio (standard input/output), which is how Cursor runs MCP servers. The client:

1. Spawns the MCP server as a subprocess (using the command from `.cursor/mcp.json`)
2. Communicates with it via stdio
3. Provides a synchronous wrapper function that can be used by the PPC Agent

## Usage

Once installed, the PPC Agent will automatically try to use the MCP client:

```python
from PhotonicsAI.KnowledgeBase.agents.ppc_agent import PPCAgent
from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient

# The agent will automatically try to connect to MCP server
kb_client = KnowledgeBaseClient()
kb_client.connect()

agent = PPCAgent(kb_client=kb_client, llm_model="gemini-2.5-pro")

# This will use MCP annotation if available, otherwise fallback to PyPDF2
result = agent.process_paper(pdf_path=Path("paper.pdf"))
```

## Manual Usage

You can also use the MCP client directly:

```python
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.mcp_client import (
    MCPAnnotatorClient,
    create_sync_annotator_wrapper
)

# Option 1: Use the wrapper function
annotator_func = create_sync_annotator_wrapper()
if annotator_func:
    result = annotator_func(
        file_path="/path/to/file.pdf",
        query="Extract all sections about photonic components..."
    )

# Option 2: Use the client directly (async)
import asyncio
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.mcp_client import MCPAnnotatorClient

async def main():
    client = MCPAnnotatorClient()
    result = await client.annotate_file(
        file_path="/path/to/file.pdf",
        query="Extract all sections about photonic components..."
    )
    print(result)
    await client.close()

asyncio.run(main())
```

## Configuration

The MCP client uses the configuration from `.cursor/mcp.json`:

- **Server command**: `/home/tofu8/.local/bin/uvx`
- **Server args**: `["--quiet", "--from", "axiomatic-mcp", "all"]`
- **API Key**: From environment variable `AXIOMATIC_API_KEY` or the default in mcp.json

You can override these by creating a custom client:

```python
client = MCPAnnotatorClient(
    server_command="/custom/path/to/uvx",
    server_args=["--from", "axiomatic-mcp", "all"]
)
```

## Troubleshooting

### MCP SDK Not Installed

If you see `ImportError: MCP Python SDK not installed`, install it:

```bash
pip install mcp
```

### Connection Issues

If the MCP client can't connect:

1. **Check server command**: Verify `/home/tofu8/.local/bin/uvx` exists and is executable
2. **Check API key**: Ensure `AXIOMATIC_API_KEY` is set or matches the one in `.cursor/mcp.json`
3. **Check server startup**: The server should start automatically when the client connects

### Fallback Behavior

If MCP is not available, the PPC Agent automatically falls back to direct PDF extraction using PyPDF2. This is expected and works correctly.

## Testing

Test the MCP client:

```python
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.mcp_client import create_sync_annotator_wrapper

annotator = create_sync_annotator_wrapper()
if annotator:
    result = annotator(
        file_path="/absolute/path/to/sample_paper.pdf",
        query="Extract all sections about photonic components"
    )
    print(result)
else:
    print("MCP client not available")
```

