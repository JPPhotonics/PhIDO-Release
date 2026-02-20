# MCP Client Troubleshooting

## Current Issue: Initialization Timeout

The MCP client is able to start the server (you see the FastMCP banner), but the initialization handshake times out. This is a known issue with the current MCP Python SDK integration.

## Status

- ✅ MCP SDK is installed and imports correctly
- ✅ MCP client module loads successfully  
- ✅ Server process starts (FastMCP banner appears)
- ❌ Session initialization times out (30-60 seconds)
- ✅ Fallback to PyPDF2 works correctly

## Why This Happens

The MCP server starts as a subprocess via stdio, but the initialization protocol handshake isn't completing. This could be due to:

1. **Server startup time**: The server may need more time to fully initialize
2. **Communication protocol**: There may be a mismatch in the MCP protocol version
3. **Server configuration**: The server command/args might need adjustment

## Current Workaround

The PPC Agent automatically falls back to PyPDF2 when MCP initialization fails. This works correctly and extracts text from PDFs for entity extraction.

## Future Solutions

1. **Wait for MCP SDK updates**: The Python MCP SDK is still evolving
2. **Use HTTP transport**: If the server supports HTTP instead of stdio
3. **Direct subprocess call**: Call the MCP server directly via subprocess and parse output
4. **Use Cursor's MCP integration**: Access MCP tools through Cursor's interface (not from Python code)

## Testing the Server Manually

You can test if the server command works:

```bash
/home/tofu8/.local/bin/uvx --quiet --from axiomatic-mcp all
```

If this hangs or doesn't respond, the server itself may have issues.

## Recommendation

For now, use the PyPDF2 fallback which works reliably. The MCP integration can be enabled later when the initialization issue is resolved.

