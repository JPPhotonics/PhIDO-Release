"""Test script to verify MCP client setup."""

import sys

print("Testing MCP SDK availability...")

# Test 1: Check if MCP SDK is installed
try:
    import mcp
    print("✓ MCP SDK is installed")
    print(f"  MCP module location: {mcp.__file__}")
except ImportError as e:
    print(f"✗ MCP SDK not installed: {e}")
    print("  Install with: pip install mcp")
    sys.exit(1)

# Test 2: Check if we can import MCP client components
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.types import StdioServerParameters
    print("✓ MCP client components can be imported")
except ImportError as e:
    print(f"✗ Cannot import MCP client components: {e}")
    sys.exit(1)

# Test 3: Check if our MCP client module can be imported
try:
    from PhotonicsAI.KnowledgeBase.agents.ppc_agent.mcp_client import (
        MCPAnnotatorClient,
        create_sync_annotator_wrapper,
        MCP_AVAILABLE
    )
    print(f"✓ MCP client module imported successfully")
    print(f"  MCP_AVAILABLE: {MCP_AVAILABLE}")
except ImportError as e:
    print(f"✗ Cannot import MCP client module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try to create the wrapper
try:
    wrapper = create_sync_annotator_wrapper()
    if wrapper:
        print("✓ MCP annotator wrapper created successfully")
        print(f"  Wrapper type: {type(wrapper)}")
    else:
        print("⚠ MCP annotator wrapper is None (MCP_AVAILABLE might be False)")
except Exception as e:
    print(f"✗ Failed to create wrapper: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check mcp_utils
try:
    from PhotonicsAI.KnowledgeBase.agents.ppc_agent.mcp_utils import (
        get_mcp_annotator_func,
        MCP_CLIENT_AVAILABLE
    )
    print(f"✓ mcp_utils imported successfully")
    print(f"  MCP_CLIENT_AVAILABLE: {MCP_CLIENT_AVAILABLE}")
    
    # Try to get the function
    func = get_mcp_annotator_func()
    if func:
        print("✓ get_mcp_annotator_func() returned a function")
    else:
        print("⚠ get_mcp_annotator_func() returned None")
except Exception as e:
    print(f"✗ Error with mcp_utils: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All MCP client tests passed!")
print("="*60)

