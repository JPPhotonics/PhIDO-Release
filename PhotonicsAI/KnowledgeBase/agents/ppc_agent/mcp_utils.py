"""Utilities for accessing MCP tools using subprocess-based client."""

from typing import Optional, Callable

# Import subprocess-based MCP client (reliable, bypasses SDK issues)
try:
    from .mcp_client_subprocess import create_subprocess_annotator_wrapper
    MCP_CLIENT_AVAILABLE = True
    print("  Debug: Successfully imported subprocess MCP client")
except ImportError as e:
    print(f"  Debug: Failed to import subprocess MCP client: {e}")
    MCP_CLIENT_AVAILABLE = False
    create_subprocess_annotator_wrapper = None
except Exception as e:
    print(f"  Debug: Exception importing subprocess MCP client: {e}")
    MCP_CLIENT_AVAILABLE = False
    create_subprocess_annotator_wrapper = None


def get_mcp_annotator_func() -> Optional[Callable]:
    """
    Get the MCP annotator function using subprocess-based client.
    
    Uses the subprocess MCP client which bypasses SDK initialization issues.
    
    Returns:
        MCP annotator function if available, None otherwise
    """
    if MCP_CLIENT_AVAILABLE and create_subprocess_annotator_wrapper:
        try:
            print("  Debug: Creating subprocess MCP annotator wrapper...")
            wrapper = create_subprocess_annotator_wrapper()
            if wrapper:
                print("  Debug: Subprocess MCP wrapper created successfully")
                return wrapper
            else:
                print("  Debug: create_subprocess_annotator_wrapper() returned None")
        except Exception as e:
            print(f"  Debug: Subprocess MCP client creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    return None


def create_mcp_annotator_wrapper(mcp_tool_func: Optional[Callable] = None) -> Optional[Callable]:
    """
    Create a wrapper function for the MCP annotator tool.
    
    Args:
        mcp_tool_func: The MCP tool function. If None, tries to get it automatically.
        
    Returns:
        Wrapper function that can be used by MCPDocumentAnnotator, or None
    """
    if mcp_tool_func is not None:
        return mcp_tool_func
    
    # Try to get from execution context
    func = get_mcp_annotator_func()
    return func

