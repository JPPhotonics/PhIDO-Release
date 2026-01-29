"""Subprocess-based MCP client that bypasses the MCP Python SDK.

This implementation manually manages the MCP server process and JSON-RPC communication,
avoiding the initialization timeout issues in the MCP Python SDK.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from queue import Queue, Empty


class SubprocessMCPClient:
    """MCP client using direct subprocess communication (bypasses MCP SDK)."""
    
    def __init__(
        self,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """
        Initialize subprocess-based MCP client.
        
        Args:
            server_command: Command to run MCP server (defaults to 'uvx' from PATH)
            server_args: Arguments for MCP server
            env: Environment variables for server process
        """
        # Find uvx in PATH if not provided
        if server_command is None:
            uvx_path = shutil.which("uvx")
            if uvx_path:
                self.server_command = uvx_path
            else:
                # Fallback to common locations
                home = os.path.expanduser("~")
                fallback_path = os.path.join(home, ".local", "bin", "uvx")
                if os.path.exists(fallback_path):
                    self.server_command = fallback_path
                else:
                    raise RuntimeError(
                        "uvx not found in PATH or common locations. "
                        "Please install uvx: pip install uv or curl -LsSf https://astral.sh/uv/install.sh | sh"
                    )
        else:
            self.server_command = server_command
        self.server_args = server_args or ["--quiet", "--from", "axiomatic-mcp", "all"]
        self.env = env or {
            "AXIOMATIC_API_KEY": os.getenv(
                "AXIOMATIC_API_KEY",
                "3bfacdcf-0492-4fad-8af0-1d5fa34bcbf6"
            )
        }
        
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.pending_requests: Dict[int, Queue] = {}
        self.response_thread: Optional[threading.Thread] = None
        self.initialized = False
        
    def _get_next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    def _start_server(self) -> None:
        """Start the MCP server process."""
        if self.process is not None:
            return  # Already started
        
        print("  Debug: Starting MCP server via subprocess...")
        self.process = subprocess.Popen(
            [self.server_command] + self.server_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered
            env={**os.environ, **self.env}
        )
        
        # Start response reader thread
        self.response_thread = threading.Thread(
            target=self._read_responses,
            daemon=True
        )
        self.response_thread.start()
        
        # Give server a moment to start
        time.sleep(1.0)
        print("  Debug: Server process started")
    
    def _read_responses(self) -> None:
        """Read responses from server stdout in background thread."""
        if self.process is None or self.process.stdout is None:
            return
        
        try:
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break  # Process ended
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    response = json.loads(line)
                    request_id = response.get("id")
                    
                    if request_id and request_id in self.pending_requests:
                        self.pending_requests[request_id].put(response)
                    else:
                        # Notification or response without matching request
                        print(f"  Debug: Unmatched response: {response}")
                except json.JSONDecodeError:
                    # Non-JSON output (e.g., server logs to stdout)
                    if "INFO" in line or "DEBUG" in line or "ERROR" in line:
                        print(f"  Debug: Server log: {line}")
                    # Ignore non-JSON lines
        except Exception as e:
            print(f"  Debug: Error reading responses: {e}")
    
    def _send_request(self, method: str, params: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """
        Send JSON-RPC request and wait for response.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
            timeout: Timeout in seconds
            
        Returns:
            Response dictionary
        """
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Server process not started")
        
        request_id = self._get_next_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params
        }
        
        # Create queue for response
        response_queue = Queue()
        self.pending_requests[request_id] = response_queue
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            print(f"  Debug: Sent {method} request (id={request_id})")
            
            # Wait for response
            try:
                response = response_queue.get(timeout=timeout)
                
                # Check for errors
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(
                        f"MCP {method} error: {error.get('message', 'Unknown error')} "
                        f"(code: {error.get('code', 'unknown')})"
                    )
                
                return response.get("result", {})
                
            except Empty:
                raise RuntimeError(f"MCP {method} request timed out after {timeout}s")
                
        finally:
            # Clean up
            self.pending_requests.pop(request_id, None)
    
    def initialize(self) -> None:
        """Initialize MCP session."""
        if self.initialized:
            return
        
        self._start_server()
        
        print("  Debug: Sending initialize request...")
        result = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ppc-agent",
                    "version": "1.0.0"
                }
            },
            timeout=30.0
        )
        
        print(f"  Debug: Initialize result: {result}")
        
        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        if self.process and self.process.stdin:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        
        self.initialized = True
        print("  Debug: MCP session initialized successfully")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        if not self.initialized:
            self.initialize()
        
        print("  Debug: Listing tools...")
        result = self._send_request("tools/list", {}, timeout=10.0)
        tools = result.get("tools", [])
        print(f"  Debug: Found {len(tools)} tools")
        return tools
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not self.initialized:
            self.initialize()
        
        print(f"  Debug: Calling tool: {tool_name}")
        result = self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments
            },
            timeout=120.0  # Tools may take longer
        )
        return result
    
    def annotate_file(self, file_path: str, query: str) -> str:
        """
        Annotate a file using AxDocumentAnnotator tool.
        
        Args:
            file_path: Absolute path to file
            query: Annotation query/instructions
            
        Returns:
            Annotation result as string
        """
        # Ensure absolute path
        file_path = str(Path(file_path).resolve())
        
        try:
            # Initialize if needed
            if not self.initialized:
                self.initialize()
            
            # Find the annotate_file tool (must be exact match, not report_feedback)
            tools = self.list_tools()
            tool_name = None
            
            # Priority 1: Exact match for AxDocumentAnnotator_annotate_file
            for tool in tools:
                name = tool.get("name", "")
                if name == "AxDocumentAnnotator_annotate_file":
                    tool_name = name
                    print(f"  Debug: Found exact match: {tool_name}")
                    break
            
            # Priority 2: Contains "annotate_file" but NOT "report_feedback" or "report"
            if not tool_name:
                for tool in tools:
                    name = tool.get("name", "")
                    name_lower = name.lower()
                    # Must have "annotate_file" and must NOT have "report"
                    if "annotate_file" in name_lower and "report" not in name_lower:
                        tool_name = name
                        print(f"  Debug: Found by pattern match: {tool_name}")
                        break
            
            if not tool_name:
                available_tools = [t.get("name", "unknown") for t in tools]
                raise RuntimeError(
                    f"Could not find AxDocumentAnnotator tool. "
                    f"Available tools: {available_tools}"
                )
            
            # Call the tool
            result = self.call_tool(tool_name, {
                "file_path": file_path,
                "query": query
            })
            
            # Extract text from result
            if isinstance(result, dict):
                if "content" in result:
                    if isinstance(result["content"], list):
                        texts = []
                        for item in result["content"]:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                        return "\n\n".join(texts)
                    elif isinstance(result["content"], str):
                        return result["content"]
                elif "text" in result:
                    return result["text"]
                return str(result)
            elif isinstance(result, str):
                return result
            else:
                return str(result)
                
        except Exception as e:
            raise RuntimeError(f"Failed to annotate file: {e}")

    def parse_pdf_to_markdown(self, file_path: str) -> str:
        """
        Parse a PDF into structured markdown using AxDocumentParser tool.
        
        This is intended for Stage 0 preprocessing: convert the PDF to markdown,
        then feed that markdown content into the LLM for section identification.
        
        Args:
            file_path: Absolute path to the PDF file
            
        Returns:
            Markdown string representing the parsed PDF content
        """
        file_path = str(Path(file_path).resolve())
        
        try:
            if not self.initialized:
                self.initialize()
            
            tools = self.list_tools()
            tool_name = None
            
            # Priority 1: exact Axiomatic name
            for tool in tools:
                name = tool.get("name", "")
                if name == "AxDocumentParser_parse_pdf_to_md":
                    tool_name = name
                    print(f"  Debug: Found AxDocumentParser tool: {tool_name}")
                    break
            
            # Priority 2: pattern-based lookup
            if not tool_name:
                for tool in tools:
                    name = tool.get("name", "")
                    name_lower = name.lower()
                    if "documentparser" in name_lower or "parse_pdf_to_md" in name_lower:
                        tool_name = name
                        print(f"  Debug: Found AxDocumentParser by pattern: {tool_name}")
                        break
            
            if not tool_name:
                available = [t.get("name", "unknown") for t in tools]
                raise RuntimeError(
                    f"Could not find AxDocumentParser tool. Available tools: {available}"
                )
            
            result = self.call_tool(
                tool_name,
                {"file_path": file_path}
            )
            
            # Extract text result first
            text_result = ""
            if isinstance(result, dict):
                if "content" in result:
                    if isinstance(result["content"], list):
                        texts = []
                        for item in result["content"]:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, str):
                                texts.append(item)
                        text_result = "\n\n".join(texts)
                    elif isinstance(result["content"], str):
                        text_result = result["content"]
                elif "text" in result:
                    text_result = result["text"]
                else:
                    text_result = str(result)
            elif isinstance(result, str):
                text_result = result
            else:
                text_result = str(result)

            # Check if result is a file path message
            if "Generated markdown at:" in text_result:
                print(f"  Debug: Detected file path in result: {text_result[:100]}...")
                try:
                    # Extract path - assuming format "Generated markdown at: /path/to/file.md"
                    # It might be followed by a newline and other text
                    for line in text_result.split('\n'):
                        if "Generated markdown at:" in line:
                            md_path = line.split("Generated markdown at:")[1].strip()
                            if os.path.exists(md_path):
                                print(f"  Debug: Reading content from {md_path}")
                                with open(md_path, 'r', encoding='utf-8') as f:
                                    return f.read()
                            else:
                                print(f"  Warning: Generated markdown file not found at {md_path}")
                except Exception as e:
                    print(f"  Warning: Failed to read generated markdown file: {e}")
            
            # Fallback: return the text result directly
            return text_result
        
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF with AxDocumentParser: {e}")
    
    def close(self) -> None:
        """Close the MCP client and stop the server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            finally:
                self.process = None
                self.initialized = False


# Import os for environment variables
import os


def create_subprocess_annotator_wrapper() -> Optional[Callable]:
    """
    Create a synchronous wrapper for subprocess-based MCP annotator.
    
    Returns:
        Callable function (file_path, query) -> str, or None if unavailable
    """
    try:
        client = SubprocessMCPClient()
        
        def annotate_file_sync(file_path: str, query: str) -> str:
            """Synchronous wrapper for annotate_file."""
            try:
                result = client.annotate_file(file_path, query)
                return result
            finally:
                client.close()
        
        return annotate_file_sync
    except Exception as e:
        print(f"  Debug: Failed to create subprocess MCP client: {e}")
        return None


if __name__ == "__main__":
    # Test the subprocess client
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mcp_client_subprocess.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    query = "Extract all sections about photonic components and architectures"
    
    client = SubprocessMCPClient()
    try:
        result = client.annotate_file(pdf_path, query)
        print(f"\nAnnotation result ({len(result)} chars):")
        print(result[:500] + "..." if len(result) > 500 else result)
    finally:
        client.close()
