"""Test script to diagnose MCP server startup issues."""

import asyncio
import subprocess
import sys
from pathlib import Path

async def test_server_command():
    """Test if the server command works and what it outputs."""
    print("=" * 60)
    print("Test 1: Testing server command directly")
    print("=" * 60)
    
    server_command = "/home/tofu8/.local/bin/uvx"
    server_args = ["--quiet", "--from", "axiomatic-mcp", "all"]
    
    print(f"Command: {server_command}")
    print(f"Args: {server_args}")
    print("\nStarting server process (will timeout after 5 seconds)...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [server_command] + server_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                "AXIOMATIC_API_KEY": "3bfacdcf-0492-4fad-8af0-1d5fa34bcbf6"
            }
        )
        
        # Wait a bit and then check if it's still running
        await asyncio.sleep(2)
        
        if process.poll() is None:
            print("✓ Server process is running (not exited)")
        else:
            print(f"✗ Server process exited with code: {process.returncode}")
            stdout, stderr = process.communicate()
            print(f"  stdout: {stdout[:200]}")
            print(f"  stderr: {stderr[:200]}")
            return
        
        # Try to read from stdout (non-blocking)
        print("\nTrying to read from stdout...")
        try:
            # Set a timeout for reading
            stdout_data = await asyncio.wait_for(
                asyncio.to_thread(lambda: process.stdout.read(100) if process.stdout else ""),
                timeout=2.0
            )
            if stdout_data:
                print(f"  Read from stdout: {stdout_data[:200]}")
            else:
                print("  No data from stdout yet")
        except asyncio.TimeoutError:
            print("  Timeout reading from stdout (server may be waiting for input)")
        
        # Check stderr
        print("\nChecking stderr...")
        try:
            stderr_data = await asyncio.wait_for(
                asyncio.to_thread(lambda: process.stderr.read(100) if process.stderr else ""),
                timeout=2.0
            )
            if stderr_data:
                print(f"  Read from stderr: {stderr_data[:200]}")
            else:
                print("  No data from stderr")
        except asyncio.TimeoutError:
            print("  Timeout reading from stderr")
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        
        print("\n✓ Server command test completed")
        
    except Exception as e:
        print(f"✗ Error testing server command: {e}")
        import traceback
        traceback.print_exc()


async def test_mcp_sdk_connection():
    """Test MCP SDK connection with detailed logging."""
    print("\n" + "=" * 60)
    print("Test 2: Testing MCP SDK connection")
    print("=" * 60)
    
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        print("✓ MCP SDK imported")
    except ImportError as e:
        print(f"✗ Cannot import MCP SDK: {e}")
        return
    
    server_params = StdioServerParameters(
        command="/home/tofu8/.local/bin/uvx",
        args=["--quiet", "--from", "axiomatic-mcp", "all"],
        env={
            "AXIOMATIC_API_KEY": "3bfacdcf-0492-4fad-8af0-1d5fa34bcbf6"
        }
    )
    
    print("Attempting to connect via stdio_client...")
    try:
        async with stdio_client(server_params) as (read, write):
            print("✓ Entered stdio_client context")
            print("  Creating ClientSession...")
            session = ClientSession(read, write)
            print("✓ ClientSession created")
            
            print("  Waiting 2 seconds for server to start...")
            await asyncio.sleep(2.0)
            
            print("  Calling session.initialize() with 10s timeout...")
            try:
                init_result = await asyncio.wait_for(
                    session.initialize(),
                    timeout=10.0
                )
                print("✓ Session initialized successfully!")
                print(f"  Protocol version: {init_result.protocol_version}")
                print(f"  Server info: {init_result.server_info}")
                
                # Try listing tools
                print("\n  Listing tools...")
                tools = await session.list_tools()
                print(f"✓ Found {len(tools.tools)} tools")
                for tool in tools.tools[:5]:  # Show first 5
                    print(f"    - {tool.name}")
                
            except asyncio.TimeoutError:
                print("✗ Session initialization timed out after 10 seconds")
                print("  This suggests the server is not responding to initialization requests")
            except Exception as e:
                print(f"✗ Error during initialization: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"✗ Error with stdio_client: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all diagnostic tests."""
    await test_server_command()
    await test_mcp_sdk_connection()
    
    print("\n" + "=" * 60)
    print("Diagnostic tests completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

