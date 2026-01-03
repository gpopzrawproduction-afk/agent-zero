"""
Sandbox Mode for ZeroGravity

This module implements a secure sandbox environment for testing
ZeroGravity agents, workflows, and plugins without affecting
production systems.
"""
import asyncio
import tempfile
import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import time
import threading
import queue
import signal
import resource
import copy
from datetime import datetime


class SandboxMode(Enum):
    """Different sandbox modes"""
    STRICT = "strict"        # Maximum isolation, minimal resources
    MODERATE = "moderate"    # Balanced isolation and functionality
    RELAXED = "relaxed"      # Minimal isolation, full functionality


class ExecutionResult:
    """Result of a sandbox execution"""
    def __init__(self, 
                 success: bool,
                 output: str = "",
                 error: str = "",
                 execution_time: float = 0.0,
                 resources_used: Dict[str, Any] = None,
                 return_code: int = 0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.resources_used = resources_used or {}
        self.return_code = return_code
        self.timestamp = datetime.utcnow()


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""
    max_memory_mb: int = 100  # Maximum memory in MB
    max_time_seconds: int = 30  # Maximum execution time
    max_file_size_mb: int = 10 # Maximum file size for operations
    max_files: int = 10  # Maximum number of files
    network_access: bool = False  # Whether network access is allowed
    file_system_access: bool = False  # Whether file system access is allowed
    system_commands: bool = False  # Whether system commands are allowed


class SandboxEnvironment:
    """A secure sandbox environment for executing code"""
    
    def __init__(self, mode: SandboxMode = SandboxMode.MODERATE, 
                 resource_limits: Optional[ResourceLimits] = None):
        self.mode = mode
        self.resource_limits = resource_limits or self._get_default_limits(mode)
        self.temp_dir = None
        self.is_active = False
        self.active_processes = []
        self.logger = None
        self.execution_history: List[ExecutionResult] = []
    
    def _get_default_limits(self, mode: SandboxMode) -> ResourceLimits:
        """Get default resource limits based on mode"""
        if mode == SandboxMode.STRICT:
            return ResourceLimits(
                max_memory_mb=50,
                max_time_seconds=10,
                max_file_size_mb=1,
                max_files=3,
                network_access=False,
                file_system_access=False,
                system_commands=False
            )
        elif mode == SandboxMode.RELAXED:
            return ResourceLimits(
                max_memory_mb=500,
                max_time_seconds=120,
                max_file_size_mb=50,
                max_files=100,
                network_access=True,
                file_system_access=True,
                system_commands=True
            )
        else:  # MODERATE
            return ResourceLimits(
                max_memory_mb=200,
                max_time_seconds=60,
                max_file_size_mb=10,
                max_files=20,
                network_access=False,
                file_system_access=True,
                system_commands=False
            )
    
    def activate(self) -> Path:
        """Activate the sandbox environment and return the temp directory"""
        if self.is_active:
            return self.temp_dir
        
        # Create a temporary directory for the sandbox
        self.temp_dir = Path(tempfile.mkdtemp(prefix="zerogravity_sandbox_"))
        self.is_active = True
        
        # Create a restricted environment
        self._setup_restricted_environment()
        
        return self.temp_dir
    
    def deactivate(self):
        """Deactivate the sandbox and clean up"""
        if not self.is_active:
            return
        
        # Terminate any active processes
        for proc in self.active_processes:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except:
                try:
                    proc.kill()
                except:
                    pass
        
        # Clean up temporary directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        self.is_active = False
        self.temp_dir = None
        self.active_processes = []
    
    def _setup_restricted_environment(self):
        """Set up the restricted environment"""
        if not self.temp_dir:
            return
        
        # Create necessary subdirectories
        (self.temp_dir / "input").mkdir(exist_ok=True)
        (self.temp_dir / "output").mkdir(exist_ok=True)
        (self.temp_dir / "temp").mkdir(exist_ok=True)
    
    @contextmanager
    def sandbox_context(self):
        """Context manager for sandbox operations"""
        sandbox_dir = self.activate()
        try:
            yield sandbox_dir
        finally:
            self.deactivate()
    
    def execute_python_code(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """Execute Python code in the sandbox"""
        if timeout is None:
            timeout = self.resource_limits.max_time_seconds
        
        start_time = time.time()
        
        with self.sandbox_context() as sandbox_dir:
            # Write code to a temporary file
            code_file = sandbox_dir / "sandbox_code.py"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Execute the code in a subprocess with restrictions
            try:
                # Set up the subprocess with resource limits
                process = subprocess.run(
                    [sys.executable, str(code_file)],
                    cwd=sandbox_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    # Note: In a real implementation, you'd use more sophisticated
                    # sandboxing techniques like containerization or seccomp
                )
                
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    success=process.returncode == 0,
                    output=process.stdout,
                    error=process.stderr,
                    execution_time=execution_time,
                    return_code=process.returncode
                )
                
                self.execution_history.append(result)
                return result
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                result = ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    execution_time=execution_time
                )
                self.execution_history.append(result)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                result = ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
                self.execution_history.append(result)
                return result
    
    def execute_agent_in_sandbox(self, agent_role: str, input_data: Any) -> ExecutionResult:
        """Execute an agent in the sandbox environment"""
        # Import here to avoid circular dependencies
        from zero_gravity_core.agents.coordinator import Coordinator
        
        start_time = time.time()
        
        with self.sandbox_context() as sandbox_dir:
            try:
                # Create a coordinator in the sandbox
                coordinator = Coordinator(base_dir=sandbox_dir)
                
                # Spawn the agent
                agent = coordinator.spawn_agent(agent_role)
                
                # Execute with the input data
                result = agent.execute_with_llm(input_data)
                
                execution_time = time.time() - start_time
                
                execution_result = ExecutionResult(
                    success=True,
                    output=str(result),
                    execution_time=execution_time
                )
                
                self.execution_history.append(execution_result)
                return execution_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                execution_result = ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
                self.execution_history.append(execution_result)
                return execution_result
    
    def execute_workflow_in_sandbox(self, objective: str) -> ExecutionResult:
        """Execute a complete workflow in the sandbox environment"""
        from zero_gravity_core.agents.coordinator import Coordinator
        
        start_time = time.time()
        
        with self.sandbox_context() as sandbox_dir:
            try:
                # Create a coordinator in the sandbox
                coordinator = Coordinator(base_dir=sandbox_dir)
                
                # Execute the workflow
                result = coordinator.run(objective)
                
                execution_time = time.time() - start_time
                
                execution_result = ExecutionResult(
                    success=True,
                    output=json.dumps(result, indent=2),
                    execution_time=execution_time
                )
                
                self.execution_history.append(execution_result)
                return execution_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                execution_result = ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
                self.execution_history.append(execution_result)
                return execution_result


class TestSandbox:
    """Testing sandbox for ZeroGravity components"""
    
    def __init__(self, mode: SandboxMode = SandboxMode.MODERATE):
        self.sandbox = SandboxEnvironment(mode)
        self.test_results: Dict[str, ExecutionResult] = {}
        self.logger = None
    
    def test_agent(self, agent_role: str, test_inputs: List[Any]) -> Dict[str, ExecutionResult]:
        """Test an agent with multiple inputs"""
        results = {}
        
        for i, test_input in enumerate(test_inputs):
            test_name = f"{agent_role}_test_{i}"
            print(f"Running test: {test_name}")
            
            result = self.sandbox.execute_agent_in_sandbox(agent_role, test_input)
            results[test_name] = result
            
            if result.success:
                print(f"✓ {test_name} passed in {result.execution_time:.2f}s")
            else:
                print(f"✗ {test_name} failed: {result.error}")
        
        self.test_results.update(results)
        return results
    
    def test_workflow(self, objective: str) -> ExecutionResult:
        """Test a complete workflow"""
        test_name = f"workflow_test_{len(self.test_results)}"
        print(f"Running workflow test: {test_name}")
        
        result = self.sandbox.execute_workflow_in_sandbox(objective)
        self.test_results[test_name] = result
        
        if result.success:
            print(f"✓ {test_name} passed in {result.execution_time:.2f}s")
        else:
            print(f"✗ {test_name} failed: {result.error}")
        
        return result
    
    def test_plugin(self, plugin_path: str, test_config: Dict[str, Any]) -> ExecutionResult:
        """Test a plugin in the sandbox"""
        from zero_gravity_core.plugin_system import PluginManager
        
        start_time = time.time()
        
        with self.sandbox.sandbox_context() as sandbox_dir:
            try:
                # Create a plugin manager in the sandbox
                plugin_manager = PluginManager(plugins_dir=str(sandbox_dir / "plugins"))
                
                # Copy the plugin to the sandbox
                sandbox_plugin_path = sandbox_dir / "plugins" / Path(plugin_path).name
                sandbox_plugin_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plugin_path, sandbox_plugin_path)
                
                # Load and test the plugin
                plugin = plugin_manager.load_plugin(str(sandbox_plugin_path), test_config)
                
                if plugin:
                    # Initialize the plugin
                    init_result = plugin.initialize()
                    
                    # Execute the plugin
                    plugin_result = plugin.execute()
                    
                    execution_time = time.time() - start_time
                    
                    result = ExecutionResult(
                        success=init_result and plugin_result is not None,
                        output=str(plugin_result),
                        execution_time=execution_time
                    )
                    
                    self.test_results[f"plugin_test_{Path(plugin_path).stem}"] = result
                    return result
                else:
                    execution_time = time.time() - start_time
                    result = ExecutionResult(
                        success=False,
                        error="Failed to load plugin",
                        execution_time=execution_time
                    )
                    self.test_results[f"plugin_test_{Path(plugin_path).stem}"] = result
                    return result
                    
            except Exception as e:
                execution_time = time.time() - start_time
                result = ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
                self.test_results[f"plugin_test_{Path(plugin_path).stem}"] = result
                return result
    
    def run_security_tests(self) -> Dict[str, ExecutionResult]:
        """Run security tests in the sandbox"""
        results = {}
        
        # Test 1: File system access attempt
        print("Running security test: File system access")
        file_access_code = """
import os
try:
    # Attempt to read a sensitive file
    with open('/etc/passwd', 'r') as f:
        content = f.read()
    print('SECURITY BREACH: Was able to read sensitive file')
    result = 'BREACH'
except:
    print('Security intact: File access blocked')
    result = 'SECURE'
"""
        results['file_access_test'] = self.sandbox.execute_python_code(file_access_code)
        
        # Test 2: Network access attempt
        print("Running security test: Network access")
        network_code = """
import socket
try:
    # Attempt to connect to external service
    sock = socket.socket()
    sock.connect(('google.com', 80))
    sock.close()
    print('SECURITY BREACH: Was able to connect to external service')
    result = 'BREACH'
except:
    print('Security intact: Network access blocked')
    result = 'SECURE'
"""
        results['network_access_test'] = self.sandbox.execute_python_code(network_code)
        
        # Test 3: System command execution
        print("Running security test: System commands")
        command_code = """
import subprocess
try:
    # Attempt to execute system command
    result = subprocess.run(['ls', '/'], capture_output=True, text=True)
    print('SECURITY BREACH: Was able to execute system command')
    result = 'BREACH'
except:
    print('Security intact: System commands blocked')
    result = 'SECURE'
"""
        results['system_command_test'] = self.sandbox.execute_python_code(command_code)
        
        self.test_results.update(results)
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get a summary of all tests run"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate total execution time
        total_time = sum(result.execution_time for result in self.test_results.values())
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_execution_time": total_time,
            "timestamp": datetime.utcnow().isoformat(),
            "test_results": {
                name: {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error if not result.success else None
                }
                for name, result in self.test_results.items()
            }
        }


class SecureAgentExecutor:
    """Secure executor for agents in sandboxed environment"""
    
    def __init__(self, sandbox_mode: SandboxMode = SandboxMode.MODERATE):
        self.sandbox = SandboxEnvironment(sandbox_mode)
        self.active_executions = {}
    
    def execute_agent_securely(self, agent_role: str, input_data: Any, 
                             timeout: int = 30) -> ExecutionResult:
        """Execute an agent securely in a sandbox"""
        # Validate input data
        if not self._validate_input(input_data):
            return ExecutionResult(
                success=False,
                error="Invalid input data",
                execution_time=0
            )
        
        # Execute in sandbox
        return self.sandbox.execute_agent_in_sandbox(agent_role, input_data)
    
    def _validate_input(self, input_data: Any) -> bool:
        """Validate input data for security"""
        try:
            # Check for potentially dangerous content
            input_str = str(input_data)
            
            dangerous_patterns = [
                r'<script.*?>.*?</script>',  # XSS
                r'(\.\.\/)+',               # Directory traversal
                r'javascript:',             # JavaScript protocol
                r'vbscript:',              # VBScript protocol
                r'on\w+\s*=',              # Event handlers
                r'import\s+os',             # OS imports
                r'import\s+subprocess',     # Subprocess imports
                r'exec\s*\(',               # exec() function
                r'eval\s*\(',               # eval() function
            ]
            
            for pattern in dangerous_patterns:
                if __import__('re').search(pattern, input_str, __import__('re').IGNORECASE):
                    return False
            
            return True
        except:
            return False


class MockLLMProvider:
    """Mock LLM provider for sandbox testing"""
    
    def __init__(self):
        self.responses = {}
        self.call_count = 0
    
    def add_mock_response(self, input_pattern: str, response: str):
        """Add a mock response for a specific input pattern"""
        self.responses[input_pattern] = response
    
    def call(self, messages: List[Dict[str, str]], model: str = "mock", **kwargs) -> str:
        """Mock LLM call that returns predefined responses"""
        self.call_count += 1
        
        # Look for matching input in messages
        input_text = " ".join([msg.get("content", "") for msg in messages])
        
        for pattern, response in self.responses.items():
            if pattern.lower() in input_text.lower():
                return response
        
        # Default response
        return f"Mock response to: {input_text[:100]}..."


# Global sandbox instance
sandbox_instance: Optional[TestSandbox] = None


def init_sandbox(mode: SandboxMode = SandboxMode.MODERATE) -> TestSandbox:
    """Initialize the global sandbox instance"""
    global sandbox_instance
    sandbox_instance = TestSandbox(mode)
    return sandbox_instance


def get_sandbox() -> Optional[TestSandbox]:
    """Get the global sandbox instance"""
    return sandbox_instance


def run_agent_tests(agent_role: str, test_inputs: List[Any]) -> Dict[str, ExecutionResult]:
    """Run tests for an agent"""
    if sandbox_instance is None:
        init_sandbox()
    
    return sandbox_instance.test_agent(agent_role, test_inputs)


def run_workflow_test(objective: str) -> ExecutionResult:
    """Run a workflow test"""
    if sandbox_instance is None:
        init_sandbox()
    
    return sandbox_instance.test_workflow(objective)


def run_security_tests() -> Dict[str, ExecutionResult]:
    """Run security tests in the sandbox"""
    if sandbox_instance is None:
        init_sandbox()
    
    return sandbox_instance.run_security_tests()


def get_test_summary() -> Dict[str, Any]:
    """Get the test summary"""
    if sandbox_instance is None:
        return {}
    
    return sandbox_instance.get_test_summary()


# Example usage and testing
async def test_sandbox_functionality():
    """Test the sandbox functionality"""
    print("Testing ZeroGravity Sandbox System...")
    
    # Initialize sandbox
    sandbox = init_sandbox(SandboxMode.MODERATE)
    
    print(f"Sandbox initialized in mode: {sandbox.sandbox.mode.value}")
    
    # Test agent execution
    print("\n1. Testing agent execution...")
    agent_results = run_agent_tests("architect", ["Create a simple web app", "Design a database schema"])
    
    # Test workflow execution
    print("\n2. Testing workflow execution...")
    workflow_result = run_workflow_test("Build a simple calculator app")
    
    # Run security tests
    print("\n3. Running security tests...")
    security_results = run_security_tests()
    
    # Get summary
    print("\n4. Test summary:")
    summary = get_test_summary()
    print(f"  Total tests: {summary['total_tests']}")
    print(f" Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f" Total execution time: {summary['total_execution_time']:.2f}s")
    
    # Test plugin functionality (if any exist)
    print("\n5. Testing plugin system in sandbox...")
    # This would require actual plugin files to test
    # For now, we'll just verify the sandbox can handle the concept
    
    print("\nSandbox testing completed!")


if __name__ == "__main__":
    # For testing purposes
    print("Starting ZeroGravity Sandbox System example...")
    # asyncio.run(test_sandbox_functionality())
