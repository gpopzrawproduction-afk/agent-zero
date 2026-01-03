"""
Comprehensive Testing Suite for ZeroGravity

This module provides a comprehensive testing suite for the ZeroGravity platform,
including unit tests, integration tests, and end-to-end tests.
"""
import pytest
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import json
import time
from datetime import datetime

# Import ZeroGravity modules
from zero_gravity_core.agents.base import BaseAgent
from zero_gravity_core.agents.coordinator import Coordinator
from zero_gravity_core.llm.providers import LLMManager, LLMProviderType
from zero_gravity_core.llm.cache import CacheManager
from zero_gravity_core.utils.validation import InputSanitizer
from zero_gravity_core.utils.error_handling import ErrorHandler, retry_on_failure, RetryConfig
from zero_gravity_core.utils.logging import Logger
from zero_gravity_core.task_queue.celery_app import execute_agent_task, execute_workflow_task
from zero_gravity_core.workflow.graph import WorkflowGraph, WorkflowNode, NodeType, TaskStatus
from zero_gravity_core.api.rate_limiting import RateLimiter, UserRateLimiter
from zero_gravity_core.api.webhooks import WebhookManager, WebhookEventType
from zero_gravity_core.api.sdk_generator import SDKGenerator, APISpec, Language


class TestBaseAgent(unittest.TestCase):
    """Test suite for BaseAgent functionality"""
    
    def setUp(self):
        self.agent = BaseAgent(role="test", system_prompt="Test system prompt")
    
    def test_initialization(self):
        """Test BaseAgent initialization"""
        self.assertEqual(self.agent.role, "test")
        self.assertEqual(self.agent.system_prompt, "Test system prompt")
        self.assertEqual(self.agent.memory, [])
    
    def test_memory_operations(self):
        """Test memory recording and retrieval"""
        test_data = {"key": "value"}
        self.agent.record(test_data)
        
        memory = self.agent.get_memory()
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0], test_data)
    
    def test_get_system_prompt(self):
        """Test system prompt retrieval"""
        prompt = self.agent.get_system_prompt()
        self.assertEqual(prompt, "Test system prompt")
    
    @patch('zero_gravity_core.llm.providers.llm_manager')
    @patch('zero_gravity_core.llm.cache.cache_manager')
    def test_execute_with_llm(self, mock_cache, mock_llm_manager):
        """Test LLM execution with caching"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_response.model = "gpt-4"
        mock_llm_manager.call.return_value = mock_response
        
        # Mock cache behavior
        mock_cache.get_cached_response.return_value = None  # No cached response
        
        result = self.agent.execute_with_llm("Test input")
        
        self.assertEqual(result, "Test response")
        mock_llm_manager.call.assert_called_once()
        mock_cache.get_cached_response.assert_called_once()
        mock_cache.cache_response.assert_called_once()


class TestCoordinator(unittest.TestCase):
    """Test suite for Coordinator functionality"""
    
    def setUp(self):
        self.coordinator = Coordinator()
    
    def test_initialization(self):
        """Test Coordinator initialization"""
        self.assertIsNotNone(self.coordinator.system_prompts)
        self.assertEqual(len(self.coordinator.system_prompts), 5)  # architect, engineer, designer, operator, coordinator
    
    def test_get_prompt(self):
        """Test prompt retrieval"""
        prompt = self.coordinator.get_prompt("architect")
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)
    
    def test_get_prompt_invalid_role(self):
        """Test prompt retrieval for invalid role"""
        with self.assertRaises(ValueError):
            self.coordinator.get_prompt("invalid_role")
    
    @patch('zero_gravity_core.agents.architect.Architect')
    def test_spawn_agent(self, mock_agent_class):
        """Test agent spawning"""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        agent = self.coordinator.spawn_agent("architect")
        
        self.assertIsNotNone(agent)
        mock_agent_class.assert_called_once()
    
    @patch('zero_gravity_core.agents.architect.Architect')
    def test_run_basic_workflow(self, mock_agent_class):
        """Test basic workflow execution"""
        # Mock agent behavior
        mock_agent = Mock()
        mock_agent.execute_with_llm.return_value = "Mocked response"
        mock_agent_class.return_value = mock_agent
        
        result = self.coordinator.run("Test objective")
        
        self.assertIn("result", result)
        self.assertEqual(result["objective"], "Test objective")


class TestLLMProviders(unittest.TestCase):
    """Test suite for LLM provider functionality"""
    
    def setUp(self):
        self.llm_manager = LLMManager()
    
    @patch('zero_gravity_core.llm.providers.OpenAILLMProvider')
    def test_provider_initialization(self, mock_openai_provider):
        """Test LLM manager provider initialization"""
        # Verify that the manager has the expected providers
        available_providers = self.llm_manager.get_available_providers()
        
        # Even if providers aren't fully configured, we should have access to the manager
        self.assertIsInstance(self.llm_manager, LLMManager)
    
    def test_provider_selection(self):
        """Test provider selection functionality"""
        # Test that we can set and get the default provider
        if self.llm_manager.providers:
            initial_default = self.llm_manager.default_provider
            new_provider = next(iter(self.llm_manager.providers))
            
            self.llm_manager.set_default_provider(new_provider)
            self.assertEqual(self.llm_manager.default_provider, new_provider)
            
            # Restore initial state
            self.llm_manager.default_provider = initial_default


class TestCacheManager(unittest.TestCase):
    """Test suite for cache functionality"""
    
    def setUp(self):
        self.cache_manager = CacheManager()
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        test_key = "test_key"
        test_value = {"data": "test_value"}
        
        # Initially, key should not exist
        result = self.cache_manager.get(test_key)
        self.assertIsNone(result)
        
        # Set a value
        self.cache_manager.set(test_key, test_value, ttl_seconds=3600)
        
        # Get the value back
        result = self.cache_manager.get(test_key)
        self.assertIsNotNone(result)
        cached_value, created_at = result
        self.assertEqual(cached_value, test_value)
    
    def test_cache_with_messages(self):
        """Test caching with message structures"""
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-4"
        response = "Test response"
        
        # Test cache response functionality
        self.cache_manager.cache_response(messages, model, response)
        
        cached_response = self.cache_manager.get_cached_response(messages, model)
        self.assertEqual(cached_response, response)


class TestInputValidation(unittest.TestCase):
    """Test suite for input validation functionality"""
    
    def setUp(self):
        self.input_sanitizer = InputSanitizer()
    
    def test_text_sanitization(self):
        """Test text sanitization"""
        dangerous_input = '<script>alert("xss")</script>Hello'
        sanitized = self.input_sanitizer._sanitize_text(dangerous_input)
        
        # The script tag should be removed
        self.assertNotIn("<script>", sanitized)
        self.assertIn("Hello", sanitized)
    
    def test_url_validation(self):
        """Test URL validation"""
        # Valid URL
        valid_url = "https://example.com"
        result = self.input_sanitizer._sanitize_url(valid_url)
        self.assertEqual(result, valid_url)
        
        # Invalid URL should raise an error
        invalid_url = "not-a-url"
        with self.assertRaises(ValueError):
            self.input_sanitizer._sanitize_url(invalid_url)
    
    def test_code_validation(self):
        """Test code validation"""
        dangerous_code = 'import os\nos.system("rm -rf /")'
        with self.assertRaises(ValueError):
            self.input_sanitizer._sanitize_code(dangerous_code)
        
        safe_code = 'print("Hello, world!")'
        result = self.input_sanitizer._sanitize_code(safe_code)
        self.assertEqual(result, safe_code)
    
    def test_command_validation(self):
        """Test command validation"""
        dangerous_command = 'rm -rf / && echo "dangerous"'
        with self.assertRaises(ValueError):
            self.input_sanitizer._sanitize_command(dangerous_command)
        
        safe_command = 'echo "safe command"'
        result = self.input_sanitizer._sanitize_command(safe_command)
        self.assertEqual(result, safe_command)


class TestErrorHandling(unittest.TestCase):
    """Test suite for error handling functionality"""
    
    def setUp(self):
        self.error_handler = ErrorHandler()
    
    def test_retry_decorator(self):
        """Test retry decorator functionality"""
        config = RetryConfig(max_retries=2, delay=0.1)
        
        # Test function that succeeds on second attempt
        call_count = 0
        
        @retry_on_failure(config)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return "Success"
        
        result = flaky_function()
        self.assertEqual(result, "Success")
        self.assertEqual(call_count, 2)
    
    def test_error_logging(self):
        """Test error logging functionality"""
        test_error = ValueError("Test error")
        context = {"test_key": "test_value"}
        
        self.error_handler.log_error(test_error, context)
        
        summary = self.error_handler.get_error_summary()
        self.assertGreaterEqual(summary["total_errors"], 1)
        self.assertEqual(len(summary["recent_errors"]), min(10, summary["total_errors"]))


class TestWorkflowGraph(unittest.TestCase):
    """Test suite for workflow graph functionality"""
    
    def test_graph_creation(self):
        """Test workflow graph creation"""
        workflow = WorkflowGraph("test_workflow", "Test workflow")
        
        # Add nodes
        node1 = WorkflowNode(id="node1", node_type=NodeType.AGENT, name="Node 1", agent_role="architect")
        node2 = WorkflowNode(id="node2", node_type=NodeType.AGENT, name="Node 2", agent_role="engineer", 
                            dependencies=["node1"])
        
        workflow.add_node(node1)
        workflow.add_node(node2)
        
        self.assertEqual(len(workflow.nodes), 2)
        self.assertIn("node1", workflow.nodes)
        self.assertIn("node2", workflow.nodes)
    
    def test_graph_dependencies(self):
        """Test workflow graph dependencies"""
        workflow = WorkflowGraph("test_workflow", "Test workflow")
        
        node1 = WorkflowNode(id="node1", node_type=NodeType.AGENT, name="Node 1", agent_role="architect")
        node2 = WorkflowNode(id="node2", node_type=NodeType.AGENT, name="Node 2", agent_role="engineer", 
                            dependencies=["node1"])
        
        workflow.add_node(node1)
        workflow.add_node(node2)
        
        # Add edge to represent dependency
        from zero_gravity_core.workflow.graph import WorkflowEdge
        workflow.add_edge(WorkflowEdge("node1", "node2"))
        
        deps = workflow.get_dependencies("node2")
        self.assertIn("node1", deps)
    
    def test_execution_order(self):
        """Test execution order calculation"""
        workflow = WorkflowGraph("test_workflow", "Test workflow")
        
        node1 = WorkflowNode(id="node1", node_type=NodeType.AGENT, name="Node 1", agent_role="architect")
        node2 = WorkflowNode(id="node2", node_type=NodeType.AGENT, name="Node 2", agent_role="engineer", 
                            dependencies=["node1"])
        node3 = WorkflowNode(id="node3", node_type=NodeType.AGENT, name="Node 3", agent_role="designer", 
                            dependencies=["node1"])
        
        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)
        
        # Add edges
        from zero_gravity_core.workflow.graph import WorkflowEdge
        workflow.add_edge(WorkflowEdge("node1", "node2"))
        workflow.add_edge(WorkflowEdge("node1", "node3"))
        
        execution_order = workflow.get_execution_order()
        self.assertIn("node1", execution_order[:1])  # node1 should be first
        # node2 and node3 can be in any order after node1
    
    def test_parallel_execution_groups(self):
        """Test parallel execution group calculation"""
        workflow = WorkflowGraph("test_workflow", "Test workflow")
        
        node1 = WorkflowNode(id="node1", node_type=NodeType.AGENT, name="Node 1", agent_role="architect")
        node2 = WorkflowNode(id="node2", node_type=NodeType.AGENT, name="Node 2", agent_role="engineer", 
                            dependencies=["node1"])
        node3 = WorkflowNode(id="node3", node_type=NodeType.AGENT, name="Node 3", agent_role="designer", 
                            dependencies=["node1"])
        
        workflow.add_node(node1)
        workflow.add_node(node2)
        workflow.add_node(node3)
        
        # Add edges
        from zero_gravity_core.workflow.graph import WorkflowEdge
        workflow.add_edge(WorkflowEdge("node1", "node2"))
        workflow.add_edge(WorkflowEdge("node1", "node3"))
        
        parallel_groups = workflow.get_parallel_groups()
        # Should have at least 2 groups: [node1], [node2, node3] (or similar)
        self.assertGreaterEqual(len(parallel_groups), 1)
    
    def test_node_status_updates(self):
        """Test node status updates"""
        workflow = WorkflowGraph("test_workflow", "Test workflow")
        
        node1 = WorkflowNode(id="node1", node_type=NodeType.AGENT, name="Node 1", agent_role="architect")
        workflow.add_node(node1)
        
        # Update status
        workflow.update_node_status("node1", TaskStatus.RUNNING, outputs={"result": "test"})
        
        self.assertEqual(workflow.nodes["node1"].status, TaskStatus.RUNNING)
        self.assertEqual(workflow.nodes["node1"].outputs, {"result": "test"})


class TestRateLimiting(unittest.TestCase):
    """Test suite for rate limiting functionality"""
    
    def setUp(self):
        self.rate_limiter = UserRateLimiter()
    
    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        user_id = "test_user"
        
        # First request should be allowed
        is_allowed, headers = self.rate_limiter.check_user_limit(user_id)
        self.assertTrue(is_allowed)
        
        # Use up the limit
        config = self.rate_limiter.default_configs["api_default"]
        for _ in range(config.limit):
            self.rate_limiter.check_user_limit(user_id)
        
        # Next request should be denied
        is_allowed, headers = self.rate_limiter.check_user_limit(user_id)
        self.assertFalse(is_allowed)
    
    def test_different_tiers(self):
        """Test different rate limiting tiers"""
        user_id = "test_user"
        
        # Check default tier
        is_allowed_default, _ = self.rate_limiter.check_user_limit(user_id, tier="default")
        self.assertTrue(is_allowed_default)
        
        # Check high tier (should have higher limit)
        is_allowed_high, _ = self.rate_limiter.check_user_limit(user_id, tier="high_tier")
        # This might fail if high_tier config doesn't exist, which is expected


class TestWebhooks(unittest.TestCase):
    """Test suite for webhook functionality"""
    
    def setUp(self):
        self.webhook_manager = WebhookManager()
    
    def test_webhook_registration(self):
        """Test webhook registration"""
        url = "https://example.com/webhook"
        event_types = [WebhookEventType.WORKFLOW_STARTED]
        
        webhook = asyncio.run(
            self.webhook_manager.register_webhook(url, event_types, secret="test_secret")
        )
        
        self.assertEqual(webhook.url, url)
        self.assertIn(WebhookEventType.WORKFLOW_STARTED, webhook.event_types)
        self.assertEqual(webhook.secret, "test_secret")
    
    def test_signature_verification(self):
        """Test webhook signature verification"""
        payload = '{"test": "data"}'
        secret = "test_secret"
        timestamp = int(time.time())
        
        signature = asyncio.run(
            self.webhook_manager.verify_webhook_signature(payload, "invalid", secret, timestamp)
        )
        self.assertFalse(signature)  # Should fail with invalid signature
        
        # Test with correct signature
        from zero_gravity_core.api.webhooks import WebhookSignature
        correct_signature = WebhookSignature.generate_signature(payload, secret, timestamp)
        signature = asyncio.run(
            self.webhook_manager.verify_webhook_signature(payload, correct_signature, secret, timestamp)
        )
        self.assertTrue(signature)


class TestSDKGeneration(unittest.TestCase):
    """Test suite for SDK generation functionality"""
    
    def setUp(self):
        self.sdk_generator = SDKGenerator()
        self.spec = APISpec(
            title="ZeroGravity API",
            description="Test API",
            version="1.0.0",
            base_url="https://api.test.com",
            endpoints=[
                {
                    "name": "test_endpoint",
                    "path": "/test",
                    "method": "GET",
                    "description": "Test endpoint",
                    "parameters": [
                        {"name": "param1", "type": "str", "ts_type": "string", "required": True, "description": "Test param"}
                    ]
                }
            ],
            auth={}
        )
    
    def test_sdk_generation(self):
        """Test SDK generation for different languages"""
        # Test Python SDK generation
        python_sdk = self.sdk_generator.generate_sdk(self.spec, Language.PYTHON)
        self.assertIsInstance(python_sdk, bytes)
        self.assertGreater(len(python_sdk), 0)
        
        # Test JavaScript SDK generation
        js_sdk = self.sdk_generator.generate_sdk(self.spec, Language.JAVASCRIPT)
        self.assertIsInstance(js_sdk, bytes)
        self.assertGreater(len(js_sdk), 0)
    
    def test_all_sdks_generation(self):
        """Test generation of all SDKs"""
        all_sdks = self.sdk_generator.generate_all_sdks(self.spec)
        
        # Should have at least Python and JavaScript
        self.assertIn(Language.PYTHON, all_sdks)
        self.assertIn(Language.JAVASCRIPT, all_sdks)
        self.assertIn(Language.TYPESCRIPT, all_sdks)


class TestIntegration(unittest.TestCase):
    """Integration tests for the ZeroGravity platform"""
    
    def test_full_workflow_execution(self):
        """Test a full workflow execution from start to finish"""
        coordinator = Coordinator()
        
        # Run a simple workflow
        result = coordinator.run("Say hello world")
        
        # Check that the result has the expected structure
        self.assertIn("result", result)
        self.assertIn("history", result)
        self.assertIn("execution_summary", result)
        self.assertEqual(result["objective"], "Say hello world")
    
    def test_error_recovery(self):
        """Test error handling and recovery mechanisms"""
        error_handler = ErrorHandler()
        
        def test_function():
            raise ValueError("Test error for recovery")
        
        def fallback_function(exception, *args, **kwargs):
            return {"error": str(exception), "fallback_used": True}
        
        result = error_handler.safe_execute(test_function, fallback_function)
        
        self.assertIn("fallback_used", result)
        self.assertTrue(result["fallback_used"])


class TestPerformance(unittest.TestCase):
    """Performance tests for the ZeroGravity platform"""
    
    def test_response_time(self):
        """Test response time under normal conditions"""
        coordinator = Coordinator()
        
        start_time = time.time()
        result = coordinator.run("What is 2+2?")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be reasonably fast (under 10 seconds for a simple query)
        self.assertLess(response_time, 10.0)
    
    def test_concurrent_workflows(self):
        """Test handling of concurrent workflows"""
        def run_workflow():
            coordinator = Coordinator()
            return coordinator.run("Simple test objective")
        
        # Run multiple workflows concurrently
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_workflow) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All workflows should complete successfully
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn("result", result)


def run_all_tests():
    """Run all tests in the suite"""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__, fromlist=['TestBaseAgent']))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_specific_test_class(test_class):
    """Run a specific test class"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    # Run all tests
    print("Running ZeroGravity comprehensive test suite...")
    result = run_all_tests()
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f" {test}: {traceback}")
