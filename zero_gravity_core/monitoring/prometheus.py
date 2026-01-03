"""
Prometheus Monitoring System for ZeroGravity

This module implements Prometheus-based monitoring for the ZeroGravity platform,
including metrics collection, exposition, and integration with the platform's
components.
"""
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    REGISTRY, CollectorRegistry
)
from prometheus_client.multiprocess import MultiProcessCollector
import time
import functools
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import threading
import json
from pathlib import Path


# Define metrics for ZeroGravity platform
class ZeroGravityMetrics:
    """Collection of metrics for ZeroGravity platform"""
    
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'zerogravity_requests_total',
            'Total requests processed',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'zerogravity_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'zerogravity_active_requests',
            'Number of active requests'
        )
        
        # Agent execution metrics
        self.agent_executions_total = Counter(
            'zerogravity_agent_executions_total',
            'Total agent executions',
            ['agent_role']
        )
        
        self.agent_execution_duration = Histogram(
            'zerogravity_agent_execution_duration_seconds',
            'Agent execution duration in seconds',
            ['agent_role']
        )
        
        self.agent_errors_total = Counter(
            'zerogravity_agent_errors_total',
            'Total agent errors',
            ['agent_role', 'error_type']
        )
        
        # LLM call metrics
        self.llm_calls_total = Counter(
            'zerogravity_llm_calls_total',
            'Total LLM API calls',
            ['provider', 'model']
        )
        
        self.llm_call_duration = Histogram(
            'zerogravity_llm_call_duration_seconds',
            'LLM API call duration in seconds',
            ['provider', 'model']
        )
        
        self.llm_tokens_total = Counter(
            'zerogravity_llm_tokens_total',
            'Total tokens processed',
            ['provider', 'model', 'token_type']  # token_type: input, output, total
        )
        
        self.llm_errors_total = Counter(
            'zerogravity_llm_errors_total',
            'Total LLM API errors',
            ['provider', 'model', 'error_type']
        )
        
        # Workflow metrics
        self.workflows_total = Counter(
            'zerogravity_workflows_total',
            'Total workflows executed',
            ['status']  # status: completed, failed, running
        )
        
        self.workflow_duration = Histogram(
            'zerogravity_workflow_duration_seconds',
            'Workflow execution duration in seconds'
        )
        
        # Cache metrics
        self.cache_requests_total = Counter(
            'zerogravity_cache_requests_total',
            'Total cache requests',
            ['operation']  # operation: get, set, delete
        )
        
        self.cache_hits_total = Counter(
            'zerogravity_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses_total = Counter(
            'zerogravity_cache_misses_total',
            'Total cache misses'
        )
        
        # Task queue metrics
        self.task_queue_size = Gauge(
            'zerogravity_task_queue_size',
            'Current size of task queue',
            ['queue_name']
        )
        
        self.task_processing_duration = Histogram(
            'zerogravity_task_processing_duration_seconds',
            'Task processing duration in seconds',
            ['task_type']
        )
        
        # System metrics
        self.system_uptime = Gauge(
            'zerogravity_system_uptime_seconds',
            'System uptime in seconds'
        )
        
        self.active_workers = Gauge(
            'zerogravity_active_workers',
            'Number of active workers'
        )
        
        # Initialize start time for uptime calculation
        self.start_time = time.time()
    
    def update_uptime(self):
        """Update system uptime metric"""
        current_uptime = time.time() - self.start_time
        self.system_uptime.set(current_uptime)


# Global metrics instance
metrics = ZeroGravityMetrics()


def monitor_request(endpoint: str, method: str = "GET"):
    """Decorator to monitor HTTP requests"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.active_requests.inc()
            
            try:
                result = func(*args, **kwargs)
                status_code = getattr(result, 'status_code', 200)  # Default to 200
                metrics.requests_total.labels(method=method, endpoint=endpoint, status=status_code).inc()
                return result
            except Exception as e:
                metrics.requests_total.labels(method=method, endpoint=endpoint, status=500).inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                metrics.active_requests.dec()
        
        return wrapper
    return decorator


def monitor_agent_execution(agent_role: str):
    """Decorator to monitor agent execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.agent_executions_total.labels(agent_role=agent_role).inc()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                metrics.agent_errors_total.labels(
                    agent_role=agent_role, 
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.agent_execution_duration.labels(agent_role=agent_role).observe(duration)
        
        return wrapper
    return decorator


def monitor_llm_call(provider: str, model: str):
    """Decorator to monitor LLM API calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            metrics.llm_calls_total.labels(provider=provider, model=model).inc()
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                metrics.llm_errors_total.labels(
                    provider=provider,
                    model=model,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.llm_call_duration.labels(provider=provider, model=model).observe(duration)
        
        return wrapper
    return decorator


def monitor_workflow_execution():
    """Decorator to monitor workflow execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                metrics.workflows_total.labels(status="completed").inc()
                return result
            except Exception as e:
                metrics.workflows_total.labels(status="failed").inc()
                raise
            finally:
                duration = time.time() - start_time
                metrics.workflow_duration.observe(duration)
        
        return wrapper
    return decorator


def record_cache_operation(operation: str, hit: bool = None):
    """Record cache operation metrics"""
    metrics.cache_requests_total.labels(operation=operation).inc()
    
    if operation == "get" and hit is not None:
        if hit:
            metrics.cache_hits_total.inc()
        else:
            metrics.cache_misses_total.inc()


def record_llm_tokens(provider: str, model: str, input_tokens: int, output_tokens: int):
    """Record LLM token usage"""
    metrics.llm_tokens_total.labels(
        provider=provider,
        model=model,
        token_type="input"
    ).inc(input_tokens)
    
    metrics.llm_tokens_total.labels(
        provider=provider,
        model=model,
        token_type="output"
    ).inc(output_tokens)
    
    metrics.llm_tokens_total.labels(
        provider=provider,
        model=model,
        token_type="total"
    ).inc(input_tokens + output_tokens)


def record_task_processing(task_type: str, duration: float):
    """Record task processing metrics"""
    metrics.task_processing_duration.labels(task_type=task_type).observe(duration)


class MetricsCollector:
    """Centralized metrics collector and exporter"""
    
    def __init__(self, multiprocess_mode: bool = False):
        self.multiprocess_mode = multiprocess_mode
        self.collector = None
        
        if multiprocess_mode:
            self.registry = CollectorRegistry()
            self.collector = MultiProcessCollector(self.registry)
        else:
            self.registry = REGISTRY
    
    def collect_metrics(self) -> bytes:
        """Collect and return current metrics in Prometheus format"""
        # Update uptime
        metrics.update_uptime()
        
        return generate_latest(self.registry)
    
    def get_metrics_text(self) -> str:
        """Get metrics as plain text"""
        return self.collect_metrics().decode('utf-8')
    
    def write_metrics_to_file(self, filepath: str):
        """Write metrics to a file"""
        with open(filepath, 'wb') as f:
            f.write(self.collect_metrics())
    
    def start_background_collection(self, interval: int = 30):
        """Start background thread for periodic metrics collection"""
        def collection_loop():
            while True:
                time.sleep(interval)
                # The metrics are collected on-demand, so we just update uptime
                metrics.update_uptime()
        
        thread = threading.Thread(target=collection_loop, daemon=True)
        thread.start()


class MetricsMiddleware:
    """Middleware for collecting metrics in web frameworks"""
    
    def __init__(self, collector: MetricsCollector = None):
        self.collector = collector or MetricsCollector()
    
    def process_request(self, request):
        """Process incoming request and collect metrics"""
        # Record the start time and increment active requests
        request._start_time = time.time()
        metrics.active_requests.inc()
    
    def process_response(self, request, response):
        """Process outgoing response and collect metrics"""
        # Calculate duration and record metrics
        if hasattr(request, '_start_time'):
            duration = time.time() - request._start_time
            method = getattr(request, 'method', 'GET')
            endpoint = getattr(request, 'path', '/')
            status_code = getattr(response, 'status_code', 20)
            
            metrics.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            metrics.requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()
        
        metrics.active_requests.dec()
        return response


class MetricsExporter:
    """HTTP handler for exposing metrics to Prometheus"""
    
    def __init__(self, collector: MetricsCollector = None):
        self.collector = collector or MetricsCollector()
    
    def get_metrics_response(self):
        """Get HTTP response with metrics data"""
        metrics_data = self.collector.collect_metrics()
        
        # In a real web framework, you would return this as an HTTP response
        # For example, in Flask: return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)
        return {
            'body': metrics_data.decode('utf-8'),
            'status': 200,
            'headers': {'Content-Type': CONTENT_TYPE_LATEST}
        }


# Utility functions for manual metric recording
def increment_request_counter(method: str, endpoint: str, status: int):
    """Manually increment request counter"""
    metrics.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()


def observe_request_duration(method: str, endpoint: str, duration: float):
    """Manually observe request duration"""
    metrics.request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def increment_agent_execution(agent_role: str):
    """Manually increment agent execution counter"""
    metrics.agent_executions_total.labels(agent_role=agent_role).inc()


def observe_agent_duration(agent_role: str, duration: float):
    """Manually observe agent execution duration"""
    metrics.agent_execution_duration.labels(agent_role=agent_role).observe(duration)


def increment_llm_call(provider: str, model: str):
    """Manually increment LLM call counter"""
    metrics.llm_calls_total.labels(provider=provider, model=model).inc()


def observe_llm_duration(provider: str, model: str, duration: float):
    """Manually observe LLM call duration"""
    metrics.llm_call_duration.labels(provider=provider, model=model).observe(duration)


def increment_workflow_counter(status: str):
    """Manually increment workflow counter"""
    metrics.workflows_total.labels(status=status).inc()


def observe_workflow_duration(duration: float):
    """Manually observe workflow duration"""
    metrics.workflow_duration.observe(duration)


def update_task_queue_size(queue_name: str, size: int):
    """Update task queue size metric"""
    metrics.task_queue_size.labels(queue_name=queue_name).set(size)


def update_active_workers(count: int):
    """Update active workers metric"""
    metrics.active_workers.set(count)


# Initialize metrics collector
metrics_collector = MetricsCollector()
metrics_exporter = MetricsExporter(metrics_collector)
metrics_middleware = MetricsMiddleware(metrics_collector)


def init_monitoring(multiprocess_mode: bool = False):
    """Initialize the monitoring system"""
    global metrics_collector, metrics_exporter, metrics_middleware
    
    metrics_collector = MetricsCollector(multiprocess_mode)
    metrics_exporter = MetricsExporter(metrics_collector)
    metrics_middleware = MetricsMiddleware(metrics_collector)
    
    # Start background collection
    metrics_collector.start_background_collection()
    
    return metrics_collector, metrics_exporter, metrics_middleware


def get_metrics():
    """Get current metrics as a dictionary"""
    # This function parses the Prometheus text format into a more usable structure
    metrics_text = metrics_collector.get_metrics_text()
    
    # Parse the text format into a structured format
    parsed_metrics = {}
    current_metric = None
    
    for line in metrics_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.startswith(('zerogravity_',)):
            # This is a new metric
            parts = line.split()
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0]
                value = float(parts[1])
                
                if metric_name not in parsed_metrics:
                    parsed_metrics[metric_name] = []
                
                parsed_metrics[metric_name].append(value)
    
    return parsed_metrics


def export_metrics_json() -> str:
    """Export metrics in JSON format"""
    metrics_dict = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": get_metrics()
    }
    return json.dumps(metrics_dict, indent=2)


# Example usage and initialization
if __name__ == "__main__":
    # Initialize monitoring
    init_monitoring()
    
    print("ZeroGravity Monitoring System initialized")
    print("Metrics available at /metrics endpoint")
    
    # Example of manual metric recording
    increment_request_counter("GET", "/api/test", 200)
    observe_request_duration("GET", "/api/test", 0.15)
    
    # Example of using decorators
    @monitor_request("/api/example", "POST")
    def example_api_call():
        time.sleep(0.1)  # Simulate work
        return {"status": "ok"}
    
    # Execute the example
    result = example_api_call()
    print(f"Example result: {result}")
    
    # Print some metrics
    print("\nCurrent metrics:")
    current_metrics = get_metrics()
    for metric_name, values in list(current_metrics.items())[:5]:  # Show first 5 metrics
        print(f"  {metric_name}: {values}")
