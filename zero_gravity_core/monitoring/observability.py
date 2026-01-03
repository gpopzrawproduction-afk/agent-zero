"""
Observability System for ZeroGravity

This module implements comprehensive monitoring and observability
for the ZeroGravity platform, including metrics collection,
distributed tracing, and health monitoring.
"""
import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import psutil
import GPUtil
from contextlib import contextmanager, asynccontextmanager
import traceback
import threading
from collections import defaultdict, deque
import statistics
from functools import wraps


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ComponentType(Enum):
    """Types of system components monitored"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    LLM_PROVIDER = "llm_provider"
    API_GATEWAY = "api_gateway"
    DATABASE = "database"
    CACHE = "cache"
    TASK_QUEUE = "task_queue"
    SYSTEM = "system"


@dataclass
class MetricPoint:
    """A single metric measurement point"""
    name: str
    value: Union[int, float]
    type: MetricType
    labels: Dict[str, str]
    timestamp: datetime
    component: ComponentType
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component.value
        }


class MetricsCollector:
    """Collects and aggregates metrics from various system components"""
    
    def __init__(self, retention_minutes: int = 60):
        self.retention_minutes = retention_minutes
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=retention_minutes * 60))  # Store per second for retention period
        self.lock = threading.Lock()
        self.logger = logging.getLogger("MetricsCollector")
    
    def add_metric(self, metric: MetricPoint):
        """Add a metric point to the collector"""
        with self.lock:
            key = f"{metric.name}:{':'.join(sorted([f'{k}={v}' for k, v in metric.labels.items()]))}"
            self.metrics[key].append(metric)
    
    def get_metrics(self, name: str = None, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get metrics by name and optional labels"""
        with self.lock:
            if name is None:
                # Return all metrics
                all_metrics = []
                for metric_list in self.metrics.values():
                    all_metrics.extend(list(metric_list))
                return sorted(all_metrics, key=lambda x: x.timestamp, reverse=True)
            
            # Build search key
            label_str = ':'.join(sorted([f'{k}={v}' for k, v in (labels or {}).items()]))
            search_key = f"{name}:{label_str}" if label_str else name
            
            # Find matching metrics
            results = []
            for key, metric_list in self.metrics.items():
                if key.startswith(search_key):
                    results.extend(list(metric_list))
            
            return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def get_aggregated_metrics(self, name: str, labels: Dict[str, str] = None, 
                              window_minutes: int = 5) -> Dict[str, float]:
        """Get aggregated metrics for a time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        metrics = self.get_metrics(name, labels)
        
        # Filter by time window
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics if isinstance(m.value, (int, float))]
        
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "p50": statistics.median(values) if len(values) > 1 else values[0],
            "p95": sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0],
            "p99": sorted(values)[int(0.99 * len(values))] if len(values) > 1 else values[0]
        }
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format"""
        output = []
        
        with self.lock:
            for key, metric_list in self.metrics.items():
                if not metric_list:
                    continue
                
                # Parse key to get name and labels
                parts = key.split(':')
                name = parts[0]
                labels_part = ':'.join(parts[1:]) if len(parts) > 1 else ""
                
                # Parse labels from labels_part
                labels = {}
                if labels_part:
                    for label_pair in labels_part.split(','):
                        if '=' in label_pair:
                            k, v = label_pair.split('=', 1)
                            labels[k] = v
                
                # Get the latest value
                latest = metric_list[-1]
                
                # Format labels
                label_str = ','.join([f'{k}="{v}"' for k, v in labels.items()])
                if label_str:
                    label_str = f'{{{label_str}}}'
                
                # Add metric in Prometheus format
                output.append(f"# TYPE {name} {latest.type.value}")
                output.append(f"{name}{label_str} {latest.value} {int(latest.timestamp.timestamp() * 1000)}")
        
        return '\n'.join(output)


class SystemMonitor:
    """Monitors system-level metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("SystemMonitor")
        self.previous_io = psutil.disk_io_counters()
        self.previous_net = psutil.net_io_counters()
    
    def collect_system_metrics(self) -> List[MetricPoint]:
        """Collect system-level metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        metrics.append(MetricPoint(
            name="system_cpu_percent",
            value=cpu_percent,
            type=MetricType.GAUGE,
            labels={"type": "overall"},
            timestamp=timestamp,
            component=ComponentType.SYSTEM
        ))
        
        metrics.append(MetricPoint(
            name="system_cpu_count",
            value=cpu_count,
            type=MetricType.GAUGE,
            labels={},
            timestamp=timestamp,
            component=ComponentType.SYSTEM
        ))
        
        if cpu_freq:
            metrics.append(MetricPoint(
                name="system_cpu_frequency_mhz",
                value=cpu_freq.current,
                type=MetricType.GAUGE,
                labels={},
                timestamp=timestamp,
                component=ComponentType.SYSTEM
            ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        metrics.append(MetricPoint(
            name="system_memory_used_bytes",
            value=memory.used,
            type=MetricType.GAUGE,
            labels={},
            timestamp=timestamp,
            component=ComponentType.SYSTEM
        ))
        
        metrics.append(MetricPoint(
            name="system_memory_available_bytes",
            value=memory.available,
            type=MetricType.GAUGE,
            labels={},
            timestamp=timestamp,
            component=ComponentType.SYSTEM
        ))
        
        metrics.append(MetricPoint(
            name="system_memory_percent",
            value=memory.percent,
            type=MetricType.GAUGE,
            labels={},
            timestamp=timestamp,
            component=ComponentType.SYSTEM
        ))
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        if disk_io:
            read_bytes = disk_io.read_bytes
            write_bytes = disk_io.write_bytes
            
            if self.previous_io:
                read_bytes_per_sec = (read_bytes - self.previous_io.read_bytes) / 1.0  # 1 second interval
                write_bytes_per_sec = (write_bytes - self.previous_io.write_bytes) / 1.0
                
                metrics.append(MetricPoint(
                    name="system_disk_read_bytes_per_second",
                    value=read_bytes_per_sec,
                    type=MetricType.GAUGE,
                    labels={},
                    timestamp=timestamp,
                    component=ComponentType.SYSTEM
                ))
                
                metrics.append(MetricPoint(
                    name="system_disk_write_bytes_per_second",
                    value=write_bytes_per_sec,
                    type=MetricType.GAUGE,
                    labels={},
                    timestamp=timestamp,
                    component=ComponentType.SYSTEM
                ))
            
            self.previous_io = disk_io
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        if net_io:
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            
            if self.previous_net:
                bytes_sent_per_sec = (bytes_sent - self.previous_net.bytes_sent) / 1.0
                bytes_recv_per_sec = (bytes_recv - self.previous_net.bytes_recv) / 1.0
                
                metrics.append(MetricPoint(
                    name="system_network_sent_bytes_per_second",
                    value=bytes_sent_per_sec,
                    type=MetricType.GAUGE,
                    labels={},
                    timestamp=timestamp,
                    component=ComponentType.SYSTEM
                ))
                
                metrics.append(MetricPoint(
                    name="system_network_recv_bytes_per_second",
                    value=bytes_recv_per_sec,
                    type=MetricType.GAUGE,
                    labels={},
                    timestamp=timestamp,
                    component=ComponentType.SYSTEM
                ))
            
            self.previous_net = net_io
        
        # Process metrics
        process = psutil.Process()
        with process.oneshot():
            metrics.append(MetricPoint(
                name="process_memory_rss_bytes",
                value=process.memory_info().rss,
                type=MetricType.GAUGE,
                labels={},
                timestamp=timestamp,
                component=ComponentType.SYSTEM
            ))
            
            metrics.append(MetricPoint(
                name="process_cpu_percent",
                value=process.cpu_percent(),
                type=MetricType.GAUGE,
                labels={},
                timestamp=timestamp,
                component=ComponentType.SYSTEM
            ))
        
        return metrics


class AgentMonitor:
    """Monitors agent-specific metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("AgentMonitor")
        self.agent_stats = defaultdict(lambda: {
            "execution_count": 0,
            "total_duration": 0.0,
            "error_count": 0,
            "last_execution": None
        })
    
    def record_agent_execution(self, agent_role: str, duration: float, success: bool = True):
        """Record an agent execution"""
        stats = self.agent_stats[agent_role]
        stats["execution_count"] += 1
        stats["total_duration"] += duration
        stats["last_execution"] = datetime.utcnow()
        
        if not success:
            stats["error_count"] += 1
    
    def collect_agent_metrics(self) -> List[MetricPoint]:
        """Collect agent-specific metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        for agent_role, stats in self.agent_stats.items():
            labels = {"agent_role": agent_role}
            
            metrics.append(MetricPoint(
                name="agent_executions_total",
                value=stats["execution_count"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.AGENT
            ))
            
            if stats["execution_count"] > 0:
                avg_duration = stats["total_duration"] / stats["execution_count"]
                
                metrics.append(MetricPoint(
                    name="agent_duration_seconds_avg",
                    value=avg_duration,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.AGENT
                ))
                
                metrics.append(MetricPoint(
                    name="agent_error_total",
                    value=stats["error_count"],
                    type=MetricType.COUNTER,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.AGENT
                ))
        
        return metrics


class WorkflowMonitor:
    """Monitors workflow-specific metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("WorkflowMonitor")
        self.workflow_stats = defaultdict(lambda: {
            "execution_count": 0,
            "success_count": 0,
            "total_duration": 0.0,
            "error_count": 0,
            "last_execution": None
        })
    
    def record_workflow_execution(self, workflow_id: str, duration: float, success: bool = True):
        """Record a workflow execution"""
        stats = self.workflow_stats[workflow_id]
        stats["execution_count"] += 1
        stats["total_duration"] += duration
        stats["last_execution"] = datetime.utcnow()
        
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
    
    def collect_workflow_metrics(self) -> List[MetricPoint]:
        """Collect workflow-specific metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        for workflow_id, stats in self.workflow_stats.items():
            labels = {"workflow_id": workflow_id}
            
            metrics.append(MetricPoint(
                name="workflow_executions_total",
                value=stats["execution_count"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.WORKFLOW
            ))
            
            metrics.append(MetricPoint(
                name="workflow_success_total",
                value=stats["success_count"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.WORKFLOW
            ))
            
            metrics.append(MetricPoint(
                name="workflow_error_total",
                value=stats["error_count"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.WORKFLOW
            ))
            
            if stats["execution_count"] > 0:
                avg_duration = stats["total_duration"] / stats["execution_count"]
                
                metrics.append(MetricPoint(
                    name="workflow_duration_seconds_avg",
                    value=avg_duration,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.WORKFLOW
                ))
                
                success_rate = stats["success_count"] / stats["execution_count"]
                metrics.append(MetricPoint(
                    name="workflow_success_rate",
                    value=success_rate,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.WORKFLOW
                ))
        
        return metrics


class LLMProviderMonitor:
    """Monitors LLM provider metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("LLMProviderMonitor")
        self.provider_stats = defaultdict(lambda: {
            "api_calls_total": 0,
            "tokens_sent_total": 0,
            "tokens_received_total": 0,
            "total_duration": 0.0,
            "error_count": 0,
            "last_call": None
        })
    
    def record_llm_call(self, provider: str, duration: float, 
                       tokens_sent: int = 0, tokens_received: int = 0, 
                       success: bool = True):
        """Record an LLM API call"""
        stats = self.provider_stats[provider]
        stats["api_calls_total"] += 1
        stats["tokens_sent_total"] += tokens_sent
        stats["tokens_received_total"] += tokens_received
        stats["total_duration"] += duration
        stats["last_call"] = datetime.utcnow()
        
        if not success:
            stats["error_count"] += 1
    
    def collect_llm_metrics(self) -> List[MetricPoint]:
        """Collect LLM provider metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        for provider, stats in self.provider_stats.items():
            labels = {"provider": provider}
            
            metrics.append(MetricPoint(
                name="llm_api_calls_total",
                value=stats["api_calls_total"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.LLM_PROVIDER
            ))
            
            metrics.append(MetricPoint(
                name="llm_tokens_sent_total",
                value=stats["tokens_sent_total"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.LLM_PROVIDER
            ))
            
            metrics.append(MetricPoint(
                name="llm_tokens_received_total",
                value=stats["tokens_received_total"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.LLM_PROVIDER
            ))
            
            metrics.append(MetricPoint(
                name="llm_error_total",
                value=stats["error_count"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.LLM_PROVIDER
            ))
            
            if stats["api_calls_total"] > 0:
                avg_duration = stats["total_duration"] / stats["api_calls_total"]
                
                metrics.append(MetricPoint(
                    name="llm_duration_seconds_avg",
                    value=avg_duration,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.LLM_PROVIDER
                ))
                
                avg_tokens_sent = stats["tokens_sent_total"] / stats["api_calls_total"]
                metrics.append(MetricPoint(
                    name="llm_tokens_sent_avg",
                    value=avg_tokens_sent,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.LLM_PROVIDER
                ))
                
                avg_tokens_received = stats["tokens_received_total"] / stats["api_calls_total"]
                metrics.append(MetricPoint(
                    name="llm_tokens_received_avg",
                    value=avg_tokens_received,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.LLM_PROVIDER
                ))
        
        return metrics


class APIMonitor:
    """Monitors API gateway metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger("APIMonitor")
        self.endpoint_stats = defaultdict(lambda: {
            "requests_total": 0,
            "errors_total": 0,
            "total_duration": 0.0,
            "status_codes": defaultdict(int)
        })
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          duration: float):
        """Record an API request"""
        key = f"{method}:{endpoint}"
        stats = self.endpoint_stats[key]
        
        stats["requests_total"] += 1
        stats["total_duration"] += duration
        stats["status_codes"][status_code] += 1
        
        if 400 <= status_code < 600:
            stats["errors_total"] += 1
    
    def collect_api_metrics(self) -> List[MetricPoint]:
        """Collect API gateway metrics"""
        metrics = []
        timestamp = datetime.utcnow()
        
        for endpoint_method, stats in self.endpoint_stats.items():
            method, endpoint = endpoint_method.split(':', 1)
            labels = {"method": method, "endpoint": endpoint}
            
            metrics.append(MetricPoint(
                name="api_requests_total",
                value=stats["requests_total"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.API_GATEWAY
            ))
            
            metrics.append(MetricPoint(
                name="api_errors_total",
                value=stats["errors_total"],
                type=MetricType.COUNTER,
                labels=labels,
                timestamp=timestamp,
                component=ComponentType.API_GATEWAY
            ))
            
            if stats["requests_total"] > 0:
                avg_duration = stats["total_duration"] / stats["requests_total"]
                
                metrics.append(MetricPoint(
                    name="api_duration_seconds_avg",
                    value=avg_duration,
                    type=MetricType.GAUGE,
                    labels=labels,
                    timestamp=timestamp,
                    component=ComponentType.API_GATEWAY
                ))
                
                # Status code breakdown
                for status_code, count in stats["status_codes"].items():
                    status_labels = {**labels, "status_code": str(status_code)}
                    
                    metrics.append(MetricPoint(
                        name="api_status_codes_total",
                        value=count,
                        type=MetricType.COUNTER,
                        labels=status_labels,
                        timestamp=timestamp,
                        component=ComponentType.API_GATEWAY
                    ))
        
        return metrics


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self):
        self.logger = logging.getLogger("HealthChecker")
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_health_check = datetime.min
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
        
        self.last_health_check = datetime.utcnow()
        return results
    
    def is_system_healthy(self) -> bool:
        """Check if the overall system is healthy"""
        results = self.run_health_checks()
        return all(results.values()) if results else True


class ObservabilityManager:
    """Main observability manager that coordinates all monitoring components"""
    
    def __init__(self, retention_minutes: int = 60):
        self.metrics_collector = MetricsCollector(retention_minutes)
        self.system_monitor = SystemMonitor()
        self.agent_monitor = AgentMonitor()
        self.workflow_monitor = WorkflowMonitor()
        self.llm_monitor = LLMProviderMonitor()
        self.api_monitor = APIMonitor()
        self.health_checker = HealthChecker()
        
        self.logger = logging.getLogger("ObservabilityManager")
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring = False
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
            self.logger.info("Observability monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        self.logger.info("Observability monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect all metrics
                self._collect_all_metrics()
                
                # Sleep until next collection
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def _collect_all_metrics(self):
        """Collect metrics from all monitors"""
        timestamp = datetime.utcnow()
        
        # Collect system metrics
        try:
            system_metrics = self.system_monitor.collect_system_metrics()
            for metric in system_metrics:
                self.metrics_collector.add_metric(metric)
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        # Collect agent metrics
        try:
            agent_metrics = self.agent_monitor.collect_agent_metrics()
            for metric in agent_metrics:
                self.metrics_collector.add_metric(metric)
        except Exception as e:
            self.logger.error(f"Error collecting agent metrics: {e}")
        
        # Collect workflow metrics
        try:
            workflow_metrics = self.workflow_monitor.collect_workflow_metrics()
            for metric in workflow_metrics:
                self.metrics_collector.add_metric(metric)
        except Exception as e:
            self.logger.error(f"Error collecting workflow metrics: {e}")
        
        # Collect LLM metrics
        try:
            llm_metrics = self.llm_monitor.collect_llm_metrics()
            for metric in llm_metrics:
                self.metrics_collector.add_metric(metric)
        except Exception as e:
            self.logger.error(f"Error collecting LLM metrics: {e}")
        
        # Collect API metrics
        try:
            api_metrics = self.api_monitor.collect_api_metrics()
            for metric in api_metrics:
                self.metrics_collector.add_metric(metric)
        except Exception as e:
            self.logger.error(f"Error collecting API metrics: {e}")
    
    def get_metrics(self, name: str = None, labels: Dict[str, str] = None) -> List[MetricPoint]:
        """Get collected metrics"""
        return self.metrics_collector.get_metrics(name, labels)
    
    def get_aggregated_metrics(self, name: str, labels: Dict[str, str] = None, 
                              window_minutes: int = 5) -> Dict[str, float]:
        """Get aggregated metrics"""
        return self.metrics_collector.get_aggregated_metrics(name, labels, window_minutes)
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return self.metrics_collector.export_prometheus_format()
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run health checks"""
        return self.health_checker.run_health_checks()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_results = self.run_health_checks()
        is_healthy = all(health_results.values()) if health_results else True
        
        # Get some key metrics for health assessment
        system_metrics = self.get_metrics("system_cpu_percent")
        cpu_usage = system_metrics[0].value if system_metrics else 0
        
        memory_metrics = self.get_metrics("system_memory_percent")
        memory_usage = memory_metrics[0].value if memory_metrics else 0
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": health_results,
            "system_metrics": {
                "cpu_percent": cpu_usage,
                "memory_percent": memory_usage
            }
        }
    
    # Decorators for easy metric collection
    
    def measure_agent_execution(self, agent_role: str):
        """Decorator to measure agent execution time"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    self.agent_monitor.record_agent_execution(agent_role, duration, success)
                    
                    # Add to metrics collector
                    self.metrics_collector.add_metric(MetricPoint(
                        name="agent_duration_seconds",
                        value=duration,
                        type=MetricType.HISTOGRAM,
                        labels={"agent_role": agent_role, "success": str(success)},
                        timestamp=datetime.utcnow(),
                        component=ComponentType.AGENT
                    ))
            return wrapper
        return decorator
    
    def measure_workflow_execution(self, workflow_id: str):
        """Decorator to measure workflow execution time"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    self.workflow_monitor.record_workflow_execution(workflow_id, duration, success)
                    
                    # Add to metrics collector
                    self.metrics_collector.add_metric(MetricPoint(
                        name="workflow_duration_seconds",
                        value=duration,
                        type=MetricType.HISTOGRAM,
                        labels={"workflow_id": workflow_id, "success": str(success)},
                        timestamp=datetime.utcnow(),
                        component=ComponentType.WORKFLOW
                    ))
            return wrapper
        return decorator
    
    def measure_llm_call(self, provider: str):
        """Decorator to measure LLM API calls"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                tokens_sent = 0
                tokens_received = 0
                
                try:
                    result = await func(*args, **kwargs)
                    # Try to estimate token counts if possible
                    if isinstance(result, str):
                        tokens_received = len(result.split())  # Rough estimation
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    self.llm_monitor.record_llm_call(provider, duration, 
                                                   tokens_sent, tokens_received, success)
                    
                    # Add to metrics collector
                    self.metrics_collector.add_metric(MetricPoint(
                        name="llm_duration_seconds",
                        value=duration,
                        type=MetricType.HISTOGRAM,
                        labels={"provider": provider, "success": str(success)},
                        timestamp=datetime.utcnow(),
                        component=ComponentType.LLM_PROVIDER
                    ))
            return wrapper
        return decorator


# Global observability manager instance
observability_manager: Optional[ObservabilityManager] = None


def init_observability(retention_minutes: int = 60) -> ObservabilityManager:
    """Initialize the observability system"""
    global observability_manager
    observability_manager = ObservabilityManager(retention_minutes)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    return observability_manager


def get_observability_manager() -> Optional[ObservabilityManager]:
    """Get the global observability manager instance"""
    return observability_manager


def start_observability_monitoring(interval: float = 30.0):
    """Start the observability monitoring system"""
    if observability_manager is None:
        init_observability()
    
    observability_manager.start_monitoring(interval)


def stop_observability_monitoring():
    """Stop the observability monitoring system"""
    global observability_manager
    if observability_manager:
        observability_manager.stop_monitoring()
        observability_manager = None


def get_current_metrics(name: str = None, labels: Dict[str, str] = None) -> List[MetricPoint]:
    """Get current metrics from the observability system"""
    if observability_manager is None:
        return []
    
    return observability_manager.get_metrics(name, labels)


def get_system_health_status() -> Dict[str, Any]:
    """Get the current system health status"""
    if observability_manager is None:
        return {"status": "unknown", "timestamp": datetime.utcnow().isoformat()}
    
    return observability_manager.get_system_health()


def export_prometheus_format() -> str:
    """Export metrics in Prometheus format"""
    if observability_manager is None:
        return ""
    
    return observability_manager.export_prometheus_metrics()


# Example usage and testing
async def test_observability_system():
    """Test the observability system"""
    print("Testing ZeroGravity Observability System...")
    
    # Initialize observability
    obs_manager = init_observability(retention_minutes=10)
    
    # Start monitoring
    start_observability_monitoring(interval=10.0)  # Check every 10 seconds
    
    print("Observability system initialized and monitoring started")
    
    # Simulate some activity
    print("Simulating system activity...")
    
    # Record some agent executions
    obs_manager.agent_monitor.record_agent_execution("architect", 2.5, success=True)
    obs_manager.agent_monitor.record_agent_execution("engineer", 1.8, success=True)
    obs_manager.agent_monitor.record_agent_execution("designer", 2.1, success=False)
    
    # Record some workflow executions
    obs_manager.workflow_monitor.record_workflow_execution("wf_001", 15.2, success=True)
    obs_manager.workflow_monitor.record_workflow_execution("wf_002", 12.7, success=True)
    
    # Record some LLM calls
    obs_manager.llm_monitor.record_llm_call("openai", 3.2, tokens_sent=100, tokens_received=250, success=True)
    obs_manager.llm_monitor.record_llm_call("anthropic", 2.8, tokens_sent=80, tokens_received=200, success=True)
    
    # Wait a bit to see metrics collection
    await asyncio.sleep(5)
    
    # Get some metrics
    print("\nRecent metrics:")
    cpu_metrics = get_current_metrics("system_cpu_percent")
    if cpu_metrics:
        latest_cpu = cpu_metrics[0]
        print(f"  Latest CPU usage: {latest_cpu.value}%")
    
    agent_metrics = get_current_metrics("agent_executions_total", {"agent_role": "architect"})
    if agent_metrics:
        latest_agent = agent_metrics[0]
        print(f"  Architect executions: {latest_agent.value}")
    
    # Get aggregated metrics
    print("\nAggregated metrics (last 5 minutes):")
    agg_metrics = obs_manager.get_aggregated_metrics("agent_duration_seconds_avg", 
                                                    {"agent_role": "architect"}, 
                                                    window_minutes=5)
    print(f"  Architect avg duration: {agg_metrics}")
    
    # Get system health
    print("\nSystem health:")
    health = get_system_health_status()
    print(f"  Status: {health['status']}")
    print(f"  CPU: {health['system_metrics']['cpu_percent']}%")
    print(f"  Memory: {health['system_metrics']['memory_percent']}%")
    
    # Export Prometheus format
    print("\nMetrics in Prometheus format:")
    prometheus_data = export_prometheus_format()
    print(prometheus_data[:500] + "..." if len(prometheus_data) > 500 else prometheus_data)
    
    # Stop monitoring
    stop_observability_monitoring()
    print("\nObservability system stopped")


if __name__ == "__main__":
    # For testing purposes
    print("Starting ZeroGravity Observability System example...")
    # asyncio.run(test_observability_system())
