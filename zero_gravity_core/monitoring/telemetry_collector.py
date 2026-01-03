# Telemetry Collector for AI Ops Agent
# Collects structured metrics from all agents and stores them for analysis

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
from pathlib import Path
import aiofiles

from zero_gravity_core.schemas.ai_ops_schemas import MetricData

logger = logging.getLogger(__name__)

class TelemetryCollector:
    """
    Collects and processes telemetry data from all agents in the system.
    Provides real-time metric processing and validation.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "zero_gravity_core/monitoring/telemetry_data"
        self.metrics_buffer: List[MetricData] = []
        self.buffer_size_limit = 1000
        self.flush_interval = 30  # seconds
        self.validation_rules = []
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Start background flush task
        self.flush_task = None
        self.is_running = False
        
        logger.info(f"Telemetry Collector initialized with storage path: {self.storage_path}")
    
    async def start(self):
        """Start the telemetry collector with background tasks"""
        self.is_running = True
        self.flush_task = asyncio.create_task(self._background_flush())
        logger.info("Telemetry Collector started")
    
    async def stop(self):
        """Stop the telemetry collector and flush remaining data"""
        self.is_running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Flush any remaining data
        await self.flush()
        logger.info("Telemetry Collector stopped")
    
    async def emit_telemetry(self, metric_data: MetricData):
        """
        Emit telemetry data from an agent execution
        """
        # Validate the metric data
        if not await self._validate_metric(metric_data):
            logger.warning(f"Invalid metric data received: {metric_data}")
            return False
        
        # Add to buffer
        self.metrics_buffer.append(metric_data)
        
        # Check if buffer needs to be flushed
        if len(self.metrics_buffer) >= self.buffer_size_limit:
            await self.flush()
        
        logger.debug(f"Telemetry emitted for agent {metric_data.agent}, task {metric_data.task_id}")
        return True
    
    async def _validate_metric(self, metric_data: MetricData) -> bool:
        """
        Validate metric data against defined rules
        """
        # Basic validation - all required fields should be present
        if not metric_data.agent:
            logger.error("Metric validation failed: Missing agent name")
            return False
        
        if not metric_data.task_id:
            logger.error("Metric validation failed: Missing task ID")
            return False
        
        if metric_data.latency_ms < 0:
            logger.error(f"Metric validation failed: Invalid latency {metric_data.latency_ms}")
            return False
        
        if metric_data.cost_usd < 0:
            logger.error(f"Metric validation failed: Invalid cost {metric_data.cost_usd}")
            return False
        
        if not (0.0 <= metric_data.quality_score <= 1.0):
            logger.error(f"Metric validation failed: Invalid quality score {metric_data.quality_score}")
            return False
        
        # Apply additional validation rules if any
        for rule in self.validation_rules:
            if not rule(metric_data):
                logger.error(f"Metric validation failed: Custom rule violation for {metric_data.task_id}")
                return False
        
        return True
    
    async def flush(self):
        """
        Flush buffered metrics to storage
        """
        if not self.metrics_buffer:
            return
        
        try:
            # Group metrics by date for file storage
            metrics_by_date = {}
            for metric in self.metrics_buffer:
                date_str = datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d')
                if date_str not in metrics_by_date:
                    metrics_by_date[date_str] = []
                metrics_by_date[date_str].append(asdict(metric))
            
            # Write metrics to files
            for date_str, metrics_list in metrics_by_date.items():
                file_path = Path(self.storage_path) / f"telemetry_{date_str}.jsonl"
                
                async with aiofiles.open(file_path, 'a') as f:
                    for metric_dict in metrics_list:
                        await f.write(json.dumps(metric_dict) + '\n')
            
            logger.info(f"Flushed {len(self.metrics_buffer)} metrics to storage")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing telemetry data: {str(e)}")
            # Keep metrics in buffer for retry
            raise
    
    async def _background_flush(self):
        """
        Background task to periodically flush metrics
        """
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                if self.metrics_buffer:
                    await self.flush()
            except asyncio.CancelledError:
                logger.info("Background flush task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in background flush task: {str(e)}")
    
    async def query_metrics(
        self, 
        agent_name: Optional[str] = None, 
        start_time: Optional[float] = None, 
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[MetricData]:
        """
        Query metrics based on filters
        """
        # This is a simplified implementation - in production, you'd want to use
        # a proper database or more efficient file querying
        all_metrics = await self._load_all_metrics()
        
        filtered_metrics = []
        for metric_dict in all_metrics:
            metric = MetricData(**metric_dict)
            
            # Apply filters
            if agent_name and metric.agent != agent_name:
                continue
            
            if start_time and metric.timestamp < start_time:
                continue
                
            if end_time and metric.timestamp > end_time:
                continue
            
            filtered_metrics.append(metric)
        
        # Apply limit
        if limit:
            filtered_metrics = filtered_metrics[-limit:]  # Return most recent
        
        return filtered_metrics
    
    async def _load_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Load all metrics from storage (simplified implementation)
        """
        all_metrics = []
        
        # Look for telemetry files in the storage directory
        telemetry_files = list(Path(self.storage_path).glob("telemetry_*.jsonl"))
        
        for file_path in telemetry_files:
            try:
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    for line in content.strip().split('\n'):
                        if line.strip():
                            all_metrics.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error reading telemetry file {file_path}: {str(e)}")
        
        return all_metrics
    
    async def get_agent_performance_summary(
        self, 
        agent_name: str, 
        start_time: Optional[float] = None, 
        end_time: Optional[float] = None
    ):
        """
        Get performance summary for a specific agent
        """
        metrics = await self.query_metrics(
            agent_name=agent_name,
            start_time=start_time,
            end_time=end_time
        )
        
        if not metrics:
            return {
                "agent_name": agent_name,
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_cost": 0.0,
                "total_latency": 0.0,
                "average_quality": 0.0,
                "start_time": start_time,
                "end_time": end_time
            }
        
        total_tasks = len(metrics)
        successful_tasks = sum(1 for m in metrics if m.success)
        total_cost = sum(m.cost_usd for m in metrics)
        total_latency = sum(m.latency_ms for m in metrics)
        average_quality = sum(m.quality_score for m in metrics) / len(metrics) if metrics else 0.0
        
        return {
            "agent_name": agent_name,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "total_cost": total_cost,
            "total_latency": total_latency,
            "average_quality": average_quality,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def add_validation_rule(self, rule: Callable[[MetricData], bool]):
        """
        Add a custom validation rule for metrics
        """
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.__name__}")
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health metrics
        """
        # Get all metrics from the last hour
        one_hour_ago = time.time() - 3600
        recent_metrics = await self.query_metrics(start_time=one_hour_ago)
        
        if not recent_metrics:
            return {
                "status": "no_data",
                "total_agents": 0,
                "tasks_completed_last_hour": 0,
                "avg_latency_ms": 0,
                "total_cost_last_hour": 0.0,
                "success_rate": 0.0
            }
        
        # Calculate health metrics
        total_agents = len(set(m.agent for m in recent_metrics))
        tasks_completed_last_hour = len(recent_metrics)
        avg_latency_ms = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
        total_cost_last_hour = sum(m.cost_usd for m in recent_metrics)
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        
        # Determine system status based on thresholds
        status = "healthy"
        if success_rate < 0.8:
            status = "degraded"
        elif success_rate < 0.95:
            status = "warning"
        
        return {
            "status": status,
            "total_agents": total_agents,
            "tasks_completed_last_hour": tasks_completed_last_hour,
            "avg_latency_ms": avg_latency_ms,
            "total_cost_last_hour": total_cost_last_hour,
            "success_rate": success_rate
        }
