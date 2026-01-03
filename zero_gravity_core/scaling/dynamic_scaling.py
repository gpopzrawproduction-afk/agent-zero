"""
Dynamic Resource Scaling System for ZeroGravity

This module implements dynamic resource scaling to automatically
adjust compute resources based on workload demands for the ZeroGravity platform.
"""
import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import statistics
import psutil
import os
from pathlib import Path
import logging


class ResourceType(Enum):
    """Types of resources that can be scaled"""
    CPU = "cpu"
    MEMORY = "memory"
    WORKERS = "workers"
    CONCURRENCY = "concurrency"
    LLM_CONNECTIONS = "llm_connections"
    CACHE_SIZE = "cache_size"
    DATABASE_CONNECTIONS = "database_connections"


class ScalingAction(Enum):
    """Actions that can be taken during scaling"""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    ALERT = "alert"


class ScalingStrategy(Enum):
    """Different scaling strategies"""
    AGGRESSIVE = "aggressive"      # Scale quickly with high demand
    CONSERVATIVE = "conservative"  # Scale slowly to minimize costs
    PREDICTIVE = "predictive"      # Use predictive algorithms
    HYBRID = "hybrid"              # Combination of strategies


@dataclass
class ResourceMetrics:
    """Current resource metrics"""
    cpu_percent: float
    memory_percent: float
    active_workers: int
    queue_size: int
    pending_tasks: int
    active_llm_connections: int
    cache_hit_rate: float
    avg_response_time: float
    requests_per_second: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ScalingRecommendation:
    """Recommendation for scaling action"""
    action: ScalingAction
    resource_type: ResourceType
    suggested_change: float  # Percentage change
    confidence: float  # 0.0 to 1.0
    reason: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ResourceMonitor:
    """Monitors resource usage and collects metrics"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history: List[ResourceMetrics] = []
        self.logger = logging.getLogger("ResourceMonitor")
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history size within limits
                if len(self.metrics_history) > self.history_size:
                    self.metrics_history = self.metrics_history[-self.history_size:]
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        # Get system-level metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # For now, using mock values for application-specific metrics
        # In a real implementation, these would come from the actual system
        active_workers = getattr(self, '_active_workers', 4)
        queue_size = getattr(self, '_queue_size', 0)
        pending_tasks = getattr(self, '_pending_tasks', 0)
        active_llm_connections = getattr(self, '_active_llm_connections', 2)
        cache_hit_rate = getattr(self, '_cache_hit_rate', 0.85)
        avg_response_time = getattr(self, '_avg_response_time', 0.5)
        requests_per_second = getattr(self, '_requests_per_second', 10.0)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_workers=active_workers,
            queue_size=queue_size,
            pending_tasks=pending_tasks,
            active_llm_connections=active_llm_connections,
            cache_hit_rate=cache_hit_rate,
            avg_response_time=avg_response_time,
            requests_per_second=requests_per_second
        )
    
    def get_average_metrics(self, window_minutes: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over a time window"""
        if not self.metrics_history:
            return None
        
        # Filter metrics within the time window
        cutoff_time = datetime.utcnow().timestamp() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp.timestamp() >= cutoff_time
        ]
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        cpu_percent = statistics.mean([m.cpu_percent for m in recent_metrics])
        memory_percent = statistics.mean([m.memory_percent for m in recent_metrics])
        active_workers = statistics.mean([m.active_workers for m in recent_metrics])
        queue_size = statistics.mean([m.queue_size for m in recent_metrics])
        pending_tasks = statistics.mean([m.pending_tasks for m in recent_metrics])
        active_llm_connections = statistics.mean([m.active_llm_connections for m in recent_metrics])
        cache_hit_rate = statistics.mean([m.cache_hit_rate for m in recent_metrics])
        avg_response_time = statistics.mean([m.avg_response_time for m in recent_metrics])
        requests_per_second = statistics.mean([m.requests_per_second for m in recent_metrics])
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_workers=active_workers,
            queue_size=queue_size,
            pending_tasks=pending_tasks,
            active_llm_connections=active_llm_connections,
            cache_hit_rate=cache_hit_rate,
            avg_response_time=avg_response_time,
            requests_per_second=requests_per_second
        )
    
    def get_trend_metrics(self, window_minutes: int = 10) -> Dict[str, float]:
        """Get trend metrics (rate of change)"""
        if len(self.metrics_history) < 2:
            return {}
        
        # Get metrics from the specified window
        cutoff_time = datetime.utcnow().timestamp() - (window_minutes * 60)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp.timestamp() >= cutoff_time
        ]
        
        if len(recent_metrics) < 2:
            return {}
        
        # Sort by timestamp
        recent_metrics.sort(key=lambda x: x.timestamp)
        
        # Calculate trends (rate of change)
        first = recent_metrics[0]
        last = recent_metrics[-1]
        
        time_diff = (last.timestamp - first.timestamp).total_seconds() / 60  # in minutes
        
        if time_diff == 0:
            return {}
        
        trends = {
            'cpu_trend': (last.cpu_percent - first.cpu_percent) / time_diff,
            'memory_trend': (last.memory_percent - first.memory_percent) / time_diff,
            'workers_trend': (last.active_workers - first.active_workers) / time_diff,
            'queue_trend': (last.queue_size - first.queue_size) / time_diff,
            'pending_tasks_trend': (last.pending_tasks - first.pending_tasks) / time_diff,
            'llm_connections_trend': (last.active_llm_connections - first.active_llm_connections) / time_diff,
            'response_time_trend': (last.avg_response_time - first.avg_response_time) / time_diff,
            'rps_trend': (last.requests_per_second - first.requests_per_second) / time_diff
        }
        
        return trends


class ScalingPolicy:
    """Defines scaling policies and thresholds"""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.thresholds = self._get_default_thresholds()
        self.logger = logging.getLogger("ScalingPolicy")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default thresholds based on strategy"""
        base_thresholds = {
            ResourceType.CPU: {
                'high_threshold': 80.0,  # Scale up when CPU > 80%
                'low_threshold': 30.0,   # Scale down when CPU < 30%
                'critical_threshold': 95.0  # Critical level
            },
            ResourceType.MEMORY: {
                'high_threshold': 85.0,
                'low_threshold': 40.0,
                'critical_threshold': 95.0
            },
            ResourceType.WORKERS: {
                'high_threshold': 80.0,  # When queue/pending tasks are high
                'low_threshold': 20.0,   # When system is underutilized
                'critical_threshold': 95.0
            },
            ResourceType.LLM_CONNECTIONS: {
                'high_threshold': 90.0,
                'low_threshold': 25.0,
                'critical_threshold': 98.0
            }
        }
        
        # Adjust thresholds based on strategy
        if self.strategy == ScalingStrategy.AGGRESSIVE:
            # Lower thresholds for more aggressive scaling
            for resource_type, thresholds in base_thresholds.items():
                thresholds['high_threshold'] *= 0.8  # Scale up earlier
                thresholds['low_threshold'] *= 1.2   # Scale down later
        elif self.strategy == ScalingStrategy.CONSERVATIVE:
            # Higher thresholds for more conservative scaling
            for resource_type, thresholds in base_thresholds.items():
                thresholds['high_threshold'] *= 1.1  # Scale up later
                thresholds['low_threshold'] *= 0.8   # Scale down earlier
        
        return base_thresholds
    
    def evaluate_scaling_need(self, current_metrics: ResourceMetrics, 
                           trends: Dict[str, float]) -> List[ScalingRecommendation]:
        """Evaluate if scaling is needed based on current metrics"""
        recommendations = []
        
        # Evaluate CPU scaling
        cpu_recommendation = self._evaluate_cpu_scaling(current_metrics, trends)
        if cpu_recommendation:
            recommendations.append(cpu_recommendation)
        
        # Evaluate memory scaling
        memory_recommendation = self._evaluate_memory_scaling(current_metrics, trends)
        if memory_recommendation:
            recommendations.append(memory_recommendation)
        
        # Evaluate worker scaling
        worker_recommendation = self._evaluate_worker_scaling(current_metrics, trends)
        if worker_recommendation:
            recommendations.append(worker_recommendation)
        
        # Evaluate LLM connection scaling
        llm_recommendation = self._evaluate_llm_scaling(current_metrics, trends)
        if llm_recommendation:
            recommendations.append(llm_recommendation)
        
        return recommendations
    
    def _evaluate_cpu_scaling(self, metrics: ResourceMetrics, trends: Dict[str, float]) -> Optional[ScalingRecommendation]:
        """Evaluate if CPU scaling is needed"""
        cpu_thresholds = self.thresholds[ResourceType.CPU]
        
        # Check for critical condition first
        if metrics.cpu_percent > cpu_thresholds['critical_threshold']:
            return ScalingRecommendation(
                action=ScalingAction.INCREASE,
                resource_type=ResourceType.CPU,
                suggested_change=50.0,  # Increase by 50%
                confidence=0.95,
                reason=f"CPU usage critical: {metrics.cpu_percent}%"
            )
        
        # Check for high usage
        if metrics.cpu_percent > cpu_thresholds['high_threshold']:
            # Consider trend
            cpu_trend = trends.get('cpu_trend', 0)
            if cpu_trend > 0:
                # CPU usage is increasing, scale up
                change = min(30.0, max(10.0, cpu_trend * 2))  # Scale by trend factor
                return ScalingRecommendation(
                    action=ScalingAction.INCREASE,
                    resource_type=ResourceType.CPU,
                    suggested_change=change,
                    confidence=0.8,
                    reason=f"High CPU usage: {metrics.cpu_percent}%, increasing trend: {cpu_trend:.2f}%/min"
                )
            else:
                # High but stable, moderate scale up
                return ScalingRecommendation(
                    action=ScalingAction.INCREASE,
                    resource_type=ResourceType.CPU,
                    suggested_change=15.0,
                    confidence=0.7,
                    reason=f"High CPU usage: {metrics.cpu_percent}%"
                )
        
        # Check for low usage
        if metrics.cpu_percent < cpu_thresholds['low_threshold']:
            cpu_trend = trends.get('cpu_trend', 0)
            if cpu_trend < 0:
                # CPU usage is decreasing, scale down
                change = min(20.0, max(5.0, abs(cpu_trend) * 2))
                return ScalingRecommendation(
                    action=ScalingAction.DECREASE,
                    resource_type=ResourceType.CPU,
                    suggested_change=change,
                    confidence=0.7,
                    reason=f"Low CPU usage: {metrics.cpu_percent}%, decreasing trend: {cpu_trend:.2f}%/min"
                )
            else:
                # Low but stable, conservative scale down
                return ScalingRecommendation(
                    action=ScalingAction.DECREASE,
                    resource_type=ResourceType.CPU,
                    suggested_change=10.0,
                    confidence=0.6,
                    reason=f"Low CPU usage: {metrics.cpu_percent}%"
                )
        
        # No scaling needed
        return None
    
    def _evaluate_memory_scaling(self, metrics: ResourceMetrics, trends: Dict[str, float]) -> Optional[ScalingRecommendation]:
        """Evaluate if memory scaling is needed"""
        memory_thresholds = self.thresholds[ResourceType.MEMORY]
        
        # Check for critical condition
        if metrics.memory_percent > memory_thresholds['critical_threshold']:
            return ScalingRecommendation(
                action=ScalingAction.ALERT,
                resource_type=ResourceType.MEMORY,
                suggested_change=0.0,
                confidence=0.95,
                reason=f"Memory usage critical: {metrics.memory_percent}%"
            )
        
        # Check for high usage
        if metrics.memory_percent > memory_thresholds['high_threshold']:
            memory_trend = trends.get('memory_trend', 0)
            if memory_trend > 0:
                return ScalingRecommendation(
                    action=ScalingAction.INCREASE,
                    resource_type=ResourceType.MEMORY,
                    suggested_change=25.0,
                    confidence=0.8,
                    reason=f"High memory usage: {metrics.memory_percent}%, increasing trend: {memory_trend:.2f}%/min"
                )
        
        # Check for low usage
        if metrics.memory_percent < memory_thresholds['low_threshold']:
            memory_trend = trends.get('memory_trend', 0)
            if memory_trend < 0:
                return ScalingRecommendation(
                    action=ScalingAction.DECREASE,
                    resource_type=ResourceType.MEMORY,
                    suggested_change=15.0,
                    confidence=0.6,
                    reason=f"Low memory usage: {metrics.memory_percent}%, decreasing trend: {memory_trend:.2f}%/min"
                )
        
        return None
    
    def _evaluate_worker_scaling(self, metrics: ResourceMetrics, trends: Dict[str, float]) -> Optional[ScalingRecommendation]:
        """Evaluate if worker scaling is needed"""
        worker_thresholds = self.thresholds[ResourceType.WORKERS]
        
        # Use queue size and pending tasks as indicators
        workload_indicator = max(metrics.queue_size, metrics.pending_tasks)
        
        # Calculate utilization percentage (relative to current workers)
        utilization = (workload_indicator / max(metrics.active_workers, 1)) * 100
        
        if utilization > worker_thresholds['high_threshold']:
            queue_trend = trends.get('queue_trend', 0)
            pending_trend = trends.get('pending_tasks_trend', 0)
            
            if queue_trend > 0 or pending_trend > 0:
                # Workload is increasing, scale up workers
                suggested_workers = max(1, int(metrics.active_workers * 1.5))  # Increase by 50%
                change_percent = ((suggested_workers - metrics.active_workers) / metrics.active_workers) * 100
                return ScalingRecommendation(
                    action=ScalingAction.INCREASE,
                    resource_type=ResourceType.WORKERS,
                    suggested_change=change_percent,
                    confidence=0.85,
                    reason=f"High workload: queue={metrics.queue_size}, pending={metrics.pending_tasks}, increasing trend"
                )
        
        elif utilization < worker_thresholds['low_threshold'] and metrics.active_workers > 1:
            # Workload is low, consider scaling down
            queue_trend = trends.get('queue_trend', 0)
            pending_trend = trends.get('pending_tasks_trend', 0)
            
            if queue_trend < 0 and pending_trend < 0:
                # Workload is decreasing, scale down workers
                suggested_workers = max(1, int(metrics.active_workers * 0.7))  # Decrease by 30%
                change_percent = ((metrics.active_workers - suggested_workers) / metrics.active_workers) * 100
                return ScalingRecommendation(
                    action=ScalingAction.DECREASE,
                    resource_type=ResourceType.WORKERS,
                    suggested_change=change_percent,
                    confidence=0.7,
                    reason=f"Low workload: queue={metrics.queue_size}, pending={metrics.pending_tasks}, decreasing trend"
                )
        
        return None
    
    def _evaluate_llm_scaling(self, metrics: ResourceMetrics, trends: Dict[str, float]) -> Optional[ScalingRecommendation]:
        """Evaluate if LLM connection scaling is needed"""
        llm_thresholds = self.thresholds[ResourceType.LLM_CONNECTIONS]
        
        # Calculate utilization
        # This is a simplified metric - in reality, you'd have more complex logic
        llm_utilization = (metrics.active_llm_connections / 10.0) * 100  # Assuming max 10 connections
        
        if llm_utilization > llm_thresholds['high_threshold']:
            llm_trend = trends.get('llm_connections_trend', 0)
            if llm_trend > 0:
                # Need more LLM connections
                suggested_change = max(20.0, llm_trend * 5)  # Scale by trend
                return ScalingRecommendation(
                    action=ScalingAction.INCREASE,
                    resource_type=ResourceType.LLM_CONNECTIONS,
                    suggested_change=suggested_change,
                    confidence=0.75,
                    reason=f"High LLM utilization: {llm_utilization}%, increasing trend: {llm_trend:.2f}/min"
                )
        
        elif llm_utilization < llm_thresholds['low_threshold']:
            llm_trend = trends.get('llm_connections_trend', 0)
            if llm_trend < 0:
                # Can reduce LLM connections
                suggested_change = min(15.0, abs(llm_trend) * 3)
                return ScalingRecommendation(
                    action=ScalingAction.DECREASE,
                    resource_type=ResourceType.LLM_CONNECTIONS,
                    suggested_change=suggested_change,
                    confidence=0.65,
                    reason=f"Low LLM utilization: {llm_utilization}%, decreasing trend: {llm_trend:.2f}/min"
                )
        
        return None


class ResourceScaler:
    """Executes scaling actions"""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.logger = logging.getLogger("ResourceScaler")
        self.current_resources = {
            ResourceType.WORKERS: 4,
            ResourceType.LLM_CONNECTIONS: 2,
            ResourceType.CACHE_SIZE: 100,  # MB
            ResourceType.DATABASE_CONNECTIONS: 10
        }
        self.scaling_lock = threading.Lock()
    
    def execute_scaling(self, recommendations: List[ScalingRecommendation]) -> List[bool]:
        """Execute scaling recommendations"""
        results = []
        
        with self.scaling_lock:
            for recommendation in recommendations:
                result = self._execute_single_scaling(recommendation)
                results.append(result)
        
        return results
    
    def _execute_single_scaling(self, recommendation: ScalingRecommendation) -> bool:
        """Execute a single scaling action"""
        try:
            resource_type = recommendation.resource_type
            action = recommendation.action
            suggested_change = recommendation.suggested_change
            
            if action == ScalingAction.MAINTAIN:
                return True  # Nothing to do
            
            elif action == ScalingAction.INCREASE:
                return self._increase_resource(resource_type, suggested_change)
            
            elif action == ScalingAction.DECREASE:
                return self._decrease_resource(resource_type, suggested_change)
            
            elif action == ScalingAction.ALERT:
                self.logger.warning(f"Alert needed for {resource_type}: {recommendation.reason}")
                return True  # Alert handled
            
            else:
                self.logger.error(f"Unknown scaling action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing scaling for {recommendation.resource_type}: {e}")
            return False
    
    def _increase_resource(self, resource_type: ResourceType, percent_increase: float) -> bool:
        """Increase a resource by percentage"""
        try:
            current_value = self.current_resources.get(resource_type, 0)
            
            if resource_type == ResourceType.WORKERS:
                # Increase workers (with limits)
                new_value = max(1, min(50, int(current_value * (1 + percent_increase / 100))))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling up workers from {current_value} to {new_value}")
                
                # In a real implementation, you would actually spin up more workers
                # This is where you'd interact with your task queue system
                return True
            
            elif resource_type == ResourceType.LLM_CONNECTIONS:
                # Increase LLM connections (with limits)
                new_value = max(1, min(20, int(current_value * (1 + percent_increase / 100))))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling up LLM connections from {current_value} to {new_value}")
                return True
            
            elif resource_type == ResourceType.CACHE_SIZE:
                # Increase cache size
                new_value = int(current_value * (1 + percent_increase / 100))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling up cache size from {current_value}MB to {new_value}MB")
                return True
            
            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                # Increase DB connections
                new_value = max(1, min(100, int(current_value * (1 + percent_increase / 100))))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling up DB connections from {current_value} to {new_value}")
                return True
            
            else:
                # For other resource types, just update the tracking value
                new_value = int(current_value * (1 + percent_increase / 100))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Updated {resource_type.value} from {current_value} to {new_value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error increasing {resource_type}: {e}")
            return False
    
    def _decrease_resource(self, resource_type: ResourceType, percent_decrease: float) -> bool:
        """Decrease a resource by percentage"""
        try:
            current_value = self.current_resources.get(resource_type, 0)
            
            if resource_type == ResourceType.WORKERS:
                # Decrease workers (with minimum)
                new_value = max(1, int(current_value * (1 - percent_decrease / 100)))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling down workers from {current_value} to {new_value}")
                
                # In a real implementation, you would actually reduce workers
                # This is where you'd interact with your task queue system
                return True
            
            elif resource_type == ResourceType.LLM_CONNECTIONS:
                # Decrease LLM connections (with minimum)
                new_value = max(1, int(current_value * (1 - percent_decrease / 100)))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling down LLM connections from {current_value} to {new_value}")
                return True
            
            elif resource_type == ResourceType.CACHE_SIZE:
                # Decrease cache size (with minimum)
                new_value = max(10, int(current_value * (1 - percent_decrease / 100)))  # Minimum 10MB
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling down cache size from {current_value}MB to {new_value}MB")
                return True
            
            elif resource_type == ResourceType.DATABASE_CONNECTIONS:
                # Decrease DB connections (with minimum)
                new_value = max(1, int(current_value * (1 - percent_decrease / 100)))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Scaling down DB connections from {current_value} to {new_value}")
                return True
            
            else:
                # For other resource types, just update the tracking value
                new_value = max(1, int(current_value * (1 - percent_decrease / 100)))
                self.current_resources[resource_type] = new_value
                self.logger.info(f"Updated {resource_type.value} from {current_value} to {new_value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error decreasing {resource_type}: {e}")
            return False
    
    def get_current_resources(self) -> Dict[ResourceType, int]:
        """Get current resource allocations"""
        return self.current_resources.copy()


class DynamicScaler:
    """Main dynamic scaling system"""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID, 
                 monitor_interval: float = 5.0):
        self.policy = ScalingPolicy(strategy)
        self.monitor = ResourceMonitor()
        self.scaler = ResourceScaler(self.policy)
        self.monitor_interval = monitor_interval
        self.scaling_enabled = False
        self.scaling_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("DynamicScaler")
    
    def start_scaling(self):
        """Start the dynamic scaling system"""
        if not self.scaling_enabled:
            self.scaling_enabled = True
            self.monitor.start_monitoring(self.monitor_interval)
            self.scaling_task = asyncio.create_task(self._scaling_loop())
            self.logger.info("Dynamic scaling system started")
    
    def stop_scaling(self):
        """Stop the dynamic scaling system"""
        self.scaling_enabled = False
        self.monitor.stop_monitoring()
        if self.scaling_task:
            self.scaling_task.cancel()
        self.logger.info("Dynamic scaling system stopped")
    
    async def _scaling_loop(self):
        """Main scaling loop"""
        while self.scaling_enabled:
            try:
                # Wait for some metrics to accumulate
                await asyncio.sleep(self.monitor_interval * 2)
                
                # Get current metrics
                current_metrics = self.monitor.get_average_metrics(window_minutes=1)
                if current_metrics is None:
                    continue
                
                # Get trend metrics
                trends = self.monitor.get_trend_metrics(window_minutes=5)
                
                # Evaluate scaling needs
                recommendations = self.policy.evaluate_scaling_need(current_metrics, trends)
                
                # Execute scaling if needed
                if recommendations:
                    self.logger.info(f"Executing {len(recommendations)} scaling recommendations")
                    results = self.scaler.execute_scaling(recommendations)
                    
                    # Log results
                    for i, (rec, result) in enumerate(zip(recommendations, results)):
                        status = "SUCCESS" if result else "FAILED"
                        self.logger.info(f"Scaling action {i+1}: {status} - {rec.action.value} {rec.resource_type.value} by {rec.suggested_change:.1f}%")
                
                # Wait before next evaluation
                await asyncio.sleep(self.monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                await asyncio.sleep(self.monitor_interval)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        current_metrics = self.monitor.get_average_metrics(window_minutes=1)
        current_resources = self.scaler.get_current_resources()
        
        return {
            "scaling_enabled": self.scaling_enabled,
            "current_metrics": current_metrics.to_dict() if current_metrics else None,
            "current_resources": {k.value: v for k, v in current_resources.items()},
            "last_recommendations": [],  # Would track recent recommendations in a real implementation
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def manual_scale(self, resource_type: ResourceType, action: ScalingAction, 
                          change_percent: float) -> bool:
        """Manually trigger a scaling action"""
        recommendation = ScalingRecommendation(
            action=action,
            resource_type=resource_type,
            suggested_change=change_percent,
            confidence=1.0,
            reason="Manual scaling request"
        )
        
        return self.scaler.execute_scaling([recommendation])[0]


# Global scaler instance
dynamic_scaler: Optional[DynamicScaler] = None


def init_scaling_system(strategy: ScalingStrategy = ScalingStrategy.HYBRID) -> DynamicScaler:
    """Initialize the dynamic scaling system"""
    global dynamic_scaler
    
    dynamic_scaler = DynamicScaler(strategy)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    return dynamic_scaler


def get_scaler() -> Optional[DynamicScaler]:
    """Get the global scaler instance"""
    return dynamic_scaler


def start_scaling_system(strategy: ScalingStrategy = ScalingStrategy.HYBRID):
    """Start the scaling system"""
    scaler = init_scaling_system(strategy)
    scaler.start_scaling()
    return scaler


def stop_scaling_system():
    """Stop the scaling system"""
    global dynamic_scaler
    if dynamic_scaler:
        dynamic_scaler.stop_scaling()
        dynamic_scaler = None


# Example usage and testing
async def test_scaling_system():
    """Test the scaling system"""
    print("Testing ZeroGravity Dynamic Scaling System...")
    
    # Initialize and start the scaler
    scaler = start_scaling_system(ScalingStrategy.HYBRID)
    
    print(f"Scaling system initialized with strategy: {scaler.policy.strategy.value}")
    
    # Get initial status
    status = scaler.get_current_status()
    print(f"Initial status: {status['current_resources']}")
    
    # Simulate some scaling actions
    await asyncio.sleep(2)  # Wait for metrics to collect
    
    # Try manual scaling
    success = await scaler.manual_scale(
        ResourceType.WORKERS, 
        ScalingAction.INCREASE, 
        25.0
    )
    print(f"Manual scale request result: {success}")
    
    # Get updated status
    updated_status = scaler.get_current_status()
    print(f"Updated status: {updated_status['current_resources']}")
    
    # Let it run for a bit to see auto-scaling in action
    print("Running auto-scaling for 30 seconds...")
    await asyncio.sleep(30)
    
    # Stop the system
    stop_scaling_system()
    print("Scaling system stopped")


if __name__ == "__main__":
    # For testing purposes
    print("Starting ZeroGravity Dynamic Scaling System example...")
    # asyncio.run(test_scaling_system())
