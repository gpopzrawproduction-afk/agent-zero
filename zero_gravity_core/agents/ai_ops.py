# AI Ops Agent
# Autonomous orchestration, optimization, governance, and supervision of all other agents and workflows

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from zero_gravity_core.agents.base import BaseAgent
from zero_gravity_core.engines.decision_engine import DecisionEngine
from zero_gravity_core.engines.policy_engine import PolicyEngine
from zero_gravity_core.engines.optimization_engine import OptimizationEngine
from zero_gravity_core.controllers.escalation_controller import EscalationController
from zero_gravity_core.monitoring.telemetry_collector import TelemetryCollector, MetricData
from zero_gravity_core.schemas.ai_ops_schemas import (
    AgentPerformance, 
    PolicyConfig, 
    OptimizationRecommendation,
    EscalationEvent
)
from zero_gravity_core.config.ai_ops_config import AIOpsConfig, get_ai_ops_config

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"

@dataclass
class TaskRequest:
    """Represents a request from another agent to execute a task"""
    task_id: str
    agent_name: str
    task_description: str
    priority: int  # 1-10 scale
    estimated_cost: float
    estimated_time: float  # in seconds
    complexity: str  # 'low', 'medium', 'high'
    required_model: Optional[str] = None

@dataclass
class TaskApproval:
    """Result of AI Ops approval decision"""
    approved: bool
    model_selection: str
    priority_adjustment: int
    cost_limit: float
    time_limit: float
    escalation_required: bool
    optimization_recommendation: Optional[str] = None

class AIOpsAgent(BaseAgent):
    """
    AI Ops Agent - The meta-agent that orchestrates, optimizes, governs, and supervises 
    all other agents and workflows in the ZeroGravity system.
    
    Core Responsibilities:
    - Orchestration Control: Decides which agent runs, in what order, with what priority
    - Performance Monitoring: Continuously monitors agent performance metrics
    - Policy Enforcement: Enforces budget, compliance, and security policies
    - Continuous Optimization: Dynamically adjusts models, prompts, and workflows
    """
    
    def __init__(self, name: str = "ai_ops", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        
        # Load AI Ops configuration
        import asyncio
        try:
            # Try to get the config asynchronously
            loop = asyncio.get_event_loop()
            self.ai_ops_config = loop.run_until_complete(get_ai_ops_config())
        except RuntimeError:
            # If no event loop is running, create a temporary one
            import asyncio
            temp_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(temp_loop)
            self.ai_ops_config = temp_loop.run_until_complete(get_ai_ops_config())
            temp_loop.close()
        
        # Initialize internal modules with configuration
        self.telemetry_collector = TelemetryCollector(
            storage_path=self.ai_ops_config.telemetry_storage_path
        ) if self.ai_ops_config.telemetry_enabled else None
        
        self.decision_engine = DecisionEngine() if self.ai_ops_config.decision_engine_enabled else None
        self.policy_engine = PolicyEngine() if self.ai_ops_config.policy_engine_enabled else None
        self.optimization_engine = OptimizationEngine(
            storage_path=self.ai_ops_config.optimization_storage_path
        ) if self.ai_ops_config.optimization_engine_enabled else None
        self.escalation_controller = EscalationController(
            storage_path=self.ai_ops_config.escalation_storage_path
        ) if self.ai_ops_config.escalation_controller_enabled else None
        
        # Task tracking
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_approvals: Dict[str, TaskApproval] = {}
        self.agent_performance: Dict[str, AgentPerformance] = {}
        
        # Initialize default policies if policy engine is enabled
        if self.policy_engine:
            self._initialize_default_policies()
        
        logger.info("AI Ops Agent initialized successfully")
    
    def _initialize_default_policies(self):
        """Initialize default policies for the system"""
        default_policies = [
            PolicyConfig(
                name="max_cost_per_task",
                agent="*",
                condition="cost > 1.00",
                action="downgrade_model",
                description="Downgrade model if task cost exceeds $1.00"
            ),
            PolicyConfig(
                name="max_retries",
                agent="*",
                condition="retry_count > 3",
                action="escalate",
                description="Escalate if task fails more than 3 times"
            ),
            PolicyConfig(
                name="priority_threshold",
                agent="*",
                condition="priority < 3",
                action="queue_for_batch_processing",
                description="Queue low priority tasks for batch processing"
            )
        ]
        
        for policy in default_policies:
            self.policy_engine.add_policy(policy)
    
    async def approve_task(self, task_request: TaskRequest) -> TaskApproval:
        """
        Main orchestration method - decides if a task should be approved and under what conditions
        """
        logger.info(f"Processing task approval request: {task_request.task_id} from {task_request.agent_name}")
        
        # Store the task request
        self.active_tasks[task_request.task_id] = task_request
        
        # Check policies first
        policy_violations = await self.policy_engine.check_policies(
            agent_name=task_request.agent_name,
            task_data=asdict(task_request)
        )
        
        if policy_violations:
            logger.warning(f"Policy violations detected for task {task_request.task_id}: {policy_violations}")
            
            # Handle policy violations based on policy actions
            escalation_required = False
            for violation in policy_violations:
                if violation.action == "escalate":
                    escalation_required = True
                elif violation.action == "downgrade_model":
                    # For now, just log - actual model downgrading happens in decision engine
                    logger.info(f"Model downgrading required for task {task_request.task_id}")
            
            return TaskApproval(
                approved=not any(v.action == "block" for v in policy_violations),
                model_selection="gpt-3.5-turbo",  # Default to cheaper model if violations exist
                priority_adjustment=task_request.priority,
                cost_limit=min(task_request.estimated_cost, 0.50),  # Cap cost if violations exist
                time_limit=task_request.estimated_time,
                escalation_required=escalation_required
            )
        
        # Use decision engine to make approval decision
        decision = await self.decision_engine.make_decision(
            task_request=task_request,
            agent_performance=self.agent_performance.get(task_request.agent_name)
        )
        
        # Create approval result
        approval = TaskApproval(
            approved=decision.get('approved', False),
            model_selection=decision.get('model_selection', 'gpt-4'),
            priority_adjustment=decision.get('priority_adjustment', task_request.priority),
            cost_limit=decision.get('cost_limit', task_request.estimated_cost * 2),
            time_limit=decision.get('time_limit', task_request.estimated_time * 2),
            escalation_required=decision.get('escalation_required', False),
            optimization_recommendation=decision.get('optimization_recommendation')
        )
        
        # Store the approval for reference
        self.task_approvals[task_request.task_id] = approval
        
        logger.info(f"Task {task_request.task_id} approved: {approval.approved}, model: {approval.model_selection}")
        return approval
    
    async def monitor_task_execution(self, task_id: str, agent_name: str) -> Dict[str, Any]:
        """
        Monitor task execution and collect telemetry
        """
        logger.info(f"Monitoring task execution: {task_id} by {agent_name}")
        
        start_time = time.time()
        task_status = AgentStatus.RUNNING
        
        try:
            # Simulate task execution monitoring
            # In real implementation, this would connect to actual agent execution
            await asyncio.sleep(0.1)  # Simulate monitoring delay
            
            # Collect telemetry data
            execution_time = time.time() - start_time
            cost = 0.05  # Placeholder - would come from actual execution
            
            # Create metric data
            metric_data = MetricData(
                agent=agent_name,
                task_id=task_id,
                latency_ms=execution_time * 1000,
                cost_usd=cost,
                success=True,  # Placeholder - would come from actual execution
                quality_score=0.9,  # Placeholder - would come from evaluation
                model="gpt-4",  # Would come from actual execution
                timestamp=time.time()
            )
            
            # Emit telemetry
            await self.telemetry_collector.emit_telemetry(metric_data)
            
            # Update agent performance
            await self._update_agent_performance(agent_name, metric_data)
            
            # Check for optimization opportunities
            await self._check_optimization_opportunities(agent_name, metric_data)
            
            task_status = AgentStatus.COMPLETED
            
            return {
                "status": "completed",
                "execution_time": execution_time,
                "cost": cost,
                "quality_score": 0.9
            }
            
        except Exception as e:
            logger.error(f"Error monitoring task {task_id}: {str(e)}")
            task_status = AgentStatus.FAILED
            
            # Log failure telemetry
            failure_metric = MetricData(
                agent=agent_name,
                task_id=task_id,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.02,  # Partial cost for failed execution
                success=False,
                quality_score=0.0,
                model="gpt-4",
                timestamp=time.time(),
                error=str(e)
            )
            
            await self.telemetry_collector.emit_telemetry(failure_metric)
            await self._update_agent_performance(agent_name, failure_metric)
            
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "cost": 0.02
            }
        finally:
            # Update task status
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _update_agent_performance(self, agent_name: str, metric_data: MetricData):
        """Update agent performance metrics based on execution data"""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = AgentPerformance(
                agent_name=agent_name,
                total_tasks=0,
                successful_tasks=0,
                total_cost=0.0,
                total_latency=0.0,
                average_quality=0.0
            )
        
        perf = self.agent_performance[agent_name]
        perf.total_tasks += 1
        
        if metric_data.success:
            perf.successful_tasks += 1
        
        perf.total_cost += metric_data.cost_usd
        perf.total_latency += metric_data.latency_ms
        perf.average_quality = ((perf.average_quality * (perf.total_tasks - 1)) + metric_data.quality_score) / perf.total_tasks
    
    async def _check_optimization_opportunities(self, agent_name: str, metric_data: MetricData):
        """Check for optimization opportunities based on performance data"""
        if self.optimization_engine:
            # Use optimization engine to analyze performance and suggest improvements
            recommendation = await self.optimization_engine.analyze_performance(
                agent_name=agent_name,
                metric_data=metric_data,
                current_performance=self.agent_performance.get(agent_name)
            )
            
            if recommendation:
                logger.info(f"Optimization recommendation for {agent_name}: {recommendation}")
                # Apply optimization if appropriate
                await self.optimization_engine.apply_optimization(recommendation)
    
    async def enforce_policy(self, agent_name: str, action: str, context: Dict[str, Any]) -> bool:
        """
        Enforce policies for a specific agent action
        """
        violations = await self.policy_engine.check_policies(
            agent_name=agent_name,
            task_data=context
        )
        
        if violations:
            logger.warning(f"Policy violations for {agent_name} action '{action}': {violations}")
            
            # Handle violations based on policy actions
            for violation in violations:
                if violation.action == "escalate":
                    escalation_event = EscalationEvent(
                        agent_name=agent_name,
                        violation=violation,
                        context=context,
                        timestamp=time.time()
                    )
                    await self.escalation_controller.handle_escalation(escalation_event)
                elif violation.action == "block":
                    logger.error(f"Blocking action '{action}' for {agent_name} due to policy violation")
                    return False
        
        return True
    
    async def optimize_workflow(self, workflow_id: str, current_agents: List[str]) -> List[str]:
        """
        Optimize workflow by suggesting better agent assignments
        """
        optimized_agents = await self.optimization_engine.optimize_workflow(
            workflow_id=workflow_id,
            current_agents=current_agents,
            agent_performance=self.agent_performance
        )
        
        logger.info(f"Workflow {workflow_id} optimized: {current_agents} -> {optimized_agents}")
        return optimized_agents
    
    async def get_agent_kpis(self, agent_name: str) -> Dict[str, float]:
        """
        Get KPIs for a specific agent
        """
        if agent_name not in self.agent_performance:
            return {
                "cost_efficiency_score": 0.0,
                "reliability_score": 0.0,
                "quality_score": 0.0,
                "speed_score": 0.0
            }
        
        perf = self.agent_performance[agent_name]
        
        # Calculate KPIs
        success_rate = perf.successful_tasks / max(perf.total_tasks, 1)
        avg_cost = perf.total_cost / max(perf.total_tasks, 1)
        avg_latency = perf.total_latency / max(perf.total_tasks, 1)
        avg_quality = perf.average_quality
        
        # Normalize scores (0-1 scale)
        cost_efficiency_score = max(0, min(1, 1 / (avg_cost * 10)))  # Lower cost = higher score
        reliability_score = success_rate
        quality_score = avg_quality
        speed_score = max(0, min(1, 1000 / max(avg_latency, 1)))  # Lower latency = higher score
        
        return {
            "cost_efficiency_score": cost_efficiency_score,
            "reliability_score": reliability_score,
            "quality_score": quality_score,
            "speed_score": speed_score
        }
    
    async def get_system_kpis(self) -> Dict[str, float]:
        """
        Get system-level KPIs
        """
        total_tasks = sum(perf.total_tasks for perf in self.agent_performance.values())
        successful_tasks = sum(perf.successful_tasks for perf in self.agent_performance.values())
        total_cost = sum(perf.total_cost for perf in self.agent_performance.values())
        total_latency = sum(perf.total_latency for perf in self.agent_performance.values())
        
        success_rate = successful_tasks / max(total_tasks, 1)
        avg_cost_per_task = total_cost / max(total_tasks, 1)
        avg_time_to_resolution = total_latency / max(successful_tasks, 1) if successful_tasks > 0 else 0
        
        # Calculate human escalation rate
        total_escalations = len(self.escalation_controller.escalation_history)
        human_escalation_rate = total_escalations / max(total_tasks, 1)
        
        return {
            "cost_per_completed_objective": avg_cost_per_task,
            "success_rate_per_workflow": success_rate,
            "mean_time_to_resolution_ms": avg_time_to_resolution,
            "human_escalation_rate": human_escalation_rate
        }
    
    async def shutdown(self):
        """Clean shutdown of AI Ops Agent"""
        logger.info("Shutting down AI Ops Agent...")
        
        # Flush any remaining telemetry (if enabled)
        if self.telemetry_collector:
            await self.telemetry_collector.flush()
        
        # Save optimization learnings (if enabled)
        if self.optimization_engine:
            await self.optimization_engine.save_learnings()
        
        # Shutdown escalation controller (if enabled)
        if self.escalation_controller:
            await self.escalation_controller.shutdown()
        
        logger.info("AI Ops Agent shutdown complete")
