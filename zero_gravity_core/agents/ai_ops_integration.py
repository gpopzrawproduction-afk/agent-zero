# Integration utilities for AI Ops Agent
# Provides interfaces for existing agents to interact with the AI Ops system

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from zero_gravity_core.agents.ai_ops import AIOpsAgent, TaskRequest, TaskApproval
from zero_gravity_core.schemas.ai_ops_schemas import MetricData

logger = logging.getLogger(__name__)

# Global reference to the AI Ops agent instance
_ai_ops_agent: Optional[AIOpsAgent] = None

async def initialize_ai_ops_agent():
    """
    Initialize the AI Ops agent instance
    """
    global _ai_ops_agent
    if _ai_ops_agent is None:
        _ai_ops_agent = AIOpsAgent()
        logger.info("AI Ops Agent initialized")
    return _ai_ops_agent

async def get_ai_ops_agent() -> Optional[AIOpsAgent]:
    """
    Get the AI Ops agent instance
    """
    global _ai_ops_agent
    if _ai_ops_agent is None:
        await initialize_ai_ops_agent()
    return _ai_ops_agent

async def request_task_approval(
    agent_name: str,
    task_description: str,
    priority: int = 5,
    estimated_cost: float = 0.1,
    estimated_time: float = 60.0,
    complexity: str = "medium",
    required_model: Optional[str] = None
) -> Optional[TaskApproval]:
    """
    Request approval for a task from the AI Ops agent
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops:
        logger.warning("AI Ops agent not available, proceeding without approval")
        # Return default approval if AI Ops is not available
        return TaskApproval(
            approved=True,
            model_selection=required_model or "gpt-4",
            priority_adjustment=priority,
            cost_limit=estimated_cost * 2,
            time_limit=estimated_time * 2,
            escalation_required=False
        )
    
    task_id = f"{agent_name}_{int(datetime.now().timestamp())}"
    
    task_request = TaskRequest(
        task_id=task_id,
        agent_name=agent_name,
        task_description=task_description,
        priority=priority,
        estimated_cost=estimated_cost,
        estimated_time=estimated_time,
        complexity=complexity,
        required_model=required_model
    )
    
    try:
        approval = await ai_ops.approve_task(task_request)
        logger.info(f"Task approval received for {task_id}: {approval.approved}")
        return approval
    except Exception as e:
        logger.error(f"Error requesting task approval: {str(e)}")
        # Return default approval if there's an error
        return TaskApproval(
            approved=True,
            model_selection=required_model or "gpt-4",
            priority_adjustment=priority,
            cost_limit=estimated_cost * 2,
            time_limit=estimated_time * 2,
            escalation_required=False
        )

async def monitor_task_execution(
    task_id: str,
    agent_name: str
) -> Dict[str, Any]:
    """
    Monitor task execution through the AI Ops agent
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops:
        logger.warning("AI Ops agent not available, skipping monitoring")
        return {"status": "completed", "cost": 0.05, "quality_score": 0.9}
    
    try:
        result = await ai_ops.monitor_task_execution(task_id, agent_name)
        return result
    except Exception as e:
        logger.error(f"Error monitoring task execution: {str(e)}")
        return {"status": "error", "error": str(e)}

async def report_task_metrics(
    agent_name: str,
    task_id: str,
    latency_ms: float,
    cost_usd: float,
    success: bool,
    quality_score: float,
    model_used: str,
    error: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    retries: int = 0,
    priority: Optional[int] = None,
    task_complexity: Optional[str] = None,
    customer_impact: Optional[str] = None
) -> bool:
    """
    Report task metrics to the AI Ops system for performance tracking
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops or not ai_ops.telemetry_collector:
        logger.warning("AI Ops agent or telemetry collector not available, skipping metrics report")
        return False
    
    try:
        metric_data = MetricData(
            agent=agent_name,
            task_id=task_id,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=success,
            quality_score=quality_score,
            model=model_used,
            timestamp=asyncio.get_event_loop().time(),
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retries=retries,
            priority=priority,
            task_complexity=task_complexity,
            customer_impact=customer_impact
        )
        
        await ai_ops.telemetry_collector.emit_telemetry(metric_data)
        logger.debug(f"Metrics reported for task {task_id}")
        return True
    except Exception as e:
        logger.error(f"Error reporting task metrics: {str(e)}")
        return False

async def enforce_policy(agent_name: str, action: str, context: Dict[str, Any]) -> bool:
    """
    Enforce policies for an agent action through the AI Ops system
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops:
        logger.warning("AI Ops agent not available, proceeding without policy enforcement")
        return True
    
    try:
        result = await ai_ops.enforce_policy(agent_name, action, context)
        return result
    except Exception as e:
        logger.error(f"Error enforcing policy: {str(e)}")
        return True  # Allow action if there's an error

async def optimize_workflow(workflow_id: str, current_agents: List[str]) -> List[str]:
    """
    Optimize workflow agent assignments through the AI Ops system
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops:
        logger.warning("AI Ops agent not available, returning original agents")
        return current_agents
    
    try:
        optimized_agents = await ai_ops.optimize_workflow(workflow_id, current_agents)
        return optimized_agents
    except Exception as e:
        logger.error(f"Error optimizing workflow: {str(e)}")
        return current_agents

async def get_agent_kpis(agent_name: str) -> Dict[str, float]:
    """
    Get KPIs for an agent from the AI Ops system
    """
    ai_ops = await get_ai_ops_agent()
    if not ai_ops:
        logger.warning("AI Ops agent not available, returning default KPIs")
        return {
            "cost_efficiency_score": 0.5,
            "reliability_score": 0.5,
            "quality_score": 0.5,
            "speed_score": 0.5
        }
    
    try:
        kpis = await ai_ops.get_agent_kpis(agent_name)
        return kpis
    except Exception as e:
        logger.error(f"Error getting agent KPIs: {str(e)}")
        return {
            "cost_efficiency_score": 0.5,
            "reliability_score": 0.5,
            "quality_score": 0.5,
            "speed_score": 0.5
        }

async def shutdown_ai_ops_agent():
    """
    Properly shut down the AI Ops agent
    """
    global _ai_ops_agent
    if _ai_ops_agent:
        await _ai_ops_agent.shutdown()
        _ai_ops_agent = None
        logger.info("AI Ops Agent shut down")
