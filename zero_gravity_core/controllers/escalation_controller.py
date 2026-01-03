# Escalation Controller for AI Ops Agent
# Handles human-in-the-loop mechanisms, alerts, and escalation triggers

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import asdict
from pathlib import Path
import aiofiles

from zero_gravity_core.schemas.ai_ops_schemas import EscalationEvent, PolicyViolation

logger = logging.getLogger(__name__)

class EscalationController:
    """
    The Escalation Controller manages:
    - Escalation triggers (policy violations, repeated failures, etc.)
    - Human-in-the-loop mechanisms
    - Alert systems
    - Kill switch functionality
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "zero_gravity_core/escalation_data"
        self.escalation_history: List[EscalationEvent] = []
        self.active_escalations: Dict[str, EscalationEvent] = {}
        self.escalation_callbacks: Dict[str, List[Callable]] = {}
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Define escalation triggers and thresholds
        self.escalation_triggers = {
            "policy_violation": {
                "critical": ["block", "escalate"],
                "high_threshold": 1, # Any critical policy violation escalates
                "medium_threshold": 3  # 3+ medium violations escalate
            },
            "repeated_failures": {
                "threshold": 3  # 3+ failures in a row escalate
            },
            "high_cost": {
                "threshold": 5.0 # $5+ per task escalates
            },
            "legal_financial_risk": {
                "keywords": ["legal", "financial", "compliance", "regulatory", "audit", "contract", "agreement"]
            },
            "ambiguous_output": {
                "low_quality_threshold": 0.3  # Quality below 30% escalates
            },
            "high_value_decisions": {
                "high_priority_threshold": 9,  # Priority 9+ escalates
                "high_impact_threshold": 1000  # High customer impact escalates
            }
        }
        
        logger.info("Escalation Controller initialized")
    
    async def handle_escalation(self, escalation_event: EscalationEvent):
        """
        Handle an escalation event
        """
        logger.info(f"Handling escalation for agent {escalation_event.agent_name}: {escalation_event.reason}")
        
        # Add to active escalations
        self.active_escalations[escalation_event.task_id or f"escalation_{int(time.time())}"] = escalation_event
        
        # Add to history
        self.escalation_history.append(escalation_event)
        
        # Save escalation to storage
        await self._save_escalation(escalation_event)
        
        # Trigger escalation callbacks if any
        await self._trigger_callbacks(escalation_event)
        
        # Log the escalation based on severity
        if escalation_event.severity == "critical":
            logger.critical(f"CRITICAL ESCALATION: {escalation_event}")
        elif escalation_event.severity == "high":
            logger.error(f"HIGH PRIORITY ESCALATION: {escalation_event}")
        elif escalation_event.severity == "medium":
            logger.warning(f"MEDIUM PRIORITY ESCALATION: {escalation_event}")
        else:
            logger.info(f"LOW PRIORITY ESCALATION: {escalation_event}")
    
    async def check_escalation_triggers(self, agent_name: str, context: Dict[str, Any]) -> List[EscalationEvent]:
        """
        Check if any escalation triggers are activated based on context
        """
        escalations = []
        
        # Check for policy violations
        policy_violations = context.get("policy_violations", [])
        for violation in policy_violations:
            if isinstance(violation, PolicyViolation):
                escalation_needed = await self._check_policy_violation_escalation(violation)
                if escalation_needed:
                    event = EscalationEvent(
                        agent_name=agent_name,
                        violation=violation,
                        context=context,
                        timestamp=time.time(),
                        reason=f"Policy violation: {violation.policy_name}",
                        severity=violation.severity
                    )
                    escalations.append(event)
        
        # Check for repeated failures
        failure_count = context.get("failure_count", 0)
        if failure_count >= self.escalation_triggers["repeated_failures"]["threshold"]:
            event = EscalationEvent(
                agent_name=agent_name,
                violation=None,
                context=context,
                timestamp=time.time(),
                reason=f"Repeated failures: {failure_count} consecutive failures",
                severity="high"
            )
            escalations.append(event)
        
        # Check for high cost
        cost = context.get("cost", 0)
        if cost >= self.escalation_triggers["high_cost"]["threshold"]:
            event = EscalationEvent(
                agent_name=agent_name,
                violation=None,
                context=context,
                timestamp=time.time(),
                reason=f"High cost task: ${cost:.2f}",
                severity="medium"
            )
            escalations.append(event)
        
        # Check for legal/financial risk based on keywords in task description
        task_description = context.get("task_description", "")
        if any(keyword in task_description.lower() for keyword in self.escalation_triggers["legal_financial_risk"]["keywords"]):
            event = EscalationEvent(
                agent_name=agent_name,
                violation=None,
                context=context,
                timestamp=time.time(),
                reason=f"Legal/financial risk detected in task: {task_description[:100]}...",
                severity="high"
            )
            escalations.append(event)
        
        # Check for ambiguous output (low quality)
        quality_score = context.get("quality_score", 1.0)
        if quality_score <= self.escalation_triggers["ambiguous_output"]["low_quality_threshold"]:
            event = EscalationEvent(
                agent_name=agent_name,
                violation=None,
                context=context,
                timestamp=time.time(),
                reason=f"Ambiguous output detected: quality score {quality_score:.2f}",
                severity="medium"
            )
            escalations.append(event)
        
        # Check for high-value decisions
        priority = context.get("priority", 5)
        customer_impact = context.get("customer_impact", "low")
        
        if (priority >= self.escalation_triggers["high_value_decisions"]["high_priority_threshold"] or
            customer_impact == "high"):
            event = EscalationEvent(
                agent_name=agent_name,
                violation=None,
                context=context,
                timestamp=time.time(),
                reason=f"High-value decision: priority {priority}, impact {customer_impact}",
                severity="high"
            )
            escalations.append(event)
        
        return escalations
    
    async def _check_policy_violation_escalation(self, violation: PolicyViolation) -> bool:
        """
        Check if a policy violation should trigger escalation
        """
        # Critical actions always escalate
        if violation.action in self.escalation_triggers["policy_violation"]["critical"]:
            return True
        
        # For other violations, check severity
        return violation.severity in ["high", "critical"]
    
    async def _save_escalation(self, escalation_event: EscalationEvent):
        """
        Save escalation event to storage
        """
        file_path = Path(self.storage_path) / "escalations.jsonl"
        
        async with aiofiles.open(file_path, 'a') as f:
            # Convert escalation event to dict, handling the violation field specially
            event_dict = asdict(escalation_event)
            # Convert violation to string if it's a complex object
            if event_dict["violation"] and hasattr(event_dict["violation"], '__dict__'):
                event_dict["violation"] = str(event_dict["violation"])
            
            await f.write(json.dumps(event_dict) + '\n')
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback function for specific escalation events
        """
        if event_type not in self.escalation_callbacks:
            self.escalation_callbacks[event_type] = []
        
        self.escalation_callbacks[event_type].append(callback)
        logger.info(f"Registered callback for event type: {event_type}")
    
    async def _trigger_callbacks(self, escalation_event: EscalationEvent):
        """
        Trigger registered callbacks for the escalation event
        """
        # Trigger callbacks for the severity level
        if escalation_event.severity in self.escalation_callbacks:
            for callback in self.escalation_callbacks[escalation_event.severity]:
                try:
                    await callback(escalation_event)
                except Exception as e:
                    logger.error(f"Error in escalation callback: {str(e)}")
        
        # Trigger callbacks for 'all' event type
        if "all" in self.escalation_callbacks:
            for callback in self.escalation_callbacks["all"]:
                try:
                    await callback(escalation_event)
                except Exception as e:
                    logger.error(f"Error in escalation callback: {str(e)}")
    
    async def get_active_escalations(self) -> Dict[str, EscalationEvent]:
        """
        Get all active escalations
        """
        return self.active_escalations
    
    async def get_escalation_history(self, limit: Optional[int] = None) -> List[EscalationEvent]:
        """
        Get escalation history, optionally limited to last N entries
        """
        if limit:
            return self.escalation_history[-limit:]
        return self.escalation_history
    
    async def resolve_escalation(self, escalation_id: str, resolution_notes: str = ""):
        """
        Mark an escalation as resolved
        """
        if escalation_id in self.active_escalations:
            escalation = self.active_escalations[escalation_id]
            logger.info(f"Resolving escalation {escalation_id}: {resolution_notes}")
            
            # In a real implementation, you might add resolution details to the event
            # For now, we'll just remove it from active escalations
            del self.active_escalations[escalation_id]
            
            # Log resolution
            logger.info(f"Escalation {escalation_id} resolved: {resolution_notes}")
        else:
            logger.warning(f"Attempted to resolve non-existent escalation: {escalation_id}")
    
    async def get_escalation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about escalations
        """
        total_escalations = len(self.escalation_history)
        active_escalations = len(self.active_escalations)
        
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for event in self.escalation_history:
            if event.severity in severity_counts:
                severity_counts[event.severity] += 1
        
        # Count by agent
        agent_counts = {}
        for event in self.escalation_history:
            agent = event.agent_name
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        # Calculate escalation rate (per day)
        if self.escalation_history:
            oldest_timestamp = min(event.timestamp for event in self.escalation_history)
            time_span_days = (time.time() - oldest_timestamp) / (24 * 3600)  # Convert to days
            avg_daily_escalations = total_escalations / max(time_span_days, 1)
        else:
            avg_daily_escalations = 0
        
        return {
            "total_escalations": total_escalations,
            "active_escalations": active_escalations,
            "severity_distribution": severity_counts,
            "agent_distribution": agent_counts,
            "average_daily_escalations": avg_daily_escalations,
            "escalation_rate_trend": "decreasing" if avg_daily_escalations < 1 else "stable"  # Simplified
        }
    
    async def apply_kill_switch(self, agent_name: str, reason: str = "Manual override"):
        """
        Apply kill switch to stop an agent
        """
        logger.critical(f"KILL SWITCH ACTIVATED for agent {agent_name}: {reason}")
        
        # Create escalation event for the kill switch
        kill_switch_event = EscalationEvent(
            agent_name=agent_name,
            violation=None,
            context={"action": "kill_switch", "reason": reason},
            timestamp=time.time(),
            reason=f"Kill switch activated: {reason}",
            severity="critical"
        )
        
        # Handle the escalation
        await self.handle_escalation(kill_switch_event)
        
        # In a real implementation, this would actually stop the agent
        # For now, we'll just log it
    
    async def get_escalation_recommendations(self, agent_name: str, current_context: Dict[str, Any]) -> List[str]:
        """
        Get recommendations to prevent escalations for a specific agent
        """
        recommendations = []
        
        # Check if this agent has a high escalation rate
        agent_escalations = [e for e in self.escalation_history if e.agent_name == agent_name]
        total_agent_tasks = current_context.get("total_tasks", 1)
        escalation_rate = len(agent_escalations) / total_agent_tasks
        
        if escalation_rate > 0.1:  # More than 10% of tasks result in escalation
            recommendations.append(f"Agent {agent_name} has high escalation rate ({escalation_rate:.2%}). Consider retraining or reassigning tasks.")
        
        # Check common triggers for this agent
        recent_context = current_context.get("recent_context", {})
        if recent_context.get("failure_count", 0) >= 2:
            recommendations.append(f"Avoid further failures for {agent_name}, approaching escalation threshold.")
        
        if recent_context.get("cost", 0) >= self.escalation_triggers["high_cost"]["threshold"] * 0.8:
            recommendations.append(f"Task cost approaching escalation threshold for {agent_name}.")
        
        return recommendations
    
    async def setup_default_alerting(self):
        """
        Set up default alerting mechanisms
        """
        # Register default callbacks for different severity levels
        
        async def critical_alert_handler(event: EscalationEvent):
            logger.critical(f"CRITICAL ALERT: Escalation requires immediate attention - {event.reason}")
            # In a real system, this might send an email, Slack message, etc.
        
        async def high_alert_handler(event: EscalationEvent):
            logger.error(f"HIGH PRIORITY ALERT: Review needed - {event.reason}")
            # In a real system, this might send a notification
            pass
        
        async def medium_alert_handler(event: EscalationEvent):
            logger.warning(f"MEDIUM PRIORITY ALERT: Consider review - {event.reason}")
            # In a real system, this might log to a dashboard
            pass
        
        # Register the alert handlers
        self.register_callback("critical", critical_alert_handler)
        self.register_callback("high", high_alert_handler)
        self.register_callback("medium", medium_alert_handler)
        
        logger.info("Default alerting mechanisms set up")
    
    async def shutdown(self):
        """
        Clean shutdown of escalation controller
        """
        logger.info("Shutting down Escalation Controller...")
        
        # Flush any remaining data
        # In a real implementation, ensure all pending escalations are saved
        
        logger.info("Escalation Controller shutdown complete")
