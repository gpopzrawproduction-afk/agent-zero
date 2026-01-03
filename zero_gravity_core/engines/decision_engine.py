# Decision Engine for AI Ops Agent
# Implements core logic for model selection, task routing, priority scheduling, and retry strategies

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from zero_gravity_core.schemas.ai_ops_schemas import (
    TaskRequest, 
    AgentPerformance, 
    DecisionFactors,
    PolicyConfig
)

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    The Decision Engine makes intelligent decisions about:
    - Model selection (cheap vs premium)
    - Agent selection
    - Retry strategy
    - Parallelization
    - Abort conditions
    """
    
    def __init__(self):
        self.model_cost_map = {
            "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
            "gpt-4": 0.03,          # $0.03 per 1K tokens
            "gpt-4-32k": 0.06,      # $0.06 per 1K tokens
            "claude-2": 0.01102,    # $0.01102 per 1K tokens
            "claude-instant-1": 0.00163  # $0.0163 per 1K tokens
        }
        
        # Performance thresholds for decision making
        self.performance_thresholds = {
            "latency_ms": 5000,  # 5 seconds
            "cost_per_task": 1.0,  # $1.00
            "success_rate": 0.85,  # 85%
            "quality_score": 0.8   # 80%
        }
        
        # Model capabilities mapping
        self.model_capabilities = {
            "gpt-3.5-turbo": {
                "coding": 0.7,
                "analysis": 0.6,
                "creativity": 0.6,
                "complexity_handling": 0.5
            },
            "gpt-4": {
                "coding": 0.9,
                "analysis": 0.9,
                "creativity": 0.8,
                "complexity_handling": 0.9
            },
            "gpt-4-32k": {
                "coding": 0.9,
                "analysis": 0.9,
                "creativity": 0.8,
                "complexity_handling": 0.9,
                "context_handling": 0.95
            },
            "claude-2": {
                "coding": 0.8,
                "analysis": 0.9,
                "creativity": 0.8,
                "complexity_handling": 0.85
            },
            "claude-instant-1": {
                "coding": 0.6,
                "analysis": 0.7,
                "creativity": 0.7,
                "complexity_handling": 0.6
            }
        }
    
    async def make_decision(self, task_request: TaskRequest, agent_performance: Optional[AgentPerformance] = None) -> Dict[str, Any]:
        """
        Make a decision about task execution based on various factors
        """
        logger.info(f"Making decision for task: {task_request.task_id}")
        
        # Analyze the task requirements
        task_analysis = await self._analyze_task(task_request)
        
        # Consider historical performance if available
        performance_factors = await self._analyze_performance(agent_performance, task_request.complexity)
        
        # Consider cost constraints
        cost_factors = await self._analyze_cost_constraints(task_request)
        
        # Combine all factors
        decision_factors = DecisionFactors(
            task_priority=task_request.priority,
            historical_performance=agent_performance,
            cost_budget=task_request.estimated_cost * 2 if task_request.estimated_cost else 2.0,
            sla_requirements=self._get_sla_requirements(task_request.complexity),
            current_system_load=await self._get_system_load(),
            agent_availability=["architect", "engineer", "designer", "operator", "coordinator"],
            model_availability=list(self.model_cost_map.keys())
        )
        
        # Make the decision
        decision = await self._apply_decision_logic(task_request, decision_factors)
        
        logger.info(f"Decision made for task {task_request.task_id}: {decision}")
        return decision
    
    async def _analyze_task(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Analyze the specific task requirements"""
        analysis = {
            "complexity_score": self._map_complexity_to_score(task_request.complexity),
            "priority_score": task_request.priority / 10.0,  # Normalize to 0-1 scale
            "estimated_cost": task_request.estimated_cost,
            "estimated_time": task_request.estimated_time,
            "required_capabilities": self._infer_required_capabilities(task_request.task_description)
        }
        
        return analysis
    
    def _map_complexity_to_score(self, complexity: str) -> float:
        """Map complexity string to a numeric score"""
        complexity_map = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9
        }
        return complexity_map.get(complexity, 0.6)  # Default to medium
    
    def _infer_required_capabilities(self, task_description: str) -> Dict[str, float]:
        """Infer required capabilities from task description"""
        # Simple keyword-based capability inference
        description_lower = task_description.lower()
        
        capabilities = {
            "coding": 0.0,
            "analysis": 0.0,
            "creativity": 0.0,
            "complexity_handling": 0.0
        }
        
        # Coding-related keywords
        coding_keywords = ["code", "programming", "function", "class", "algorithm", "implementation", "develop"]
        if any(keyword in description_lower for keyword in coding_keywords):
            capabilities["coding"] = 0.8
        
        # Analysis-related keywords
        analysis_keywords = ["analyze", "review", "examine", "evaluate", "assess", "compare", "research"]
        if any(keyword in description_lower for keyword in analysis_keywords):
            capabilities["analysis"] = 0.8
        
        # Creativity-related keywords
        creative_keywords = ["design", "create", "innovate", "brainstorm", "imagine", "visualize", "plan"]
        if any(keyword in description_lower for keyword in creative_keywords):
            capabilities["creativity"] = 0.8
        
        # Complexity-related keywords
        complexity_keywords = ["complex", "challenging", "difficult", "advanced", "sophisticated", "intricate"]
        if any(keyword in description_lower for keyword in complexity_keywords):
            capabilities["complexity_handling"] = 0.8
        
        # If no specific capabilities identified, assign medium values
        if all(v == 0.0 for v in capabilities.values()):
            for key in capabilities:
                capabilities[key] = 0.5  # Default to medium
        
        return capabilities
    
    async def _analyze_performance(self, agent_performance: Optional[AgentPerformance], task_complexity: str) -> Dict[str, Any]:
        """Analyze historical performance data"""
        if not agent_performance:
            return {
                "performance_score": 0.5,  # Default to neutral if no historical data
                "reliability": 0.5,
                "efficiency": 0.5
            }
        
        # Calculate performance metrics
        success_rate = agent_performance.successful_tasks / max(agent_performance.total_tasks, 1)
        avg_cost = agent_performance.total_cost / max(agent_performance.total_tasks, 1) if agent_performance.total_tasks > 0 else 0
        avg_latency = agent_performance.total_latency / max(agent_performance.total_tasks, 1) if agent_performance.total_tasks > 0 else 0
        avg_quality = agent_performance.average_quality
        
        # Adjust performance score based on complexity
        complexity_factor = self._map_complexity_to_score(task_complexity)
        
        # Calculate weighted performance score
        performance_score = (
            (success_rate * 0.3) +
            (max(0, min(1, 1 - (avg_cost / self.performance_thresholds["cost_per_task"]))) * 0.2) +
            (max(0, min(1, self.performance_thresholds["latency_ms"] / max(avg_latency, 1))) * 0.2) +
            (avg_quality * 0.3)
        )
        
        # Adjust based on complexity handling
        performance_score = min(1.0, performance_score + (complexity_factor * 0.1))
        
        return {
            "performance_score": performance_score,
            "success_rate": success_rate,
            "avg_cost": avg_cost,
            "avg_latency": avg_latency,
            "avg_quality": avg_quality,
            "reliability": success_rate,
            "efficiency": 1.0 - min(1.0, avg_cost / self.performance_thresholds["cost_per_task"])
        }
    
    async def _analyze_cost_constraints(self, task_request: TaskRequest) -> Dict[str, Any]:
        """Analyze cost-related constraints and considerations"""
        return {
            "budget_available": task_request.estimated_cost * 2 if task_request.estimated_cost else 2.0,
            "cost_sensitivity": 0.5 if task_request.estimated_cost and task_request.estimated_cost < 0.5 else 0.3,
            "premium_model_threshold": 0.5  # Threshold above which premium models might be justified
        }
    
    def _get_sla_requirements(self, complexity: str) -> Dict[str, Any]:
        """Get SLA requirements based on task complexity"""
        sla_requirements = {
            "low": {
                "max_response_time": 30,  # seconds
                "min_quality": 0.6,
                "max_cost": 0.50
            },
            "medium": {
                "max_response_time": 60,  # seconds
                "min_quality": 0.75,
                "max_cost": 1.00
            },
            "high": {
                "max_response_time": 120,  # seconds
                "min_quality": 0.85,
                "max_cost": 2.00
            }
        }
        
        return sla_requirements.get(complexity, sla_requirements["medium"])
    
    async def _get_system_load(self) -> Dict[str, Any]:
        """Get current system load information"""
        # This would connect to actual system monitoring in a real implementation
        # For now, return simulated data
        return {
            "cpu_usage": 0.4,  # 40% CPU usage
            "memory_usage": 0.6,  # 60% memory usage
            "active_tasks": 5,
            "queue_length": 2,
            "model_availability": {
                "gpt-4": True,
                "gpt-3.5-turbo": True,
                "gpt-4-32k": True
            }
        }
    
    async def _apply_decision_logic(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> Dict[str, Any]:
        """Apply the core decision logic to determine the best course of action"""
        # Determine the best model based on all factors
        best_model = await self._select_best_model(task_request, decision_factors)
        
        # Determine priority adjustment
        priority_adjustment = await self._adjust_priority(task_request, decision_factors)
        
        # Determine cost limit
        cost_limit = await self._determine_cost_limit(task_request, decision_factors)
        
        # Determine time limit
        time_limit = await self._determine_time_limit(task_request, decision_factors)
        
        # Determine if escalation is required
        escalation_required = await self._check_escalation_needed(task_request, decision_factors)
        
        # Determine optimization recommendation
        optimization_recommendation = await self._generate_optimization_recommendation(task_request, decision_factors)
        
        return {
            "approved": True,  # In this implementation, we approve all tasks that pass policy checks
            "model_selection": best_model,
            "priority_adjustment": priority_adjustment,
            "cost_limit": cost_limit,
            "time_limit": time_limit,
            "escalation_required": escalation_required,
            "optimization_recommendation": optimization_recommendation
        }
    
    async def _select_best_model(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> str:
        """Select the best model for the task based on requirements and constraints"""
        # Start with available models
        available_models = decision_factors.model_availability or list(self.model_cost_map.keys())
        
        # Filter based on system availability
        system_load = decision_factors.current_system_load or {}
        model_availability = system_load.get("model_availability", {})
        
        available_models = [
            model for model in available_models 
            if model_availability.get(model, True)
        ]
        
        # Calculate model scores based on task requirements and agent performance
        model_scores = {}
        
        for model in available_models:
            score = await self._calculate_model_score(
                model, 
                task_request, 
                decision_factors.historical_performance,
                decision_factors.cost_budget
            )
            model_scores[model] = score
        
        # Select the model with the highest score
        if not model_scores:
            return "gpt-3.5-turbo"  # Default fallback
        
        best_model = max(model_scores, key=model_scores.get)
        return best_model
    
    async def _calculate_model_score(self, model: str, task_request: TaskRequest, 
                                   agent_performance: Optional[AgentPerformance], 
                                   cost_budget: Optional[float]) -> float:
        """Calculate a score for a model based on various factors"""
        # Get model capabilities
        capabilities = self.model_capabilities.get(model, {})
        
        # Get required capabilities
        required_capabilities = self._infer_required_capabilities(task_request.task_description)
        
        # Calculate capability match score
        capability_match_score = 0
        for capability, required_score in required_capabilities.items():
            model_capability = capabilities.get(capability, 0.0)
            capability_match_score += min(required_score, model_capability)
        
        capability_match_score /= len(required_capabilities) if required_capabilities else 0.5
        
        # Consider historical performance if available
        performance_factor = 1.0
        if agent_performance:
            # If this agent has historically performed well with this type of task, boost score
            if agent_performance.average_quality > 0.8:
                performance_factor = 1.2
            elif agent_performance.average_quality < 0.6:
                performance_factor = 0.8
        
        # Consider cost constraints
        cost_factor = 1.0
        model_cost = self.model_cost_map.get(model, 0.03)  # Default to GPT-4 cost
        if cost_budget and model_cost > cost_budget:
            cost_factor = 0.5  # Significant penalty for exceeding budget
        elif cost_budget and model_cost > cost_budget * 0.8:
            cost_factor = 0.8  # Moderate penalty for approaching budget
        
        # Consider task complexity
        complexity_factor = 1.0
        complexity_score = self._map_complexity_to_score(task_request.complexity)
        if complexity_score > 0.7 and model in ["gpt-3.5-turbo", "claude-instant-1"]:
            complexity_factor = 0.7  # Penalize less capable models for complex tasks
        elif complexity_score < 0.4 and model in ["gpt-4", "gpt-4-32k"]:
            complexity_factor = 0.9  # Slight penalty for over-capable models on simple tasks (cost consideration)
        
        # Calculate final score
        final_score = (
            capability_match_score * 0.4 +
            performance_factor * 0.2 +
            cost_factor * 0.2 +
            complexity_factor * 0.2
        )
        
        return final_score
    
    async def _adjust_priority(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> int:
        """Adjust task priority based on various factors"""
        base_priority = task_request.priority
        
        # Adjust based on urgency (could be determined from task description or external factors)
        urgency_adjustment = 0
        if "urgent" in task_request.task_description.lower() or "asap" in task_request.task_description.lower():
            urgency_adjustment = min(2, 10 - base_priority)  # Max boost of 2, not exceeding 10
        
        # Adjust based on historical performance
        if decision_factors.historical_performance:
            perf = decision_factors.historical_performance
            if perf.successful_tasks > 0:
                success_rate = perf.successful_tasks / perf.total_tasks
                if success_rate < 0.7:  # Low success rate might indicate need for higher priority attention
                    urgency_adjustment = max(urgency_adjustment, 1)
        
        # Adjust based on complexity
        complexity_adjustment = 0
        complexity_score = self._map_complexity_to_score(task_request.complexity)
        if complexity_score > 0.7:  # High complexity might need higher priority
            complexity_adjustment = min(1, 10 - base_priority - urgency_adjustment)
        
        adjusted_priority = min(10, max(1, base_priority + urgency_adjustment + complexity_adjustment))
        return adjusted_priority
    
    async def _determine_cost_limit(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> float:
        """Determine the cost limit for the task"""
        base_cost_limit = task_request.estimated_cost * 2 if task_request.estimated_cost else 1.0
        
        # Adjust based on budget constraints
        if decision_factors.cost_budget:
            # Don't exceed 50% of available budget for a single task
            budget_based_limit = decision_factors.cost_budget * 0.5
            base_cost_limit = min(base_cost_limit, budget_based_limit)
        
        # Adjust based on task complexity
        complexity_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }
        complexity_factor = complexity_multiplier.get(task_request.complexity, 1.0)
        adjusted_limit = base_cost_limit * complexity_factor
        
        # Ensure it's within reasonable bounds
        adjusted_limit = max(0.1, min(10.0, adjusted_limit))  # Between $0.10 and $10.00
        
        return adjusted_limit
    
    async def _determine_time_limit(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> float:
        """Determine the time limit for the task"""
        base_time_limit = task_request.estimated_time * 3 if task_request.estimated_time else 300.0  # 5 minutes default
        
        # Adjust based on SLA requirements
        sla_requirements = decision_factors.sla_requirements or {}
        sla_max_time = sla_requirements.get("max_response_time")
        
        if sla_max_time:
            base_time_limit = min(base_time_limit, sla_max_time * 2)  # SLA * 2 as upper bound
        
        # Adjust based on complexity
        complexity_multiplier = {
            "low": 1.5,
            "medium": 2.0,
            "high": 3.0
        }
        complexity_factor = complexity_multiplier.get(task_request.complexity, 2.0)
        adjusted_limit = base_time_limit * complexity_factor
        
        # Ensure it's within reasonable bounds
        adjusted_limit = max(30.0, min(3600.0, adjusted_limit))  # Between 30 seconds and 1 hour
        
        return adjusted_limit
    
    async def _check_escalation_needed(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> bool:
        """Check if the task requires escalation"""
        # Escalate if complexity is high and agent performance is poor
        if (task_request.complexity == "high" and 
            decision_factors.historical_performance and 
            decision_factors.historical_performance.average_quality < 0.6):
            return True
        
        # Escalate if estimated cost is very high
        if task_request.estimated_cost and task_request.estimated_cost > 5.0:  # More than $5
            return True
        
        # Escalate if priority is maximum and complexity is high
        if task_request.priority >= 9 and task_request.complexity == "high":
            return True
        
        # Escalate based on SLA requirements
        sla_requirements = decision_factors.sla_requirements
        if (sla_requirements and 
            sla_requirements.get("max_response_time", 300) < 60 and  # Very tight deadline
            task_request.complexity == "high"):
            return True
        
        return False
    
    async def _generate_optimization_recommendation(self, task_request: TaskRequest, decision_factors: DecisionFactors) -> Optional[str]:
        """Generate optimization recommendations based on the decision factors"""
        recommendations = []
        
        # If the selected model is expensive for a low-complexity task, recommend optimization
        if (task_request.complexity == "low" and 
            decision_factors.cost_budget and 
            decision_factors.cost_budget > 0.5):  # Expensive task for low complexity
            recommendations.append("Consider simplifying task or using cheaper model")
        
        # If agent performance is consistently poor, recommend optimization
        if (decision_factors.historical_performance and 
            decision_factors.historical_performance.average_quality < 0.6):
            recommendations.append("Agent performance optimization recommended")
        
        # If task priority is low but resource usage is high, recommend batch processing
        if (task_request.priority < 4 and 
            task_request.estimated_cost and 
            task_request.estimated_cost > 1.0):
            recommendations.append("Consider batch processing for cost optimization")
        
        return "; ".join(recommendations) if recommendations else None
