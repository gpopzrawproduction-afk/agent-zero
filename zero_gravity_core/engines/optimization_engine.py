# Optimization Engine for AI Ops Agent
# Implements performance optimization, workflow improvement, and learning mechanisms

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
from pathlib import Path
import aiofiles

from zero_gravity_core.schemas.ai_ops_schemas import (
    AgentPerformance, 
    MetricData, 
    OptimizationRecommendation,
    OptimizationHistory,
    WorkflowOptimization
)

logger = logging.getLogger(__name__)

class OptimizationEngine:
    """
    The Optimization Engine implements:
    - Performance optimization
    - Workflow improvement algorithms
    - Cost efficiency tracking
    - Model switching logic
    - Learning and adaptation mechanisms
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "zero_gravity_core/optimization_data"
        self.optimization_history: List[OptimizationHistory] = []
        self.agent_recommendations: Dict[str, List[OptimizationRecommendation]] = {}
        self.workflow_optimizations: Dict[str, WorkflowOptimization] = {}
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds for optimization triggers
        self.thresholds = {
            "low_success_rate": 0.7,  # 70%
            "high_cost_per_task": 0.5,  # $0.50
            "high_latency": 5000,  # 5 seconds
            "low_quality": 0.7  # 70%
        }
        
        logger.info("Optimization Engine initialized")
    
    async def analyze_performance(self, agent_name: str, metric_data: MetricData, 
                                current_performance: Optional[AgentPerformance]) -> Optional[OptimizationRecommendation]:
        """
        Analyze performance data and generate optimization recommendations
        """
        logger.info(f"Analyzing performance for agent: {agent_name}")
        
        # Generate a unique recommendation ID
        recommendation_id = f"opt_rec_{int(time.time())}_{agent_name}"
        
        # Check for various optimization opportunities
        recommendations = []
        
        # Check for cost optimization opportunities
        if metric_data.cost_usd > self.thresholds["high_cost_per_task"]:
            recommendations.append({
                "category": "cost_optimization",
                "recommendation": f"Task cost (${metric_data.cost_usd:.2f}) exceeds threshold (${self.thresholds['high_cost_per_task']:.2f}). Consider using a more cost-effective model.",
                "confidence": 0.8,
                "impact_estimate": metric_data.cost_usd * 0.3  # Potential 30% cost reduction
            })
        
        # Check for performance optimization opportunities
        if metric_data.latency_ms > self.thresholds["high_latency"]:
            recommendations.append({
                "category": "performance_optimization",
                "recommendation": f"Task latency ({metric_data.latency_ms:.2f}ms) exceeds threshold ({self.thresholds['high_latency']}ms). Consider optimizing prompt or using faster model.",
                "confidence": 0.7,
                "impact_estimate": metric_data.latency_ms * 0.2  # Potential 20% latency reduction
            })
        
        # Check for quality optimization opportunities
        if metric_data.quality_score < self.thresholds["low_quality"]:
            recommendations.append({
                "category": "quality_optimization",
                "recommendation": f"Task quality ({metric_data.quality_score:.2f}) below threshold ({self.thresholds['low_quality']:.2f}). Consider using a more capable model or refining prompt.",
                "confidence": 0.9,
                "impact_estimate": (self.thresholds["low_quality"] - metric_data.quality_score) * 100 # Potential quality improvement
            })
        
        # Check historical performance if available
        if current_performance:
            success_rate = current_performance.successful_tasks / max(current_performance.total_tasks, 1)
            if success_rate < self.thresholds["low_success_rate"]:
                recommendations.append({
                    "category": "success_rate_optimization",
                    "recommendation": f"Agent success rate ({success_rate:.2f}) below threshold ({self.thresholds['low_success_rate']:.2f}). Consider agent retraining or task redistribution.",
                    "confidence": 0.85,
                    "impact_estimate": (self.thresholds["low_success_rate"] - success_rate) * 100  # Potential success rate improvement
                })
        
        # If we have recommendations, create an optimization recommendation
        if recommendations:
            # For now, return the highest priority recommendation
            highest_priority = max(recommendations, key=lambda x: x["confidence"])
            
            optimization_rec = OptimizationRecommendation(
                recommendation_id=recommendation_id,
                agent_name=agent_name,
                category=highest_priority["category"],
                recommendation=highest_priority["recommendation"],
                confidence=highest_priority["confidence"],
                impact_estimate=highest_priority["impact_estimate"],
                timestamp=time.time()
            )
            
            # Store the recommendation
            if agent_name not in self.agent_recommendations:
                self.agent_recommendations[agent_name] = []
            self.agent_recommendations[agent_name].append(optimization_rec)
            
            return optimization_rec
        
        return None
    
    async def apply_optimization(self, recommendation: OptimizationRecommendation):
        """
        Apply an optimization recommendation
        """
        logger.info(f"Applying optimization: {recommendation.recommendation}")
        
        # This would contain the actual implementation of the optimization
        # For now, we'll just log the action and update the history
        
        # Create an optimization history entry
        history_entry = OptimizationHistory(
            optimization_id=recommendation.recommendation_id,
            timestamp=time.time(),
            category=recommendation.category,
            description=recommendation.recommendation,
            before_metrics={},
            after_metrics={},
            impact=recommendation.impact_estimate,
            confidence=recommendation.confidence
        )
        
        self.optimization_history.append(history_entry)
        
        # Save the optimization to history file
        await self._save_optimization_to_file(history_entry)
    
    async def optimize_workflow(self, workflow_id: str, current_agents: List[str], 
                              agent_performance: Dict[str, AgentPerformance]) -> List[str]:
        """
        Optimize workflow by suggesting better agent assignments based on performance data
        """
        logger.info(f"Optimizing workflow: {workflow_id}")
        
        # Analyze current agent performance for this workflow type
        optimized_agents = current_agents.copy()
        
        # For each agent in the workflow, check if there's a better alternative
        for i, current_agent in enumerate(current_agents):
            if current_agent in agent_performance:
                current_perf = agent_performance[current_agent]
                
                # Find alternative agents that might perform better
                better_agents = self._find_better_agents(
                    current_agent, current_perf, agent_performance
                )
                
                # If we found a better agent, consider replacing it
                if better_agents:
                    # For now, pick the best performing alternative
                    best_alternative = better_agents[0]
                    
                    # Calculate potential improvement
                    improvement = self._calculate_improvement(
                        current_perf, agent_performance[best_alternative]
                    )
                    
                    # Only replace if improvement is significant enough
                    if improvement > 0.1:  # 10% improvement threshold
                        optimized_agents[i] = best_alternative
                        logger.info(f"Replacing {current_agent} with {best_alternative} in workflow {workflow_id} (estimated improvement: {improvement:.2f})")
        
        # Create a workflow optimization record
        workflow_opt = WorkflowOptimization(
            workflow_id=workflow_id,
            original_agents=current_agents,
            optimized_agents=optimized_agents,
            expected_improvement=sum(
                self._calculate_improvement(
                    agent_performance.get(orig, AgentPerformance(orig)),
                    agent_performance.get(opt, AgentPerformance(opt))
                ) for orig, opt in zip(current_agents, optimized_agents) if orig != opt
            ),
            confidence=0.8,  # Default confidence
            timestamp=time.time()
        )
        
        self.workflow_optimizations[workflow_id] = workflow_opt
        
        # Save workflow optimization
        await self._save_workflow_optimization(workflow_opt)
        
        return optimized_agents
    
    def _find_better_agents(self, current_agent: str, current_performance: AgentPerformance,
                           all_performances: Dict[str, AgentPerformance]) -> List[str]:
        """
        Find agents that perform better than the current agent for similar tasks
        """
        better_agents = []
        
        # Calculate a performance score for the current agent
        current_score = self._calculate_agent_performance_score(current_performance)
        
        # Compare with all other agents
        for agent_name, perf in all_performances.items():
            if agent_name != current_agent:
                other_score = self._calculate_agent_performance_score(perf)
                if other_score > current_score:
                    better_agents.append(agent_name)
        
        # Sort by performance score (descending)
        better_agents.sort(
            key=lambda x: self._calculate_agent_performance_score(all_performances[x]), 
            reverse=True
        )
        
        return better_agents
    
    def _calculate_agent_performance_score(self, performance: AgentPerformance) -> float:
        """
        Calculate an overall performance score for an agent
        """
        # Calculate weighted score based on multiple factors
        success_rate = performance.successful_tasks / max(performance.total_tasks, 1)
        avg_quality = performance.average_quality
        cost_efficiency = 1.0 / (performance.total_cost / max(performance.total_tasks, 1) + 0.01)  # Add small value to avoid division by zero
        speed_score = 1000.0 / (performance.total_latency / max(performance.total_tasks, 1) + 1.0)  # Add 1 to avoid division by zero
        
        # Weighted combination (adjust weights as needed)
        score = (
            success_rate * 0.4 +
            avg_quality * 0.3 +
            min(1.0, cost_efficiency * 0.1) * 0.15 +  # Normalize cost efficiency
            min(1.0, speed_score * 0.01) * 0.15  # Normalize speed score
        )
        
        return score
    
    def _calculate_improvement(self, current_perf: AgentPerformance, new_perf: AgentPerformance) -> float:
        """
        Calculate the expected improvement from switching agents
        """
        current_score = self._calculate_agent_performance_score(current_perf)
        new_score = self._calculate_agent_performance_score(new_perf)
        
        return new_score - current_score
    
    async def _save_optimization_to_file(self, history_entry: OptimizationHistory):
        """
        Save optimization history to file
        """
        file_path = Path(self.storage_path) / "optimization_history.jsonl"
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(asdict(history_entry)) + '\n')
    
    async def _save_workflow_optimization(self, workflow_opt: WorkflowOptimization):
        """
        Save workflow optimization to file
        """
        file_path = Path(self.storage_path) / "workflow_optimizations.jsonl"
        
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(asdict(workflow_opt)) + '\n')
    
    async def get_optimization_recommendations(self, agent_name: str) -> List[OptimizationRecommendation]:
        """
        Get all optimization recommendations for a specific agent
        """
        return self.agent_recommendations.get(agent_name, [])
    
    async def get_workflow_optimizations(self, workflow_id: str) -> List[WorkflowOptimization]:
        """
        Get optimization history for a specific workflow
        """
        return [wo for wo in self.workflow_optimizations.values() if wo.workflow_id == workflow_id]
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """
        Get overall optimization engine status
        """
        total_recommendations = sum(len(recs) for recs in self.agent_recommendations.values())
        total_workflows_optimized = len(self.workflow_optimizations)
        total_history_entries = len(self.optimization_history)
        
        return {
            "total_recommendations": total_recommendations,
            "workflows_optimized": total_workflows_optimized,
            "optimization_history_count": total_history_entries,
            "agents_with_recommendations": len(self.agent_recommendations),
            "avg_confidence": sum(
                sum(rec.confidence for rec in recs) 
                for recs in self.agent_recommendations.values()
            ) / max(total_recommendations, 1)
        }
    
    async def learn_from_outcomes(self, optimization_id: str, actual_outcome: Dict[str, float]):
        """
        Learn from the actual outcomes of applied optimizations
        """
        # Find the optimization in history
        for opt in self.optimization_history:
            if opt.optimization_id == optimization_id:
                # Update the after_metrics with actual outcomes
                opt.after_metrics = actual_outcome
                
                # Calculate actual impact
                before_avg = sum(opt.before_metrics.values()) / len(opt.before_metrics) if opt.before_metrics else 0
                after_avg = sum(actual_outcome.values()) / len(actual_outcome) if actual_outcome else 0
                opt.impact = after_avg - before_avg
                
                logger.info(f"Updated optimization {optimization_id} with actual outcomes. Impact: {opt.impact}")
                
                # Save updated history
                await self._save_optimization_to_file(opt)
                return
        
        logger.warning(f"Optimization not found for learning: {optimization_id}")
    
    async def get_optimization_insights(self) -> Dict[str, Any]:
        """
        Get insights about optimization effectiveness
        """
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        # Calculate average impact
        avg_impact = sum(opt.impact for opt in self.optimization_history) / len(self.optimization_history)
        
        # Calculate success rate (positive impact means successful optimization)
        successful_optimizations = sum(1 for opt in self.optimization_history if opt.impact > 0)
        success_rate = successful_optimizations / len(self.optimization_history)
        
        # Get most effective optimization categories
        category_impacts = {}
        for opt in self.optimization_history:
            if opt.category not in category_impacts:
                category_impacts[opt.category] = []
            category_impacts[opt.category].append(opt.impact)
        
        avg_category_impacts = {
            cat: sum(impacts) / len(impacts) 
            for cat, impacts in category_impacts.items()
        }
        
        return {
            "total_optimizations": len(self.optimization_history),
            "success_rate": success_rate,
            "average_impact": avg_impact,
            "most_effective_categories": sorted(
                avg_category_impacts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],  # Top 5 categories
            "total_improvement": sum(opt.impact for opt in self.optimization_history if opt.impact > 0),
            "total_reduction": sum(opt.impact for opt in self.optimization_history if opt.impact < 0)
        }
    
    async def save_learnings(self):
        """
        Save all optimization learnings to persistent storage
        """
        # This method would save the current state of the optimization engine
        # In a real implementation, this might involve saving to a database or file
        logger.info("Saving optimization learnings...")
        
        # Save to a checkpoint file
        checkpoint_path = Path(self.storage_path) / "optimization_checkpoint.json"
        
        checkpoint_data = {
            "timestamp": time.time(),
            "optimization_history": [asdict(opt) for opt in self.optimization_history],
            "workflow_optimizations": {k: asdict(v) for k, v in self.workflow_optimizations.items()},
            "agent_recommendations": {
                agent: [asdict(rec) for rec in recs] 
                for agent, recs in self.agent_recommendations.items()
            }
        }
        
        async with aiofiles.open(checkpoint_path, 'w') as f:
            await f.write(json.dumps(checkpoint_data, indent=2))
        
        logger.info(f"Optimization learnings saved to {checkpoint_path}")
    
    async def load_learnings(self):
        """
        Load optimization learnings from persistent storage
        """
        # This method would load the saved state of the optimization engine
        checkpoint_path = Path(self.storage_path) / "optimization_checkpoint.json"
        
        if not checkpoint_path.exists():
            logger.info("No optimization checkpoint found, starting fresh")
            return
        
        try:
            async with aiofiles.open(checkpoint_path, 'r') as f:
                content = await f.read()
                checkpoint_data = json.loads(content)
            
            # Restore optimization history
            self.optimization_history = [
                OptimizationHistory(**opt_data) 
                for opt_data in checkpoint_data.get("optimization_history", [])
            ]
            
            # Restore workflow optimizations
            self.workflow_optimizations = {
                k: WorkflowOptimization(**v) 
                for k, v in checkpoint_data.get("workflow_optimizations", {}).items()
            }
            
            # Restore agent recommendations
            self.agent_recommendations = {}
            for agent, recs in checkpoint_data.get("agent_recommendations", {}).items():
                self.agent_recommendations[agent] = [
                    OptimizationRecommendation(**rec_data) 
                    for rec_data in recs
                ]
            
            logger.info(f"Optimization learnings loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading optimization learnings: {str(e)}")
    
    async def reset_optimization_engine(self):
        """
        Reset the optimization engine to initial state
        """
        self.optimization_history = []
        self.agent_recommendations = {}
        self.workflow_optimizations = {}
        
        logger.info("Optimization Engine reset to initial state")
