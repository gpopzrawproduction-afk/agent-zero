# Data schemas for AI Ops Agent
# Defines the data structures used by the AI Ops system

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time

@dataclass
class AgentPerformance:
    """Performance metrics for an individual agent"""
    agent_name: str
    total_tasks: int = 0
    successful_tasks: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0  # in milliseconds
    average_quality: float = 0.0  # 0.0 to 1.0 scale


@dataclass
class PolicyConfig:
    """Configuration for a policy rule"""
    name: str
    agent: str  # Agent name or '*' for all agents
    condition: str # Condition expression to evaluate
    action: str  # Action to take when condition is met
    description: str
    enabled: bool = True
    priority: int = 1  # Lower numbers are higher priority


@dataclass
class OptimizationRecommendation:
    """A recommendation from the optimization engine"""
    recommendation_id: str
    agent_name: Optional[str] = None
    workflow_id: Optional[str] = None
    category: str = ""  # 'model_selection', 'prompt_optimization', 'routing', etc.
    recommendation: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    impact_estimate: float = 0.0  # Expected improvement
    timestamp: float = 0.0


@dataclass
class EscalationEvent:
    """An event that requires human escalation"""
    agent_name: str
    violation: Any  # Policy violation that triggered escalation
    context: Dict[str, Any]
    timestamp: float
    reason: str = ""
    severity: str = "medium"  # 'low', 'medium', 'high', 'critical'


@dataclass
class MetricData:
    """Structured metrics data collected by telemetry"""
    agent: str
    task_id: str
    latency_ms: float
    cost_usd: float
    success: bool
    quality_score: float # 0.0 to 1.0
    model: str
    timestamp: float
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    retries: int = 0
    priority: Optional[int] = None  # 1-10 scale
    task_complexity: Optional[str] = None  # 'low', 'medium', 'high'
    customer_impact: Optional[str] = None  # 'low', 'medium', 'high'


@dataclass
class DecisionFactors:
    """Factors considered by the decision engine"""
    task_priority: int  # 1-10 scale
    historical_performance: Optional[AgentPerformance] = None
    cost_budget: Optional[float] = None
    sla_requirements: Optional[Dict[str, Any]] = None
    current_system_load: Optional[Dict[str, Any]] = None
    agent_availability: Optional[List[str]] = None
    model_availability: Optional[List[str]] = None


@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    policy_name: str
    agent_name: str
    condition: str
    action: str
    value: Any
    timestamp: float
    severity: str = "medium"


@dataclass
class OptimizationHistory:
    """Historical record of optimizations applied"""
    optimization_id: str
    timestamp: float
    category: str
    description: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    impact: float  # Positive for improvement, negative for degradation
    confidence: float  # 0.0 to 1.0


@dataclass
class SystemKPIs:
    """System-level KPIs tracked by AI Ops"""
    cost_per_completed_objective: float
    success_rate_per_workflow: float
    mean_time_to_resolution_ms: float
    human_escalation_rate: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AgentKPIs:
    """Agent-level KPIs tracked by AI Ops"""
    agent_name: str
    cost_efficiency_score: float  # 0.0 to 1.0
    reliability_score: float  # 0.0 to 1.0
    quality_score: float # 0.0 to 1.0
    speed_score: float # 0.0 to 1.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class WorkflowOptimization:
    """Optimization data for a specific workflow"""
    workflow_id: str
    original_agents: List[str]
    optimized_agents: List[str]
    expected_improvement: float
    confidence: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
