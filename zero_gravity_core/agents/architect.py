from zero_gravity_core.agents.base import BaseAgent
from typing import Any, Dict
import time
import asyncio

from zero_gravity_core.agents.ai_ops_integration import (
    request_task_approval,
    report_task_metrics,
    enforce_policy,
    get_agent_kpis
)


class Architect(BaseAgent):
    """
    Architect Agent

    Responsible for:
    - Receiving high-level objectives
    - Decomposing objectives into actionable steps
    - Producing a strategic plan for execution
    """

    def __init__(self, base_dir: str = None, role: str = "architect", system_prompt: str = None, coordinator: Any = None):
        super().__init__(base_dir=base_dir, role=role, system_prompt=system_prompt, coordinator=coordinator)

    async def execute(self, objective: str) -> Dict[str, Any]:
        """
        Entry point for Architect execution with AI Ops integration.
        Returns a plan dictionary.
        """
        start_time = time.time()
        cost_estimate = 0.05  # Estimated cost for this task
        quality_score = 0.0  # Will be set after completion

        # Request approval from AI Ops
        approval = await request_task_approval(
            agent_name="architect",
            task_description=f"Decompose objective: {objective}",
            priority=5,
            estimated_cost=cost_estimate,
            estimated_time=60.0,
            complexity="medium"
        )

        if not approval.approved:
            raise Exception(f"Task not approved by AI Ops: {approval}")

        self.record({"input_objective": objective})
        self.record({"approval": approval})

        try:
            # Use LLM to decompose the objective
            plan = self.execute_with_llm(objective)
            
            # If the result is not in the expected format, create it
            if not isinstance(plan, dict) or "steps" not in plan:
                plan = self._create_plan_with_steps(objective)

            self.record({"plan": plan})
            quality_score = 0.9  # Assume high quality for successful execution

            return plan
        except Exception as e:
            quality_score = 0.3  # Lower quality score for failed execution
            raise e
        finally:
            # Report metrics to AI Ops
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            await report_task_metrics(
                agent_name="architect",
                task_id=f"architect_{int(time.time())}",
                latency_ms=execution_time,
                cost_usd=cost_estimate,
                success=quality_score > 0.5,
                quality_score=quality_score,
                model_used=getattr(self, 'model', 'gpt-4'),  # Use model if available, default to gpt-4
                error=None if quality_score > 0.5 else str(e) if 'e' in locals() else None
            )

    def _create_plan_with_steps(self, objective: str) -> Dict[str, Any]:
        """
        Create a plan with steps when LLM integration is not available.
        """
        # Use the original decompose method as fallback
        plan_steps = self._decompose(objective)

        # Create the plan structure as specified in the system prompt
        plan = {
            "objective": objective,
            "steps": plan_steps,
            "dependencies": "Sequential execution required",
            "estimated_complexity": "medium",
            "required_resources": ["analysis_tools", "planning_tools"]
        }

        return plan

    # -------------------------
    # INTERNAL LOGIC
    # -------------------------

    def _decompose(self, objective: str) -> list[str]:
        """
        Decompose objective into step-by-step plan.
        This is currently a placeholder; eventually will integrate
        LLM reasoning or custom logic.
        """

        # Simple mock decomposition for demonstration
        steps = [
            f"Analyze the objective: '{objective}'",
            "Break it into core components",
            "Identify dependencies and requirements",
            "Define execution phases",
            "Produce structured plan for downstream agents"
        ]

        return steps
