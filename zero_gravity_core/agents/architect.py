from zero_gravity_core.agents.base import BaseAgent
from typing import Any, Dict


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

    def execute(self, objective: str) -> Dict[str, Any]:
        """
        Entry point for Architect execution.
        Returns a plan dictionary.
        """

        self.record({"input_objective": objective})

        # Use LLM to decompose the objective
        plan = self.execute_with_llm(objective)
        
        # If the result is not in the expected format, create it
        if not isinstance(plan, dict) or "steps" not in plan:
            plan = self._create_plan_with_steps(objective)

        self.record({"plan": plan})

        return plan

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
