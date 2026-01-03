from zero_gravity_core.agents.base import BaseAgent
from typing import Any, Dict, List


class Engineer(BaseAgent):
    """
    Engineer Agent

    Responsibilities:
    - Receive plan from Architect
    - Translate plan into actionable implementation steps
    - Identify dependencies and resources
    - Produce structured instructions for downstream agents
    """

    def __init__(self, base_dir: str = None, role: str = "engineer", system_prompt: str = None, coordinator: Any = None):
        super().__init__(base_dir=base_dir, role=role, system_prompt=system_prompt, coordinator=coordinator)

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for Engineer execution.
        Expects a plan dictionary from Architect.
        Returns a structured implementation blueprint.
        """

        self.record({"input_plan": plan})

        # Use LLM to convert the plan into an implementation blueprint
        blueprint = self.execute_with_llm(plan)
        
        # If the result is not in the expected format, create it
        if not isinstance(blueprint, dict) or "implementation_steps" not in blueprint:
            blueprint = self._create_blueprint_from_plan(plan)

        self.record({"blueprint": blueprint})

        return blueprint

    def _create_blueprint_from_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an implementation blueprint from a plan when LLM integration is not available.
        """
        objective = plan.get("objective", "Unknown Objective")
        steps = plan.get("steps", [])

        # Refine steps into implementation instructions
        implementation_steps = self._refine_steps(steps)

        # Create the blueprint structure as specified in the system prompt
        blueprint = {
            "objective": objective,
            "implementation_steps": implementation_steps,
            "technology_stack": ["python", "zerogravity_framework"],
            "estimated_effort": "medium",
            "critical_path": ["1"]
        }

        return blueprint

    # -------------------------
    # INTERNAL LOGIC
    # -------------------------

    def _refine_steps(self, steps: List[str]) -> List[str]:
        """
        Convert generic plan steps into actionable, implementation-ready steps.
        Placeholder logic for now; can integrate LLM or tool-based analysis later.
        """

        refined_steps = []
        for i, step in enumerate(steps, start=1):
            refined_steps.append(f"Step {i}: {step} [Engineer-processed]")

        # Optional: detect dependencies, resources, or subtasks here in the future

        return refined_steps
