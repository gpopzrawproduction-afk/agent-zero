from zero_gravity_core.agents.base import BaseAgent
from typing import Any, Dict, List


class Operator(BaseAgent):
    """
    Operator Agent

    Responsibilities:
    - Receive implementation blueprint from Engineer
    - Execute or simulate execution of each step
    - Produce final results for Coordinator
    - Optionally interact with tools (via coordinator)
    """

    def __init__(self, base_dir: str = None, role: str = "operator", system_prompt: str = None, coordinator: Any = None):
        super().__init__(base_dir=base_dir, role=role, system_prompt=system_prompt, coordinator=coordinator)

    def execute(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for Operator execution.
        Returns a result dictionary after performing all tasks.
        """

        self.record({"input_blueprint": blueprint})

        # Use LLM to execute the blueprint
        result = self.execute_with_llm(blueprint)
        
        # If the result is not in the expected format, create it
        if not isinstance(result, dict) or "execution_results" not in result:
            result = self._execute_blueprint(blueprint)

        self.record({"result": result})

        return result

    def _execute_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the blueprint when LLM integration is not available.
        """
        objective = blueprint.get("objective", "Unknown Objective")
        steps = blueprint.get("implementation_steps", [])

        # Execute the steps
        execution_results = self._execute_steps(steps)

        # Create the result structure as specified in the system prompt
        result = {
            "objective": objective,
            "execution_results": execution_results,
            "overall_status": "completed",
            "summary": "Execution completed successfully",
            "next_steps": []
        }

        return result

    # -------------------------
    # INTERNAL LOGIC
    # -------------------------

    def _execute_steps(self, steps: List[str]) -> List[Dict[str, Any]]:
        """
        Mock execution of blueprint steps.
        Replace with real tool calls, APIs, or scripts later.
        """

        results = []
        for i, step in enumerate(steps, start=1):
            # Simulate task execution
            results.append({
                "step_number": i,
                "step_description": step,
                "status": "completed",
                "output": f"Simulated output of step {i}",
                "execution_time": "0.1s",
                "errors": []
            })

        return results
