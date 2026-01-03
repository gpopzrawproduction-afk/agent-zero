from zero_gravity_core.agents.base import BaseAgent
from typing import Any, Dict, List


class Designer(BaseAgent):
    """
    Designer Agent

    Responsibilities:
    - Receive plan or blueprint
    - Generate structured visualization or presentation
    - Produce outputs for review or immersive display
    """

    def __init__(self, base_dir: str = None, role: str = "designer", system_prompt: str = None, coordinator: Any = None):
        super().__init__(base_dir=base_dir, role=role, system_prompt=system_prompt, coordinator=coordinator)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for Designer execution.
        Returns a design representation dictionary.
        """

        self.record({"input_data": input_data})

        # Use LLM to create visualization from input data
        design_output = self.execute_with_llm(input_data)
        
        # If the result is not in the expected format, create it
        if not isinstance(design_output, dict) or "visual_structure" not in design_output:
            design_output = self._create_visualization_from_data(input_data)

        self.record({"design_output": design_output})

        return design_output

    def _create_visualization_from_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a visualization from input data when LLM integration is not available.
        """
        # Extract steps from input data
        steps = input_data.get("steps") or input_data.get("implementation_steps") or []

        # Create structured visualization
        visual_structure = self._visualize_steps(steps)

        # Create the design output structure as specified in the system prompt
        design_output = {
            "summary": f"Designer output for role '{self.role}'",
            "visualization_type": "flowchart",
            "elements": visual_structure,
            "layout": {
                "orientation": "vertical",
                "hierarchy": True
            },
            "style_guide": {
                "colors": {"primary": "#3498db", "secondary": "#2ecc71"},
                "typography": {"font_family": "Arial", "sizes": {"header": 16, "body": 12}}
            }
        }

        return design_output

    # -------------------------
    # INTERNAL LOGIC
    # -------------------------

    def _visualize_steps(self, steps: List[str]) -> List[Dict[str, Any]]:
        """
        Create a mock structured representation of steps.
        Replace with real visualization / diagram logic later.
        """

        visualized = []
        for i, step in enumerate(steps, start=1):
            visualized.append({
                "id": f"step_{i}",
                "type": "process",
                "content": step,
                "position": {"x": 0, "y": i * 50},  # Placeholder for coordinates in a future diagram
                "connections": [f"step_{i+1}"] if i < len(steps) else []  # Connect to next step
            })

        return visualized
