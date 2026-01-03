from pathlib import Path
from typing import Dict, Any, List
import datetime
import json


class Coordinator:
    """
    ZeroGravity Master Coordinator

    Responsibilities:
    - Load system prompts
    - Spawn role agents
    - Route tasks
    - Evaluate outputs
    - Control execution flow
    - Log and track all activities
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parents[1]
        self.prompts_dir = self.base_dir / "prompts"

        self.session_id = datetime.datetime.utcnow().isoformat()
        self.state: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}

        # Enhanced logging capabilities
        self.log_entries: List[Dict[str, Any]] = []
        
        self._load_system_prompts()

    # -------------------------
    # PROMPT MANAGEMENT
    # -------------------------

    def _load_system_prompts(self) -> None:
        """Load all system prompts into memory"""
        self.system_prompts: Dict[str, str] = {}

        for prompt_file in self.prompts_dir.glob("*.system.md"):
            role = prompt_file.stem.replace(".system", "")
            self.system_prompts[role] = prompt_file.read_text(encoding="utf-8")

        if not self.system_prompts:
            raise RuntimeError("No system prompts found. Coordinator cannot start.")

    def get_prompt(self, role: str) -> str:
        if role not in self.system_prompts:
            raise ValueError(f"No system prompt defined for role: {role}")
        return self.system_prompts[role]

    # -------------------------
    # AGENT SPAWNING
    # -------------------------

    def spawn_agent(self, role: str):
        """
        Lazy-load and spawn an agent by role.
        Agents are created only when needed.
        """
        if role in self.agents:
            self.log(f"Reusing existing {role} agent", "INFO")
            return self.agents[role]

        module_path = f"zero_gravity_core.agents.{role}"
        try:
            module = __import__(module_path, fromlist=[None])
        except ImportError as e:
            self.log(f"Agent module for role '{role}' not found", "ERROR")
            raise ImportError(f"Agent module for role '{role}' not found") from e

        agent_class = getattr(module, role.capitalize(), None)
        if not agent_class:
            error_msg = f"Agent class '{role.capitalize()}' missing in {module_path}"
            self.log(error_msg, "ERROR")
            raise RuntimeError(error_msg)

        agent = agent_class(
            role=role,
            system_prompt=self.get_prompt(role),
            coordinator=self,
        )

        self.agents[role] = agent
        self.log(f"Successfully spawned {role} agent", "INFO")
        return agent

    # -------------------------
    # TASK ORCHESTRATION
    # -------------------------

    def run(self, objective: str) -> Dict[str, Any]:
        """
        Entry point for ZeroGravity execution
        """
        self.log(f"Starting execution for objective: {objective}", "INFO")
        
        self.state["objective"] = objective
        self.state["history"] = []
        self.state["start_time"] = datetime.datetime.utcnow().isoformat()

        try:
            # Phase 1: Architecture
            self.log("Starting Architect phase", "INFO")
            architect = self.spawn_agent("architect")
            plan = architect.execute(objective)
            self._record("architect", plan)
            self.log("Completed Architect phase", "INFO")

            # Phase 2: Engineering
            self.log("Starting Engineer phase", "INFO")
            engineer = self.spawn_agent("engineer")
            implementation = engineer.execute(plan)
            self._record("engineer", implementation)
            self.log("Completed Engineer phase", "INFO")

            # Phase 3: Design (optional / parallel later)
            self.log("Starting Designer phase", "INFO")
            designer = self.spawn_agent("designer")
            design = designer.execute(plan)
            self._record("designer", design)
            self.log("Completed Designer phase", "INFO")

            # Phase 4: Operator
            self.log("Starting Operator phase", "INFO")
            operator = self.spawn_agent("operator")
            result = operator.execute(implementation)
            self._record("operator", result)
            self.log("Completed Operator phase", "INFO")

            # Calculate execution time
            end_time = datetime.datetime.utcnow().isoformat()
            self.state["end_time"] = end_time
            
            execution_summary = {
                "workflow_status": "completed",
                "execution_summary": f"Workflow completed successfully in session {self.session_id}"
            }

            final_result = {
                "objective": objective,
                "result": result,
                "history": self.state["history"],
                "execution_summary": execution_summary
            }

            self.log("Workflow completed successfully", "INFO")
            return final_result

        except Exception as e:
            self.log(f"Workflow failed with error: {str(e)}", "ERROR")
            # Record error in history
            error_entry = {
                "role": "error",
                "output": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "status": "failed"
            }
            self.state["history"].append(error_entry)
            
            return {
                "objective": objective,
                "result": {"error": str(e)},
                "history": self.state["history"],
                "execution_summary": {
                    "workflow_status": "failed",
                    "execution_summary": f"Workflow failed in session {self.session_id}: {str(e)}"
                }
            }

    # -------------------------
    # MEMORY & LOGGING
    # -------------------------

    def _record(self, role: str, output: Any):
        self.state["history"].append({
            "role": role,
            "output": output,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

    def log(self, message: str, level: str = "INFO"):
        """
        Enhanced logging functionality
        """
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "session_id": self.session_id
        }
        self.log_entries.append(log_entry)
        
        # Print to console for immediate feedback
        print(f"[{level}] {message}")

    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieve all log entries
        """
        return self.log_entries

    def save_logs(self, filepath: str = None) -> str:
        """
        Save logs to a file
        """
        if not filepath:
            # Replace invalid characters for file names (like colons in ISO format)
            safe_session_id = self.session_id.replace(":", "_").replace("+", "_")
            filepath = f"zerogravity_logs_{safe_session_id}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_entries, f, indent=2)
        
        self.log(f"Logs saved to {filepath}", "INFO")
        return filepath
