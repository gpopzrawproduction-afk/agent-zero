"""
Workflow Graph System for ZeroGravity

This module implements workflow graphs to represent agent dependencies
and enable parallel execution of tasks where possible.
"""
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from collections import defaultdict, deque


class TaskStatus(Enum):
    """Status of a workflow task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Type of node in the workflow graph"""
    AGENT = "agent"
    CONDITION = "condition"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow graph"""
    id: str
    node_type: NodeType
    name: str
    agent_role: Optional[str] = None  # For agent nodes
    dependencies: List[str] = field(default_factory=list)  # Node IDs this node depends on
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    execution_order: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            # Generate a unique ID based on name and creation time
            self.id = hashlib.md5(f"{self.name}_{datetime.utcnow().isoformat()}".encode()).hexdigest()


@dataclass
class WorkflowEdge:
    """Represents an edge in the workflow graph"""
    source: str  # Source node ID
    target: str  # Target node ID
    condition: Optional[str] = None  # Condition for the edge (for conditional workflows)
    data_key: Optional[str] = None  # Key for data passing between nodes


class WorkflowGraph:
    """Represents a workflow as a directed acyclic graph (DAG)"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[WorkflowEdge] = []
        self.created_at = datetime.utcnow()
        self.id = hashlib.md5(f"{name}_{self.created_at.isoformat()}".encode()).hexdigest()
    
    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow graph"""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the workflow graph"""
        # Validate that both nodes exist
        if edge.source not in self.nodes:
            raise ValueError(f"Source node {edge.source} does not exist in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node {edge.target} does not exist in graph")
        
        # Add the target to the source node's dependencies
        if edge.target not in self.nodes[edge.source].dependencies:
            self.nodes[edge.target].dependencies.append(edge.source)
        
        self.edges.append(edge)
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all dependencies for a node"""
        if node_id not in self.nodes:
            return []
        return self.nodes[node_id].dependencies[:]
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node"""
        dependents = []
        for edge in self.edges:
            if edge.source == node_id:
                dependents.append(edge.target)
        return dependents
    
    def get_ready_nodes(self) -> List[str]:
        """Get nodes that are ready to execute (all dependencies satisfied)"""
        ready_nodes = []
        for node_id, node in self.nodes.items():
            if node.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            all_deps_satisfied = True
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    if self.nodes[dep_id].status != TaskStatus.COMPLETED:
                        all_deps_satisfied = False
                        break
            
            if all_deps_satisfied:
                ready_nodes.append(node_id)
        
        return ready_nodes
    
    def get_execution_order(self) -> List[str]:
        """Get a valid execution order for the workflow using topological sort"""
        # Build adjacency list
        adj_list = defaultdict(list)
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for edge in self.edges:
            adj_list[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for neighbor in adj_list[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check if there's a cycle
        if len(result) != len(self.nodes):
            raise ValueError("Workflow contains a cycle - cannot determine execution order")
        
        return result
    
    def can_execute_parallel(self, node1_id: str, node2_id: str) -> bool:
        """Check if two nodes can be executed in parallel"""
        # Two nodes can run in parallel if:
        # 1. They don't have a direct dependency relationship
        # 2. They don't share a common dependent that requires their results together
        
        # Check direct dependency
        if node2_id in self.nodes[node1_id].dependencies or node1_id in self.nodes[node2_id].dependencies:
            return False
        
        # Check if they have common dependents that require both results
        dependents1 = set(self.get_dependents(node1_id))
        dependents2 = set(self.get_dependents(node2_id))
        
        # If they have common dependents, check if those dependents need both results
        common_dependents = dependents1.intersection(dependents2)
        
        for dep_id in common_dependents:
            # If the common dependent requires both as dependencies, they can't run in parallel
            node1_in_deps = node1_id in self.nodes[dep_id].dependencies
            node2_in_deps = node2_id in self.nodes[dep_id].dependencies
            
            if node1_in_deps and node2_in_deps:
                return False
        
        return True
    
    def get_parallel_groups(self) -> List[List[str]]:
        """Group nodes that can be executed in parallel"""
        execution_order = self.get_execution_order()
        groups = []
        
        for node_id in execution_order:
            # Find which group this node can be added to
            added = False
            for group in groups:
                # Check if this node can be in the same group as others
                can_be_together = True
                for other_node_id in group:
                    if not self.can_execute_parallel(node_id, other_node_id):
                        can_be_together = False
                        break
                
                if can_be_together:
                    group.append(node_id)
                    added = True
                    break
            
            if not added:
                groups.append([node_id])
        
        return groups
    
    def update_node_status(self, node_id: str, status: TaskStatus, 
                          outputs: Dict[str, Any] = None, error: str = None) -> None:
        """Update the status of a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist in graph")
        
        node = self.nodes[node_id]
        node.status = status
        
        if outputs is not None:
            node.outputs = outputs
        
        if error is not None:
            node.error = error
        
        if status == TaskStatus.RUNNING and node.start_time is None:
            node.start_time = datetime.utcnow()
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            node.end_time = datetime.utcnow()
    
    def is_complete(self) -> bool:
        """Check if the entire workflow is complete"""
        return all(node.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                  for node in self.nodes.values())
    
    def get_workflow_status(self) -> TaskStatus:
        """Get the overall status of the workflow"""
        nodes = list(self.nodes.values())
        
        if not nodes:
            return TaskStatus.COMPLETED  # Empty workflow is considered complete
        
        if any(node.status == TaskStatus.FAILED for node in nodes):
            return TaskStatus.FAILED
        
        if any(node.status == TaskStatus.CANCELLED for node in nodes):
            return TaskStatus.CANCELLED
        
        if all(node.status == TaskStatus.COMPLETED for node in nodes):
            return TaskStatus.COMPLETED
        
        if any(node.status == TaskStatus.RUNNING for node in nodes):
            return TaskStatus.RUNNING
        
        return TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "nodes": {node_id: {
                "id": node.id,
                "node_type": node.node_type.value,
                "name": node.name,
                "agent_role": node.agent_role,
                "dependencies": node.dependencies,
                "inputs": node.inputs,
                "outputs": node.outputs,
                "status": node.status.value,
                "execution_order": node.execution_order,
                "start_time": node.start_time.isoformat() if node.start_time else None,
                "end_time": node.end_time.isoformat() if node.end_time else None,
                "error": node.error,
                "metadata": node.metadata
            } for node_id, node in self.nodes.items()},
            "edges": [{
                "source": edge.source,
                "target": edge.target,
                "condition": edge.condition,
                "data_key": edge.data_key
            } for edge in self.edges]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowGraph':
        """Create workflow from dictionary"""
        workflow = cls(data["name"], data["description"])
        workflow.id = data["id"]
        workflow.created_at = datetime.fromisoformat(data["created_at"])
        
        # Create nodes
        for node_data in data["nodes"].values():
            node = WorkflowNode(
                id=node_data["id"],
                node_type=NodeType(node_data["node_type"]),
                name=node_data["name"],
                agent_role=node_data["agent_role"],
                dependencies=node_data["dependencies"],
                inputs=node_data["inputs"],
                outputs=node_data["outputs"],
                status=TaskStatus(node_data["status"]),
                execution_order=node_data["execution_order"],
                start_time=datetime.fromisoformat(node_data["start_time"]) if node_data["start_time"] else None,
                end_time=datetime.fromisoformat(node_data["end_time"]) if node_data["end_time"] else None,
                error=node_data["error"],
                metadata=node_data["metadata"]
            )
            workflow.add_node(node)
        
        # Create edges
        for edge_data in data["edges"]:
            edge = WorkflowEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                condition=edge_data["condition"],
                data_key=edge_data["data_key"]
            )
            workflow.add_edge(edge)
        
        return workflow


class WorkflowExecutor:
    """Executes workflows based on the workflow graph"""
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowGraph] = {}
    
    def create_workflow_from_objective(self, objective: str) -> WorkflowGraph:
        """Create a default workflow based on the objective type"""
        # Analyze the objective to determine the appropriate workflow
        workflow = WorkflowGraph(f"workflow_{hashlib.md5(objective.encode()).hexdigest()[:8]}", 
                               f"Workflow for: {objective[:50]}...")
        
        # Default workflow: Architect -> (Engineer, Designer) -> Operator
        architect_node = WorkflowNode(
            id="architect",
            node_type=NodeType.AGENT,
            name="Architect",
            agent_role="architect",
            inputs={"objective": objective}
        )
        
        engineer_node = WorkflowNode(
            id="engineer",
            node_type=NodeType.AGENT,
            name="Engineer",
            agent_role="engineer",
            dependencies=["architect"]
        )
        
        designer_node = WorkflowNode(
            id="designer",
            node_type=NodeType.AGENT,
            name="Designer",
            agent_role="designer",
            dependencies=["architect"]
        )
        
        operator_node = WorkflowNode(
            id="operator",
            node_type=NodeType.AGENT,
            name="Operator",
            agent_role="operator",
            dependencies=["engineer"]  # Operator depends on engineer, not designer
        )
        
        workflow.add_node(architect_node)
        workflow.add_node(engineer_node)
        workflow.add_node(designer_node)
        workflow.add_node(operator_node)
        
        # Add edges to represent data flow
        workflow.add_edge(WorkflowEdge("architect", "engineer"))
        workflow.add_edge(WorkflowEdge("architect", "designer"))
        workflow.add_edge(WorkflowEdge("engineer", "operator"))
        
        return workflow
    
    def execute_workflow(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """Execute a workflow graph"""
        from zero_gravity_core.agents.coordinator import Coordinator
        from zero_gravity_core.utils.logging import main_logger
        from zero_gravity_core.task_queue.celery_app import execute_agent_task
        
        # Add to active workflows
        self.active_workflows[workflow.id] = workflow
        
        try:
            # Get parallel execution groups
            parallel_groups = workflow.get_parallel_groups()
            
            main_logger.info(
                "Starting workflow execution",
                workflow_id=workflow.id,
                workflow_name=workflow.name,
                parallel_groups=len(parallel_groups)
            )
            
            # Execute each group sequentially, but nodes within each group in parallel
            for group_idx, group in enumerate(parallel_groups):
                main_logger.info(
                    "Executing workflow group",
                    workflow_id=workflow.id,
                    group_index=group_idx,
                    nodes=group
                )
                
                # Execute nodes in this group in parallel
                for node_id in group:
                    node = workflow.nodes[node_id]
                    
                    if node.node_type == NodeType.AGENT and node.agent_role:
                        # Execute agent task
                        workflow.update_node_status(node_id, TaskStatus.RUNNING)
                        
                        # In a real implementation, this would be an async call
                        # For now, we'll simulate execution
                        coordinator = Coordinator()
                        agent = coordinator.spawn_agent(node.agent_role)
                        
                        try:
                            # Prepare input data
                            input_data = node.inputs.copy()
                            
                            # Get data from dependencies
                            for dep_id in node.dependencies:
                                dep_node = workflow.nodes[dep_id]
                                # Pass outputs from dependencies as inputs
                                input_data.update(dep_node.outputs)
                            
                            # Execute the agent
                            result = agent.execute_with_llm(input_data)
                            
                            # Update node with results
                            workflow.update_node_status(
                                node_id, 
                                TaskStatus.COMPLETED, 
                                outputs={"result": result}
                            )
                            
                            main_logger.info(
                                "Agent task completed",
                                workflow_id=workflow.id,
                                node_id=node_id,
                                agent_role=node.agent_role
                            )
                            
                        except Exception as e:
                            workflow.update_node_status(
                                node_id, 
                                TaskStatus.FAILED, 
                                error=str(e)
                            )
                            main_logger.error(
                                "Agent task failed",
                                workflow_id=workflow.id,
                                node_id=node_id,
                                agent_role=node.agent_role,
                                error=str(e)
                            )
                            # In a real implementation, you might want to stop execution on failure
                            # or implement retry logic
                    else:
                        # For non-agent nodes, just mark as completed
                        workflow.update_node_status(node_id, TaskStatus.COMPLETED)
            
            # Final status
            final_status = workflow.get_workflow_status()
            main_logger.info(
                "Workflow execution completed",
                workflow_id=workflow.id,
                final_status=final_status.value
            )
            
            return {
                "status": final_status.value,
                "workflow_id": workflow.id,
                "results": {node_id: node.outputs for node_id, node in workflow.nodes.items()},
                "execution_summary": self._get_execution_summary(workflow)
            }
            
        finally:
            # Remove from active workflows
            if workflow.id in self.active_workflows:
                del self.active_workflows[workflow.id]
    
    def _get_execution_summary(self, workflow: WorkflowGraph) -> Dict[str, Any]:
        """Get a summary of workflow execution"""
        completed = len([n for n in workflow.nodes.values() if n.status == TaskStatus.COMPLETED])
        failed = len([n for n in workflow.nodes.values() if n.status == TaskStatus.FAILED])
        running = len([n for n in workflow.nodes.values() if n.status == TaskStatus.RUNNING])
        pending = len([n for n in workflow.nodes.values() if n.status == TaskStatus.PENDING])
        
        total_time = None
        if workflow.nodes:
            start_times = [n.start_time for n in workflow.nodes.values() if n.start_time]
            end_times = [n.end_time for n in workflow.nodes.values() if n.end_time]
            
            if start_times and end_times:
                earliest_start = min(start_times)
                latest_end = max(end_times)
                total_time = (latest_end - earliest_start).total_seconds()
        
        return {
            "total_nodes": len(workflow.nodes),
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "total_time_seconds": total_time
        }


# Global workflow executor instance
workflow_executor = WorkflowExecutor()
