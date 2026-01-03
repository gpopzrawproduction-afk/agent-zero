# System-wide AI Ops Enforcement Plan

This document outlines the code-level integration plan to make every existing agent subordinate to AI Ops, removing any direct execution paths and forcing all execution requests through AI Ops APIs.

## Current Agent Architecture Overview

### Existing Agents
- **Architect**: Already integrated with AI Ops (✅)
- **Engineer**: Direct execution path needs to be removed
- **Designer**: Direct execution path needs to be removed
- **Operator**: Direct execution path needs to be removed
- **Coordinator**: Direct execution path needs to be removed

### Integration Requirements
1. Remove direct execution paths from all agents
2. Integrate AI Ops hooks into all agents
3. Force execution requests through AI Ops APIs
4. Ensure AI Ops becomes unavoidable

## Code-Level Integration Plan

### 1. Base Agent Class Modifications

The base agent class needs to be updated to enforce AI Ops integration:

```python
# zero_gravity_core/agents/base.py

class BaseAgent:
    def __init__(self, name, ai_ops_client=None):
        self.name = name
        self.ai_ops_client = ai_ops_client or self._get_ai_ops_client()
        self.agent_id = self._register_with_ai_ops()
    
    def _get_ai_ops_client(self):
        """Get AI Ops client instance"""
        from zero_gravity_core.agents.ai_ops_integration import AIOpsClient
        return AIOpsClient()
    
    def _register_with_ai_ops(self):
        """Register agent with AI Ops before any execution"""
        if not self.ai_ops_client:
            raise Exception("AI Ops client is required for agent registration")
        return self.ai_ops_client.register_agent({
            'name': self.name,
            'type': self.__class__.__name__,
            'capabilities': self.get_capabilities(),
            'resource_requirements': self.get_resource_requirements()
        })
    
    def execute_with_approval(self, task_description, priority=1):
        """Execute task only after AI Ops approval"""
        approval = self.ai_ops_client.request_approval({
            'agent_id': self.agent_id,
            'task_description': task_description,
            'priority': priority,
            'estimated_resources': self.estimate_resources(task_description),
            'estimated_time': self.estimate_execution_time(task_description)
        })
        
        if not approval['approved']:
            raise Exception(f"Execution denied by AI Ops: {approval.get('reason', 'Policy violation')}")
        
        # Execute with telemetry emission
        try:
            result = self._execute_task(task_description)
            
            # Emit telemetry
            self.ai_ops_client.emit_telemetry({
                'agent_id': self.agent_id,
                'task_id': approval['task_id'],
                'result': result,
                'execution_time': self.get_execution_time(),
                'resources_used': self.get_resource_usage(),
                'cost': self.calculate_cost()
            })
            
            # Await evaluation
            evaluation = self.ai_ops_client.await_evaluation(approval['task_id'])
            
            return result
        except Exception as e:
            # Emit error telemetry
            self.ai_ops_client.emit_telemetry({
                'agent_id': self.agent_id,
                'task_id': approval['task_id'],
                'error': str(e),
                'execution_time': self.get_execution_time(),
                'resources_used': self.get_resource_usage(),
                'cost': self.calculate_cost()
            })
            raise e
    
    def _execute_task(self, task_description):
        """Override this method in child classes"""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    def get_capabilities(self):
        """Override this method in child classes"""
        raise NotImplementedError("Subclasses must implement get_capabilities")
    
    def get_resource_requirements(self):
        """Override this method in child classes"""
        return {'cpu': 1, 'memory': '512MB', 'gpu': False}
    
    def estimate_resources(self, task_description):
        """Override this method in child classes"""
        return self.get_resource_requirements()
    
    def estimate_execution_time(self, task_description):
        """Override this method in child classes"""
        return 300  # Default 5 minutes
    
    def get_execution_time(self):
        """Override this method in child classes"""
        return 0
    
    def get_resource_usage(self):
        """Override this method in child classes"""
        return {'cpu_percent': 0, 'memory_mb': 0}
    
    def calculate_cost(self):
        """Override this method in child classes"""
        return 0.0
```

### 2. Individual Agent Modifications

#### Engineer Agent
- Modify `zero_gravity_core/agents/engineer.py` to inherit from BaseAgent
- Implement required methods
- Remove direct execution paths

#### Designer Agent
- Modify `zero_gravity_core/agents/designer.py` to inherit from BaseAgent
- Implement required methods
- Remove direct execution paths

#### Operator Agent
- Modify `zero_gravity_core/agents/operator.py` to inherit from BaseAgent
- Implement required methods
- Remove direct execution paths

#### Coordinator Agent
- Modify `zero_gravity_core/agents/coordinator.py` to inherit from BaseAgent
- Implement required methods
- Remove direct execution paths

### 3. AI Ops Client Implementation

```python
# zero_gravity_core/agents/ai_ops_integration.py

import requests
import json
from typing import Dict, Any

class AIOpsClient:
    def __init__(self, ai_ops_url="http://localhost:8000/ai_ops"):
        self.ai_ops_url = ai_ops_url
    
    def register_agent(self, agent_data: Dict[str, Any]) -> str:
        """Register agent with AI Ops"""
        response = requests.post(f"{self.ai_ops_url}/register_agent", json=agent_data)
        response.raise_for_status()
        return response.json()['agent_id']
    
    def request_approval(self, execution_request: Dict[str, Any]) -> Dict[str, Any]:
        """Request execution approval from AI Ops"""
        response = requests.post(f"{self.ai_ops_url}/request_approval", json=execution_request)
        response.raise_for_status()
        return response.json()
    
    def emit_telemetry(self, telemetry_data: Dict[str, Any]) -> None:
        """Emit telemetry data to AI Ops"""
        response = requests.post(f"{self.ai_ops_url}/emit_telemetry", json=telemetry_data)
        response.raise_for_status()
    
    def await_evaluation(self, task_id: str) -> Dict[str, Any]:
        """Await evaluation from AI Ops"""
        response = requests.get(f"{self.ai_ops_url}/evaluation/{task_id}")
        response.raise_for_status()
        return response.json()
```

### 4. AI Ops API Endpoints

```python
# zero_gravity_core/agents/ai_ops.py (enhanced)

from flask import Flask, request, jsonify
import uuid
from datetime import datetime
from zero_gravity_core.engines.policy_engine import PolicyEngine
from zero_gravity_core.engines.decision_engine import DecisionEngine
from zero_gravity_core.engines.optimization_engine import OptimizationEngine
from zero_gravity_core.controllers.escalation_controller import EscalationController
from zero_gravity_core.monitoring.telemetry_collector import TelemetryCollector

app = Flask(__name__)

# In-memory storage (replace with database in production)
agents = {}
executions = {}
telemetry_data = []

@app.route('/register_agent', methods=['POST'])
def register_agent():
    """Register a new agent with AI Ops"""
    agent_data = request.json
    agent_id = str(uuid.uuid4())
    
    agent_data['agent_id'] = agent_id
    agent_data['registration_time'] = datetime.utcnow().isoformat()
    agent_data['status'] = 'registered'
    
    agents[agent_id] = agent_data
    
    return jsonify({
        'agent_id': agent_id,
        'status': 'registered',
        'message': 'Agent registered successfully'
    })

@app.route('/request_approval', methods=['POST'])
def request_approval():
    """Request execution approval for an agent"""
    execution_request = request.json
    agent_id = execution_request.get('agent_id')
    
    # Verify agent exists
    if agent_id not in agents:
        return jsonify({
            'approved': False,
            'reason': 'Agent not registered with AI Ops',
            'task_id': None
        }), 400
    
    # Check policy compliance
    policy_engine = PolicyEngine()
    policy_check = policy_engine.evaluate_execution_request(execution_request)
    
    if not policy_check['compliant']:
        return jsonify({
            'approved': False,
            'reason': policy_check['reason'],
            'task_id': None
        })
    
    # Make decision using decision engine
    decision_engine = DecisionEngine()
    decision = decision_engine.make_execution_decision(execution_request)
    
    task_id = str(uuid.uuid4())
    execution_request['task_id'] = task_id
    execution_request['status'] = 'approved' if decision['approved'] else 'denied'
    execution_request['approval_time'] = datetime.utcnow().isoformat()
    
    executions[task_id] = execution_request
    
    return jsonify({
        'approved': decision['approved'],
        'reason': decision.get('reason', ''),
        'task_id': task_id,
        'estimated_cost': decision.get('estimated_cost', 0.0),
        'priority': decision.get('priority', 1)
    })

@app.route('/emit_telemetry', methods=['POST'])
def emit_telemetry():
    """Receive telemetry data from agents"""
    telemetry = request.json
    task_id = telemetry.get('task_id')
    
    # Verify execution exists
    if task_id and task_id in executions:
        executions[task_id]['telemetry'] = telemetry
    
    telemetry['timestamp'] = datetime.utcnow().isoformat()
    telemetry_data.append(telemetry)
    
    # Process with telemetry collector
    collector = TelemetryCollector()
    collector.process_telemetry(telemetry)
    
    # Run optimization engine if needed
    optimization_engine = OptimizationEngine()
    optimization_engine.process_telemetry(telemetry)
    
    return jsonify({'status': 'received'})

@app.route('/evaluation/<task_id>', methods=['GET'])
def get_evaluation(task_id):
    """Get evaluation for a completed task"""
    if task_id not in executions:
        return jsonify({'error': 'Task not found'}), 404
    
    execution = executions[task_id]
    
    # Generate evaluation
    evaluation = {
        'task_id': task_id,
        'agent_id': execution['agent_id'],
        'performance_score': calculate_performance_score(execution),
        'compliance_status': check_compliance_status(execution),
        'optimization_suggestions': get_optimization_suggestions(execution),
        'future_approval_status': determine_future_approval_status(execution)
    }
    
    return jsonify(evaluation)

def calculate_performance_score(execution):
    """Calculate performance score for execution"""
    # Implementation based on telemetry data
    return 85  # Placeholder

def check_compliance_status(execution):
    """Check compliance status for execution"""
    # Implementation based on policy checks
    return 'compliant'  # Placeholder

def get_optimization_suggestions(execution):
    """Get optimization suggestions based on execution"""
    # Implementation based on optimization engine
    return []  # Placeholder

def determine_future_approval_status(execution):
    """Determine future approval status for agent"""
    # Implementation based on past performance
    return 'approved'  # Placeholder

# Additional API endpoints for dashboard and monitoring
@app.route('/active_agents', methods=['GET'])
def get_active_agents():
    """Get list of active agents"""
    active = []
    for agent_id, agent_data in agents.items():
        active.append({
            'agent_id': agent_id,
            'name': agent_data.get('name'),
            'type': agent_data.get('type'),
            'status': agent_data.get('status'),
            'last_activity': agent_data.get('last_activity')
        })
    return jsonify(active)

@app.route('/running_tasks', methods=['GET'])
def get_running_tasks():
    """Get list of running tasks"""
    running = []
    for task_id, execution in executions.items():
        if execution.get('status') == 'approved' and not execution.get('telemetry'):
            running.append({
                'task_id': task_id,
                'agent_id': execution.get('agent_id'),
                'task_description': execution.get('task_description'),
                'start_time': execution.get('approval_time')
            })
    return jsonify(running)
```

### 5. Migration Plan

#### Phase 1: Base Class Implementation
1. Update `zero_gravity_core/agents/base.py` with AI Ops enforcement
2. Test base class functionality
3. Verify registration and approval workflows

#### Phase 2: Individual Agent Updates
1. Update Engineer agent to use base class
2. Update Designer agent to use base class
3. Update Operator agent to use base class
4. Update Coordinator agent to use base class

#### Phase 3: API Integration
1. Enhance AI Ops API endpoints
2. Implement telemetry processing
3. Add monitoring and alerting

#### Phase 4: Testing and Validation
1. End-to-end testing of enforcement
2. Performance testing
3. Validation of all agents following lifecycle

### 6. Validation Checklist

- [ ] All agents register with AI Ops before execution
- [ ] All execution requests go through approval process
- [ ] All executions emit telemetry
- [ ] All agents await evaluation
- [ ] Direct execution paths are removed
- [ ] AI Ops becomes unavoidable
- [ ] Existing functionality preserved
- [ ] Performance impact minimized
- [ ] Error handling implemented
- [ ] Audit trails maintained

## Expected Outcome

After implementation:
- AI Ops becomes unavoidable for all agents
- All agents follow the mandatory lifecycle
- Complete telemetry coverage
- Centralized governance and control
- Scalable architecture for future agents
