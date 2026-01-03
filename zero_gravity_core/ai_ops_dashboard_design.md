# AI Ops Dashboard Advanced Design

This document outlines the design for an advanced AI Ops dashboard that provides comprehensive visibility into the ZeroGravity system, turning it into a command center for AI operations.

## Dashboard Overview

### Purpose
The AI Ops Dashboard serves as the central command center for:
- Monitoring all active agents
- Tracking running tasks and their status
- Real-time cost burn visualization
- Failure rate analysis
- Optimization actions taken by AI Ops
- Escalations triggered by the system

### Target Users
- System administrators
- AI Ops engineers
- Business stakeholders
- Security and compliance officers

## Dashboard Components

### 1. Active Agents Panel
**Purpose**: Show all currently active agents in the system

**Components**:
- Agent name and type
- Current status (idle, running, paused, error)
- Last activity timestamp
- Resource utilization (CPU, memory, network)
- Performance score
- Agent health indicator

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/active_agents.html -->
<div class="dashboard-panel">
  <h3>Active Agents</h3>
  <div class="agent-grid">
    <div class="agent-card" data-agent-id="{{agent.id}}">
      <div class="agent-header">
        <span class="agent-name">{{agent.name}}</span>
        <span class="agent-type">{{agent.type}}</span>
        <span class="agent-status {{agent.status}}">{{agent.status}}</span>
      </div>
      <div class="agent-metrics">
        <div class="metric">
          <label>CPU</label>
          <div class="progress-bar">
            <div class="progress" style="width: {{agent.cpu_percent}}%"></div>
          </div>
          <span>{{agent.cpu_percent}}%</span>
        </div>
        <div class="metric">
          <label>Memory</label>
          <div class="progress-bar">
            <div class="progress" style="width: {{agent.memory_percent}}%"></div>
          </div>
          <span>{{agent.memory_mb}}MB</span>
        </div>
      <div class="agent-details">
        <span>Last activity: {{agent.last_activity}}</span>
        <span>Performance: {{agent.performance_score}}</span>
      </div>
    </div>
  </div>
</div>
```

### 2. Running Tasks Panel
**Purpose**: Display all currently executing tasks

**Components**:
- Task ID and description
- Agent executing the task
- Start time and elapsed time
- Estimated completion time
- Current status and progress
- Priority level
- Resource consumption

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/running_tasks.html -->
<div class="dashboard-panel">
  <h3>Running Tasks</h3>
  <table class="tasks-table">
    <thead>
      <tr>
        <th>Task ID</th>
        <th>Description</th>
        <th>Agent</th>
        <th>Start Time</th>
        <th>Elapsed</th>
        <th>Status</th>
        <th>Progress</th>
        <th>Priority</th>
      </tr>
    </thead>
    <tbody>
      <tr data-task-id="{{task.id}}">
        <td>{{task.id}}</td>
        <td class="task-description">{{task.description}}</td>
        <td>{{task.agent_name}}</td>
        <td>{{task.start_time}}</td>
        <td>{{task.elapsed_time}}</td>
        <td><span class="status-badge {{task.status}}">{{task.status}}</span></td>
        <td>
          <div class="progress-container">
            <div class="progress-bar" style="width: {{task.progress}}%"></div>
            <span class="progress-text">{{task.progress}}%</span>
          </div>
        </td>
        <td><span class="priority-badge priority-{{task.priority}}">{{task.priority}}</span></td>
      </tr>
    </tbody>
  </table>
</div>
```

### 3. Cost Burn Visualization
**Purpose**: Real-time visualization of system costs

**Components**:
- Current cost per minute/hour/day
- Cost trend visualization
- Budget utilization percentage
- Cost breakdown by agent type
- Projected costs
- Cost optimization opportunities

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/cost_burn.html -->
<div class="dashboard-panel">
  <h3>Cost Burn</h3>
  <div class="cost-summary">
    <div class="cost-metric">
      <h4>Current Hour</h4>
      <span class="cost-value">${{current_hour_cost}}</span>
    </div>
    <div class="cost-metric">
      <h4>Today</h4>
      <span class="cost-value">${{today_cost}}</span>
    </div>
    <div class="cost-metric">
      <h4>Budget Utilization</h4>
      <div class="budget-progress">
        <div class="progress-bar" style="width: {{budget_utilization}}%"></div>
        <span>{{budget_utilization}}%</span>
      </div>
    </div>
  </div>
 <div class="cost-chart-container">
    <canvas id="costChart"></canvas>
 </div>
  <div class="cost-breakdown">
    <h4>Cost Breakdown by Agent</h4>
    <div class="breakdown-items">
      <div class="breakdown-item">
        <span class="agent-type">Research Agent</span>
        <span class="cost-amount">${{research_agent_cost}}</span>
        <div class="cost-bar" style="width: {{research_agent_percentage}}%"></div>
      </div>
      <!-- Additional agent cost breakdowns -->
    </div>
  </div>
</div>
```

### 4. Failure Rates Panel
**Purpose**: Track and visualize failure rates across the system

**Components**:
- Overall failure rate
- Failure rate by agent type
- Failure rate trend over time
- Most common failure types
- Failure correlation with other metrics
- Mean time to recovery

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/failure_rates.html -->
<div class="dashboard-panel">
  <h3>Failure Rates</h3>
  <div class="failure-summary">
    <div class="failure-metric">
      <h4>Current Rate</h4>
      <span class="failure-value">{{current_failure_rate}}%</span>
    </div>
    <div class="failure-metric">
      <h4>24h Trend</h4>
      <span class="trend-indicator {{trend_direction}}">{{trend_value}}</span>
    </div>
    <div class="failure-metric">
      <h4>MTTR</h4>
      <span class="mttr-value">{{mean_time_to_recovery}}s</span>
    </div>
  </div>
  <div class="failure-chart-container">
    <canvas id="failureChart"></canvas>
 </div>
  <div class="failure-breakdown">
    <h4>Failures by Type</h4>
    <table class="failures-table">
      <thead>
        <tr>
          <th>Failure Type</th>
          <th>Count</th>
          <th>Rate</th>
          <th>Last Occurred</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{{failure.type}}</td>
          <td>{{failure.count}}</td>
          <td>{{failure.rate}}%</td>
          <td>{{failure.last_occurred}}</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
```

### 5. Optimization Actions Panel
**Purpose**: Display actions taken by AI Ops for optimization

**Components**:
- Count of optimization actions taken
- Types of optimizations performed
- Performance improvements achieved
- Cost savings realized
- Confidence levels of optimizations
- Timeline of optimization actions

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/optimization_actions.html -->
<div class="dashboard-panel">
  <h3>Optimization Actions</h3>
  <div class="optimization-summary">
    <div class="optimization-metric">
      <h4>Actions Taken</h4>
      <span class="action-count">{{total_optimization_actions}}</span>
    </div>
    <div class="optimization-metric">
      <h4>Performance Improvement</h4>
      <span class="improvement-value">+{{performance_improvement}}%</span>
    </div>
    <div class="optimization-metric">
      <h4>Cost Savings</h4>
      <span class="savings-value">${{cost_savings}}</span>
    </div>
  </div>
  <div class="optimization-timeline">
    <h4>Recent Actions</h4>
    <ul class="action-list">
      <li class="action-item">
        <span class="action-time">{{action.timestamp}}</span>
        <span class="action-type">{{action.type}}</span>
        <span class="action-description">{{action.description}}</span>
        <span class="confidence-badge">Confidence: {{action.confidence}}%</span>
      </li>
    </ul>
  </div>
  <div class="optimization-breakdown">
    <h4>Optimizations by Type</h4>
    <div class="optimization-types">
      <div class="optimization-type">
        <span class="type-name">{{type.name}}</span>
        <span class="type-count">{{type.count}}</span>
        <div class="type-bar" style="width: {{type.percentage}}%"></div>
      </div>
    </div>
  </div>
</div>
```

### 6. Escalations Panel
**Purpose**: Track and manage system escalations

**Components**:
- Active escalations
- Escalation severity levels
- Escalation reasons
- Time to resolution
- Escalation trends
- Manual intervention required

**Implementation**:
```html
<!-- zero_gravity_core/dashboard/components/escalations.html -->
<div class="dashboard-panel">
  <h3>Escalations</h3>
  <div class="escalation-summary">
    <div class="escalation-metric">
      <h4>Active</h4>
      <span class="escalation-count">{{active_escalations}}</span>
    </div>
    <div class="escalation-metric">
      <h4>Resolved Today</h4>
      <span class="resolved-count">{{resolved_today}}</span>
    </div>
    <div class="escalation-metric">
      <h4>Avg. Resolution Time</h4>
      <span class="resolution-time">{{avg_resolution_time}}m</span>
    </div>
  </div>
  <div class="escalation-list">
    <h4>Active Escalations</h4>
    <table class="escalations-table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Type</th>
          <th>Severity</th>
          <th>Reason</th>
          <th>Time Open</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr class="escalation-row severity-{{escalation.severity}}">
          <td>{{escalation.id}}</td>
          <td>{{escalation.type}}</td>
          <td><span class="severity-badge severity-{{escalation.severity}}">{{escalation.severity}}</span></td>
          <td>{{escalation.reason}}</td>
          <td>{{escalation.time_open}}</td>
          <td><span class="status-badge {{escalation.status}}">{{escalation.status}}</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
```

## Dashboard API Endpoints

### Backend Implementation
```python
# zero_gravity_core/dashboard/api.py

from flask import Flask, jsonify, request
from zero_gravity_core.agents.ai_ops import agents, executions, telemetry_data
from zero_gravity_core.engines.optimization_engine import get_optimization_actions
from zero_gravity_core.controllers.escalation_controller import get_active_escalations
import datetime

app = Flask(__name__)

@app.route('/dashboard/active_agents', methods=['GET'])
def get_active_agents():
    """Get list of active agents with their status and metrics"""
    active_agents = []
    for agent_id, agent_data in agents.items():
        if agent_data.get('status') == 'registered':  # Currently active
            active_agents.append({
                'id': agent_id,
                'name': agent_data.get('name'),
                'type': agent_data.get('type'),
                'status': agent_data.get('status'),
                'cpu_percent': agent_data.get('cpu_percent', 0),
                'memory_mb': agent_data.get('memory_mb', 0),
                'last_activity': agent_data.get('last_activity', ''),
                'performance_score': agent_data.get('performance_score', 0)
            })
    
    return jsonify(active_agents)

@app.route('/dashboard/running_tasks', methods=['GET'])
def get_running_tasks():
    """Get list of currently running tasks"""
    running_tasks = []
    for task_id, execution in executions.items():
        if execution.get('status') == 'approved' and not execution.get('telemetry'):
            # Calculate progress if possible
            progress = 0  # Placeholder - would calculate based on execution metrics
            running_tasks.append({
                'id': task_id,
                'description': execution.get('task_description', ''),
                'agent_name': get_agent_name_by_id(execution.get('agent_id')),
                'start_time': execution.get('approval_time', ''),
                'elapsed_time': calculate_elapsed_time(execution.get('approval_time')),
                'status': execution.get('status', ''),
                'progress': progress,
                'priority': execution.get('priority', 1)
            })
    
    return jsonify(running_tasks)

@app.route('/dashboard/cost_burn', methods=['GET'])
def get_cost_burn():
    """Get current cost burn metrics"""
    # Calculate cost metrics from telemetry data
    current_hour_cost = calculate_current_hour_cost()
    today_cost = calculate_today_cost()
    budget_utilization = calculate_budget_utilization()
    
    # Breakdown by agent type
    cost_breakdown = get_cost_breakdown_by_agent()
    
    return jsonify({
        'current_hour_cost': current_hour_cost,
        'today_cost': today_cost,
        'budget_utilization': budget_utilization,
        'cost_breakdown': cost_breakdown
    })

@app.route('/dashboard/failure_rates', methods=['GET'])
def get_failure_rates():
    """Get failure rate metrics"""
    # Calculate failure metrics from telemetry data
    current_failure_rate = calculate_current_failure_rate()
    failure_trend = calculate_failure_trend()
    mttr = calculate_mean_time_to_recovery()
    
    # Breakdown by failure type
    failure_breakdown = get_failure_breakdown_by_type()
    
    return jsonify({
        'current_failure_rate': current_failure_rate,
        'failure_trend': failure_trend,
        'mttr': mttr,
        'failure_breakdown': failure_breakdown
    })

@app.route('/dashboard/optimization_actions', methods=['GET'])
def get_optimization_actions():
    """Get optimization actions taken by AI Ops"""
    actions = get_optimization_actions()
    
    # Calculate summary metrics
    total_actions = len(actions)
    performance_improvement = calculate_performance_improvement()
    cost_savings = calculate_cost_savings()
    
    return jsonify({
        'total_actions': total_actions,
        'performance_improvement': performance_improvement,
        'cost_savings': cost_savings,
        'recent_actions': actions[:10],  # Last 10 actions
        'actions_by_type': group_actions_by_type(actions)
    })

@app.route('/dashboard/escalations', methods=['GET'])
def get_escalations():
    """Get escalation metrics"""
    active_escalations = get_active_escalations()
    
    # Calculate summary metrics
    resolved_today = calculate_resolved_today()
    avg_resolution_time = calculate_avg_resolution_time()
    
    return jsonify({
        'active_escalations': len(active_escalations),
        'resolved_today': resolved_today,
        'avg_resolution_time': avg_resolution_time,
        'escalations': active_escalations
    })

def get_agent_name_by_id(agent_id):
    """Helper function to get agent name by ID"""
    agent = agents.get(agent_id)
    return agent.get('name', 'Unknown') if agent else 'Unknown'

def calculate_elapsed_time(start_time_str):
    """Calculate elapsed time from start time string"""
    try:
        start_time = datetime.datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        elapsed = datetime.datetime.now(datetime.timezone.utc) - start_time
        return str(elapsed)
    except:
        return "Unknown"

def calculate_current_hour_cost():
    """Calculate cost for current hour"""
    # Implementation to calculate cost for current hour
    return 0.0  # Placeholder

def calculate_today_cost():
    """Calculate cost for today"""
    # Implementation to calculate cost for today
    return 0.0  # Placeholder

def calculate_budget_utilization():
    """Calculate budget utilization percentage"""
    # Implementation to calculate budget utilization
    return 0.0  # Placeholder

def get_cost_breakdown_by_agent():
    """Get cost breakdown by agent type"""
    # Implementation to get cost breakdown
    return []  # Placeholder

def calculate_current_failure_rate():
    """Calculate current failure rate"""
    # Implementation to calculate failure rate
    return 0.0  # Placeholder

def calculate_failure_trend():
    """Calculate failure trend"""
    # Implementation to calculate trend
    return {'direction': 'neutral', 'value': 0.0}  # Placeholder

def calculate_mean_time_to_recovery():
    """Calculate mean time to recovery"""
    # Implementation to calculate MTTR
    return 0.0  # Placeholder

def get_failure_breakdown_by_type():
    """Get failure breakdown by type"""
    # Implementation to get failure breakdown
    return []  # Placeholder

def calculate_performance_improvement():
    """Calculate performance improvement from optimizations"""
    # Implementation to calculate improvement
    return 0.0  # Placeholder

def calculate_cost_savings():
    """Calculate cost savings from optimizations"""
    # Implementation to calculate savings
    return 0.0  # Placeholder

def group_actions_by_type(actions):
    """Group optimization actions by type"""
    # Implementation to group actions
    return {}  # Placeholder

def calculate_resolved_today():
    """Calculate number of escalations resolved today"""
    # Implementation to calculate resolved count
    return 0  # Placeholder

def calculate_avg_resolution_time():
    """Calculate average resolution time for escalations"""
    # Implementation to calculate avg time
    return 0.0  # Placeholder
```

## Dashboard UI Implementation

### Main Dashboard Page
```html
<!-- zero_gravity_core/dashboard/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZeroGravity AI Ops Dashboard</title>
    <link rel="stylesheet" href="styles.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>ZeroGravity AI Ops Dashboard</h1>
            <div class="header-controls">
                <button id="refresh-btn" class="btn btn-primary">Refresh</button>
                <div class="time-range-selector">
                    <select id="time-range">
                        <option value="1h">Last Hour</option>
                        <option value="24h" selected>Last 24 Hours</option>
                        <option value="7d">Last 7 Days</option>
                        <option value="30d">Last 30 Days</option>
                    </select>
                </div>
            </div>
        </header>
        
        <div class="dashboard-grid">
            <div class="dashboard-row">
                <div class="dashboard-col">
                    <div id="active-agents-panel" class="dashboard-panel">
                        <!-- Active Agents Panel Content -->
                    </div>
                <div class="dashboard-col">
                    <div id="running-tasks-panel" class="dashboard-panel">
                        <!-- Running Tasks Panel Content -->
                    </div>
                </div>
            
            <div class="dashboard-row">
                <div class="dashboard-col wide">
                    <div id="cost-burn-panel" class="dashboard-panel">
                        <!-- Cost Burn Panel Content -->
                    </div>
                </div>
            
            <div class="dashboard-row">
                <div class="dashboard-col">
                    <div id="failure-rates-panel" class="dashboard-panel">
                        <!-- Failure Rates Panel Content -->
                    </div>
                <div class="dashboard-col">
                    <div id="escalations-panel" class="dashboard-panel">
                        <!-- Escalations Panel Content -->
                    </div>
                </div>
            
            <div class="dashboard-row">
                <div class="dashboard-col wide">
                    <div id="optimization-actions-panel" class="dashboard-panel">
                        <!-- Optimization Actions Panel Content -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="dashboard.js"></script>
</body>
</html>
```

### Dashboard CSS
```css
/* zero_gravity_core/dashboard/styles.css */
:root {
    --primary-color: #2563eb;
    --secondary-color: #64748b;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --background-color: #f8fafc;
    --panel-background: #ffffff;
    --text-color: #1e293b;
    --border-color: #e2e8f0;
}

body {
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}

.dashboard-container {
    padding: 20px;
    min-height: 100vh;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid var(--border-color);
}

.dashboard-header h1 {
    margin: 0;
    color: var(--primary-color);
}

.header-controls {
    display: flex;
    gap: 15px;
    align-items: center;
}

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 500;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
}

.dashboard-grid {
    display: grid;
    gap: 20px;
}

.dashboard-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.dashboard-col.wide {
    grid-column: 1 / -1;
}

.dashboard-panel {
    background-color: var(--panel-background);
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.dashboard-panel h3 {
    margin-top: 0;
    margin-bottom: 15px;
    color: var(--primary-color);
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 10px;
}

.agent-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 15px;
}

.agent-card {
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 15px;
}

.agent-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.agent-status {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
}

.agent-status.idle { background-color: #dbeafe; color: #1d4ed8; }
.agent-status.running { background-color: #d1fae5; color: #065f46; }
.agent-status.paused { background-color: #fef3c7; color: #92400e; }
.agent-status.error { background-color: #fee2e2; color: #b91c1c; }

.metric {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}

.metric label {
    width: 60px;
    font-size: 0.9em;
    color: var(--secondary-color);
}

.progress-bar {
    flex: 1;
    height: 10px;
    background-color: #e5e7eb;
    border-radius: 5px;
    overflow: hidden;
    margin: 0 10px;
}

.progress {
    height: 100%;
    background-color: var(--primary-color);
    transition: width 0.3s ease;
}

.metric span {
    width: 50px;
    text-align: right;
    font-size: 0.9em;
}

.agent-details {
    display: flex;
    justify-content: space-between;
    font-size: 0.85em;
    color: var(--secondary-color);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border-color);
}

.tasks-table {
    width: 100%;
    border-collapse: collapse;
}

.tasks-table th,
.tasks-table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.tasks-table th {
    background-color: #f1f5f9;
    font-weight: 600;
}

.status-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
}

.status-badge.pending { background-color: #fef3c7; color: #92400e; }
.status-badge.running { background-color: #d1fae5; color: #065f46; }
.status-badge.completed { background-color: #dbeafe; color: #1d4ed8; }
.status-badge.failed { background-color: #fee2e2; color: #b91c1c; }

.progress-container {
    display: flex;
    align-items: center;
}

.progress-text {
    margin-left: 10px;
    font-size: 0.9em;
    min-width: 40px;
}

.priority-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
}

.priority-1 { background-color: #dcfce7; color: #166534; }
.priority-2 { background-color: #fef3c7; color: #92400e; }
.priority-3 { background-color: #fee2e2; color: #b91c1c; }

.cost-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.cost-metric {
    text-align: center;
    padding: 15px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
}

.cost-metric h4 {
    margin: 0 0 10px 0;
    color: var(--secondary-color);
    font-size: 1em;
}

.cost-value {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--primary-color);
}

.budget-progress {
    margin-top: 10px;
    position: relative;
}

.failure-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.failure-metric {
    text-align: center;
    padding: 15px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
}

.failure-metric h4 {
    margin: 0 0 10px 0;
    color: var(--secondary-color);
    font-size: 1em;
}

.failure-value {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--danger-color);
}

.trend-indicator {
    font-size: 1.2em;
    font-weight: 600;
}

.trend-indicator.up { color: var(--danger-color); }
.trend-indicator.down { color: var(--success-color); }
.trend-indicator.neutral { color: var(--secondary-color); }

.optimization-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.optimization-metric {
    text-align: center;
    padding: 15px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
}

.optimization-metric h4 {
    margin: 0 0 10px 0;
    color: var(--secondary-color);
    font-size: 1em;
}

.action-count,
.improvement-value,
.savings-value {
    font-size: 1.5em;
    font-weight: 600;
}

.action-count { color: var(--primary-color); }
.improvement-value { color: var(--success-color); }
.savings-value { color: var(--success-color); }

.optimization-timeline {
    margin-bottom: 20px;
}

.action-list {
    list-style: none;
    padding: 0;
}

.action-item {
    padding: 10px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
}

.action-time {
    font-weight: 600;
    color: var(--primary-color);
}

.action-type {
    background-color: #e0e7ff;
    color: #3730a3;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
}

.confidence-badge {
    background-color: #ddd6fe;
    color: #6d28d9;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
}

.escalation-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.escalation-metric {
    text-align: center;
    padding: 15px;
    border: 1px solid var(--border-color);
    border-radius: 6px;
}

.escalation-metric h4 {
    margin: 0 0 10px 0;
    color: var(--secondary-color);
    font-size: 1em;
}

.escalation-count,
.resolved-count {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--danger-color);
}

.resolution-time {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--warning-color);
}

.escalations-table {
    width: 100%;
    border-collapse: collapse;
}

.escalations-table th,
.escalations-table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

.escalations-table th {
    background-color: #f1f5f9;
    font-weight: 600;
}

.severity-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: 500;
}

.severity-low { background-color: #dcfce7; color: #166534; }
.severity-medium { background-color: #fef3c7; color: #92400e; }
.severity-high { background-color: #fed7d7; color: #b91c1c; }
.severity-critical { background-color: #fee2e2; color: #b91c1c; }

.cost-chart-container,
.failure-chart-container {
    height: 300px;
    margin: 20px 0;
}

.breakdown-items,
.optimization-types {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.breakdown-item,
.optimization-type {
    display: flex;
    align-items: center;
    padding: 8px;
    border-bottom: 1px solid var(--border-color);
}

.breakdown-item .agent-type,
.optimization-type .type-name {
    flex: 1;
    font-weight: 500;
}

.breakdown-item .cost-amount,
.optimization-type .type-count {
    width: 80px;
    text-align: right;
    font-weight: 600;
}

.cost-bar,
.type-bar {
    height: 8px;
    background-color: var(--primary-color);
    border-radius: 4px;
    margin-left: 15px;
    flex: 1;
}
```

### Dashboard JavaScript
```javascript
// zero_gravity_core/dashboard/dashboard.js
class AIOpsDashboard {
    constructor() {
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.loadDashboardData();
        this.setupEventListeners();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadDashboardData();
        });

        document.getElementById('time-range').addEventListener('change', (e) => {
            this.loadDashboardData();
        });
    }

    async loadDashboardData() {
        try {
            // Load all dashboard components
            await Promise.all([
                this.loadActiveAgents(),
                this.loadRunningTasks(),
                this.loadCostBurn(),
                this.loadFailureRates(),
                this.loadOptimizationActions(),
                this.loadEscalations()
            ]);
            
            console.log('Dashboard data loaded successfully');
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    async loadActiveAgents() {
        try {
            const response = await fetch('/dashboard/active_agents');
            const agents = await response.json();
            
            const container = document.getElementById('active-agents-panel');
            container.innerHTML = this.renderActiveAgents(agents);
        } catch (error) {
            console.error('Error loading active agents:', error);
        }
    }

    async loadRunningTasks() {
        try {
            const response = await fetch('/dashboard/running_tasks');
            const tasks = await response.json();
            
            const container = document.getElementById('running-tasks-panel');
            container.innerHTML = this.renderRunningTasks(tasks);
        } catch (error) {
            console.error('Error loading running tasks:', error);
        }
    }

    async loadCostBurn() {
        try {
            const response = await fetch('/dashboard/cost_burn');
            const costData = await response.json();
            
            const container = document.getElementById('cost-burn-panel');
            container.innerHTML = this.renderCostBurn(costData);
            
            // Initialize cost chart
            this.initCostChart(costData);
        } catch (error) {
            console.error('Error loading cost burn data:', error);
        }
    }

    async loadFailureRates() {
        try {
            const response = await fetch('/dashboard/failure_rates');
            const failureData = await response.json();
            
            const container = document.getElementById('failure-rates-panel');
            container.innerHTML = this.renderFailureRates(failureData);
            
            // Initialize failure chart
            this.initFailureChart(failureData);
        } catch (error) {
            console.error('Error loading failure rates:', error);
        }
    }

    async loadOptimizationActions() {
        try {
            const response = await fetch('/dashboard/optimization_actions');
            const optimizationData = await response.json();
            
            const container = document.getElementById('optimization-actions-panel');
            container.innerHTML = this.renderOptimizationActions(optimizationData);
        } catch (error) {
            console.error('Error loading optimization actions:', error);
        }
    }

    async loadEscalations() {
        try {
            const response = await fetch('/dashboard/escalations');
            const escalationData = await response.json();
            
            const container = document.getElementById('escalations-panel');
            container.innerHTML = this.renderEscalations(escalationData);
        } catch (error) {
            console.error('Error loading escalations:', error);
        }
    }

    renderActiveAgents(agents) {
        if (!agents || agents.length === 0) {
            return '<div class="dashboard-panel"><h3>Active Agents</h3><p>No active agents found.</p></div>';
        }

        const agentCards = agents.map(agent => `
            <div class="agent-card" data-agent-id="${agent.id}">
                <div class="agent-header">
                    <span class="agent-name">${agent.name}</span>
                    <span class="agent-type">${agent.type}</span>
                    <span class="agent-status ${agent.status}">${agent.status}</span>
                </div>
                <div class="agent-metrics">
                    <div class="metric">
                        <label>CPU</label>
                        <div class="progress-bar">
                            <div class="progress" style="width: ${agent.cpu_percent || 0}%"></div>
                        </div>
                        <span>${agent.cpu_percent || 0}%</span>
                    </div>
                    <div class="metric">
                        <label>Memory</label>
                        <div class="progress-bar">
                            <div class="progress" style="width: ${(agent.memory_mb / 1024) * 100 || 0}%"></div>
                        </div>
                        <span>${agent.memory_mb || 0}MB</span>
                    </div>
                </div>
                <div class="agent-details">
                    <span>Last activity: ${agent.last_activity || 'N/A'}</span>
                    <span>Performance: ${agent.performance_score || 0}</span>
                </div>
            </div>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Active Agents</h3>
                <div class="agent-grid">
                    ${agentCards}
                </div>
            </div>
        `;
    }

    renderRunningTasks(tasks) {
        if (!tasks || tasks.length === 0) {
            return '<div class="dashboard-panel"><h3>Running Tasks</h3><p>No running tasks found.</p></div>';
        }

        const taskRows = tasks.map(task => `
            <tr data-task-id="${task.id}">
                <td>${task.id}</td>
                <td class="task-description">${task.description}</td>
                <td>${task.agent_name}</td>
                <td>${task.start_time}</td>
                <td>${task.elapsed_time}</td>
                <td><span class="status-badge ${task.status}">${task.status}</span></td>
                <td>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: ${task.progress || 0}%"></div>
                        <span class="progress-text">${task.progress || 0}%</span>
                    </div>
                </td>
                <td><span class="priority-badge priority-${task.priority}">${task.priority}</span></td>
            </tr>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Running Tasks</h3>
                <table class="tasks-table">
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Description</th>
                            <th>Agent</th>
                            <th>Start Time</th>
                            <th>Elapsed</th>
                            <th>Status</th>
                            <th>Progress</th>
                            <th>Priority</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${taskRows}
                    </tbody>
                </table>
            </div>
        `;
    }

    renderCostBurn(costData) {
        const breakdownItems = (costData.cost_breakdown || []).map(breakdown => `
            <div class="breakdown-item">
                <span class="agent-type">${breakdown.agent_type || 'Unknown'}</span>
                <span class="cost-amount">$${breakdown.cost || 0}</span>
                <div class="cost-bar" style="width: ${breakdown.percentage || 0}%"></div>
            </div>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Cost Burn</h3>
                <div class="cost-summary">
                    <div class="cost-metric">
                        <h4>Current Hour</h4>
                        <span class="cost-value">$${costData.current_hour_cost || 0}</span>
                    </div>
                    <div class="cost-metric">
                        <h4>Today</h4>
                        <span class="cost-value">$${costData.today_cost || 0}</span>
                    </div>
                    <div class="cost-metric">
                        <h4>Budget Utilization</h4>
                        <div class="budget-progress">
                            <div class="progress-bar" style="width: ${costData.budget_utilization || 0}%"></div>
                            <span>${costData.budget_utilization || 0}%</span>
                        </div>
                    </div>
                <div class="cost-chart-container">
                    <canvas id="costChart"></canvas>
                </div>
                <div class="cost-breakdown">
                    <h4>Cost Breakdown by Agent</h4>
                    <div class="breakdown-items">
                        ${breakdownItems}
                    </div>
                </div>
            </div>
        `;
    }

    renderFailureRates(failureData) {
        const failureRows = (failureData.failure_breakdown || []).map(failure => `
            <tr>
                <td>${failure.type || 'Unknown'}</td>
                <td>${failure.count || 0}</td>
                <td>${failure.rate || 0}%</td>
                <td>${failure.last_occurred || 'N/A'}</td>
            </tr>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Failure Rates</h3>
                <div class="failure-summary">
                    <div class="failure-metric">
                        <h4>Current Rate</h4>
                        <span class="failure-value">${failureData.current_failure_rate || 0}%</span>
                    </div>
                    <div class="failure-metric">
                        <h4>24h Trend</h4>
                        <span class="trend-indicator ${failureData.failure_trend?.direction || 'neutral'}">
                            ${failureData.failure_trend?.value || 0}%
                        </span>
                    </div>
                    <div class="failure-metric">
                        <h4>MTTR</h4>
                        <span class="mttr-value">${failureData.mttr || 0}s</span>
                    </div>
                </div>
                <div class="failure-chart-container">
                    <canvas id="failureChart"></canvas>
                </div>
                <div class="failure-breakdown">
                    <h4>Failures by Type</h4>
                    <table class="failures-table">
                        <thead>
                            <tr>
                                <th>Failure Type</th>
                                <th>Count</th>
                                <th>Rate</th>
                                <th>Last Occurred</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${failureRows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    renderOptimizationActions(optimizationData) {
        const actionItems = (optimizationData.recent_actions || []).slice(0, 10).map(action => `
            <li class="action-item">
                <span class="action-time">${action.timestamp || 'N/A'}</span>
                <span class="action-type">${action.type || 'Unknown'}</span>
                <span class="action-description">${action.description || 'No description'}</span>
                <span class="confidence-badge">Confidence: ${action.confidence || 0}%</span>
            </li>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Optimization Actions</h3>
                <div class="optimization-summary">
                    <div class="optimization-metric">
                        <h4>Actions Taken</h4>
                        <span class="action-count">${optimizationData.total_actions || 0}</span>
                    </div>
                    <div class="optimization-metric">
                        <h4>Performance Improvement</h4>
                        <span class="improvement-value">+${optimizationData.performance_improvement || 0}%</span>
                    </div>
                    <div class="optimization-metric">
                        <h4>Cost Savings</h4>
                        <span class="savings-value">$${optimizationData.cost_savings || 0}</span>
                    </div>
                </div>
                <div class="optimization-timeline">
                    <h4>Recent Actions</h4>
                    <ul class="action-list">
                        ${actionItems}
                    </ul>
                </div>
                <div class="optimization-breakdown">
                    <h4>Optimizations by Type</h4>
                    <div class="optimization-types">
                        ${(Object.entries(optimizationData.actions_by_type || {}).map(([type, count]) => `
                            <div class="optimization-type">
                                <span class="type-name">${type}</span>
                                <span class="type-count">${count}</span>
                                <div class="type-bar" style="width: ${(count / Math.max(...Object.values(optimizationData.actions_by_type || {0: 0})) * 100) || 0}%"></div>
                            </div>
                        `).join(''))}
                    </div>
                </div>
            </div>
        `;
    }

    renderEscalations(escalationData) {
        const escalationRows = (escalationData.escalations || []).map(escalation => `
            <tr class="escalation-row severity-${escalation.severity || 'low'}">
                <td>${escalation.id || 'N/A'}</td>
                <td>${escalation.type || 'Unknown'}</td>
                <td><span class="severity-badge severity-${escalation.severity || 'low'}">${escalation.severity || 'low'}</span></td>
                <td>${escalation.reason || 'No reason'}</td>
                <td>${escalation.time_open || 'N/A'}</td>
                <td><span class="status-badge ${escalation.status || 'open'}">${escalation.status || 'open'}</span></td>
            </tr>
        `).join('');

        return `
            <div class="dashboard-panel">
                <h3>Escalations</h3>
                <div class="escalation-summary">
                    <div class="escalation-metric">
                        <h4>Active</h4>
                        <span class="escalation-count">${escalationData.active_escalations || 0}</span>
                    </div>
                    <div class="escalation-metric">
                        <h4>Resolved Today</h4>
                        <span class="resolved-count">${escalationData.resolved_today || 0}</span>
                    </div>
                    <div class="escalation-metric">
                        <h4>Avg. Resolution Time</h4>
                        <span class="resolution-time">${escalationData.avg_resolution_time || 0}m</span>
                    </div>
                </div>
                <div class="escalation-list">
                    <h4>Active Escalations</h4>
                    <table class="escalations-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Type</th>
                                <th>Severity</th>
                                <th>Reason</th>
                                <th>Time Open</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${escalationRows}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    initCostChart(costData) {
        const ctx = document.getElementById('costChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (window.costChart) {
            window.costChart.destroy();
        }

        // Create new chart
        window.costChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['1H', '2H', '3H', '4H', '5H', '6H'],
                datasets: [{
                    label: 'Cost per Hour',
                    data: [0.1, 0.15, 0.2, 0.18, 0.22, 0.25],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Cost Trend (Last 6 Hours)'
                    }
                }
            }
        });
    }

    initFailureChart(failureData) {
        const ctx = document.getElementById('failureChart');
        if (!ctx) return;

        // Destroy existing chart if it exists
        if (window.failureChart) {
            window.failureChart.destroy();
        }

        // Create new chart
        window.failureChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['1H', '2H', '3H', '4H', '5H', '6H'],
                datasets: [{
                    label: 'Failure Count',
                    data: [2, 1, 3, 0, 2, 1],
                    backgroundColor: '#ef4444',
                    borderColor: '#b91c1c',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Failure Rate Trend (Last 6 Hours)'
                    }
                }
            }
        });
    }

    startAutoRefresh() {
        // Refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    showError(message) {
        // Create and show error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #fee2e2;
            color: #b91c1c;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #b91c1c;
            z-index: 1000;
        `;
        
        document.body.appendChild(errorDiv);
        
        // Remove after 5 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AIOpsDashboard();
});
```

## Implementation Plan

### Phase 1: Backend API Implementation
1. Implement the dashboard API endpoints in `zero_gravity_core/dashboard/api.py`
2. Connect to existing AI Ops data sources
3. Test API endpoints for each dashboard component

### Phase 2: Frontend Implementation
1. Create the main dashboard HTML structure
2. Implement CSS styling for the dashboard
3. Create JavaScript for data loading and visualization
4. Integrate Chart.js for data visualization

### Phase 3: Integration and Testing
1. Connect frontend to backend API
2. Test all dashboard components
3. Validate real-time updates
4. Optimize performance for large datasets

### Phase 4: Advanced Features
1. Add filtering and time range selection
2. Implement alerting and notification systems
3. Add export functionality for reports
4. Enhance with additional visualization types

## Expected Outcomes

- Complete visibility into ZeroGravity system operations
- Real-time monitoring of all agents and tasks
- Cost visibility and optimization opportunities
- Failure tracking and resolution metrics
- AI Ops optimization impact measurement
- Escalation management and tracking

This dashboard will transform ZeroGravity from a collection of AI agents into a command center for AI operations, providing the visibility needed to run entire business units with minimal human oversight.
