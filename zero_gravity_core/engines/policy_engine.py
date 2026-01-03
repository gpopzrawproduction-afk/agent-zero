# Policy Engine for AI Ops Agent
# Implements declarative policy system for budget, compliance, security, and performance SLAs

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict
import operator

from zero_gravity_core.schemas.ai_ops_schemas import PolicyConfig, PolicyViolation

logger = logging.getLogger(__name__)

class PolicyEngine:
    """
    The Policy Engine implements a declarative policy system that enforces:
    - Budget caps
    - Model usage rules
    - Compliance constraints
    - Data access boundaries
    - Rate limits per agent
    """
    
    def __init__(self):
        self.policies: List[PolicyConfig] = []
        self.policy_cache = {}  # Cache for compiled conditions
        
        # Define supported operators for condition evaluation
        self.operators = {
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
            'in': lambda x, y: x in y,
            'not in': lambda x, y: x not in y,
            'contains': lambda x, y: y in x,
            'matches': lambda x, y: bool(re.search(y, str(x))) if isinstance(y, str) else False
        }
        
        logger.info("Policy Engine initialized")
    
    def add_policy(self, policy: PolicyConfig):
        """Add a new policy to the system"""
        self.policies.append(policy)
        logger.info(f"Added policy: {policy.name} for agent {policy.agent}")
    
    def remove_policy(self, policy_name: str):
        """Remove a policy from the system"""
        self.policies = [p for p in self.policies if p.name != policy_name]
        logger.info(f"Removed policy: {policy_name}")
    
    def update_policy(self, policy_name: str, updated_policy: PolicyConfig):
        """Update an existing policy"""
        for i, policy in enumerate(self.policies):
            if policy.name == policy_name:
                self.policies[i] = updated_policy
                logger.info(f"Updated policy: {policy_name}")
                return
        logger.warning(f"Policy not found for update: {policy_name}")
    
    async def check_policies(self, agent_name: str, task_data: Dict[str, Any]) -> List[PolicyViolation]:
        """
        Check if the task violates any policies for the specified agent
        """
        violations = []
        
        # Filter policies that apply to this agent
        applicable_policies = [
            p for p in self.policies 
            if p.enabled and (p.agent == agent_name or p.agent == "*")
        ]
        
        # Sort by priority (lower numbers are higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        for policy in applicable_policies:
            try:
                # Evaluate the policy condition
                condition_result = await self._evaluate_condition(policy.condition, task_data)
                
                if condition_result:
                    # Condition is true, which means policy is violated
                    violation = PolicyViolation(
                        policy_name=policy.name,
                        agent_name=agent_name,
                        condition=policy.condition,
                        action=policy.action,
                        value=condition_result,
                        timestamp=asyncio.get_event_loop().time(),
                        severity="high" if policy.action == "block" else "medium"
                    )
                    violations.append(violation)
                    
                    logger.warning(f"Policy violation detected: {violation}")
            except Exception as e:
                logger.error(f"Error evaluating policy {policy.name}: {str(e)}")
        
        return violations
    
    async def _evaluate_condition(self, condition: str, task_data: Dict[str, Any]) -> bool:
        """
        Evaluate a policy condition against task data
        Supports conditions like: "cost > 1.00", "priority < 3", "model in ['gpt-4', 'claude-2']"
        """
        # Check if condition is already compiled and cached
        cache_key = (condition, tuple(sorted(task_data.items())))
        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]
        
        # Parse the condition
        # This is a simplified parser - in a production system, you'd want a more robust solution
        # that safely evaluates expressions without using eval()
        
        # Common pattern: field operator value
        # Examples: "cost > 1.00", "priority < 3", "model == 'gpt-4'"
        
        # First, try to match the pattern: field operator value
        import re
        
        # This regex matches patterns like: field operator value
        # Examples: cost > 1.00, priority < 3, model == 'gpt-4'
        pattern = r'(\w+)\s*([=!<>]+|in|not in|contains|matches)\s*(.+)'
        match = re.match(pattern, condition.strip())
        
        if match:
            field, op, value = match.groups()
            field = field.strip()
            op = op.strip()
            value = value.strip()
            
            # Clean up value (remove quotes if present)
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            elif value.startswith('[') and value.endswith(']'):
                # Handle list values like ['gpt-4', 'claude-2']
                value = [v.strip().strip("'\"") for v in value[1:-1].split(',')]
            
            # Get the field value from task data
            field_value = task_data.get(field)
            
            # Convert value to appropriate type if needed
            if field_value is not None:
                if isinstance(field_value, (int, float)) and value.replace('.', '', 1).isdigit():
                    try:
                        value = float(value) if '.' in value else int(value)
                    except ValueError:
                        pass  # Keep as string if conversion fails
            
            # Apply the operator
            if op in self.operators:
                result = self.operators[op](field_value, value)
                self.policy_cache[cache_key] = result
                return result
            else:
                logger.error(f"Unsupported operator in condition: {op}")
                return False
        else:
            # For more complex conditions, we could implement a more sophisticated parser
            # For now, return False for unparseable conditions
            logger.warning(f"Could not parse condition: {condition}")
            return False
    
    async def get_agent_policies(self, agent_name: str) -> List[PolicyConfig]:
        """Get all policies that apply to a specific agent"""
        return [
            p for p in self.policies 
            if p.agent == agent_name or p.agent == "*"
        ]
    
    async def get_policy_status(self) -> Dict[str, Any]:
        """Get overall policy engine status"""
        active_policies = [p for p in self.policies if p.enabled]
        disabled_policies = [p for p in self.policies if not p.enabled]
        
        return {
            "total_policies": len(self.policies),
            "active_policies": len(active_policies),
            "disabled_policies": len(disabled_policies),
            "agents_with_policies": len(set(p.agent for p in self.policies if p.agent != "*")),
            "global_policies": len([p for p in self.policies if p.agent == "*"])
        }
    
    async def enforce_budget_policy(self, agent_name: str, current_cost: float, budget_limit: float) -> bool:
        """
        Helper method to enforce budget policies specifically
        Returns True if the action is allowed, False if it violates budget
        """
        budget_violations = []
        
        # Check for budget-related policies
        budget_policies = [
            p for p in self.policies 
            if p.agent == agent_name or p.agent == "*"
            if "cost" in p.condition.lower() and "budget" in p.condition.lower()
        ]
        
        for policy in budget_policies:
            # Evaluate if the current cost would violate the budget policy
            condition_with_values = policy.condition.replace("cost", str(current_cost)).replace("budget", str(budget_limit))
            condition_result = await self._evaluate_condition(condition_with_values, {
                "cost": current_cost,
                "budget": budget_limit
            })
            
            if condition_result:
                budget_violations.append(policy)
        
        # If there are budget violations, return False
        return len(budget_violations) == 0
    
    async def get_policy_recommendations(self, agent_name: str, task_data: Dict[str, Any]) -> List[str]:
        """
        Get policy-related recommendations to avoid violations
        """
        recommendations = []
        
        # Check which policies might be violated and suggest how to avoid them
        applicable_policies = [
            p for p in self.policies 
            if p.enabled and (p.agent == agent_name or p.agent == "*")
        ]
        
        for policy in applicable_policies:
            try:
                # Evaluate the condition to see if it would be violated
                condition_result = await self._evaluate_condition(policy.condition, task_data)
                
                if condition_result:
                    # Policy would be violated, suggest how to avoid it
                    if policy.action == "downgrade_model":
                        recommendations.append(f"Avoid '{policy.name}' violation: Consider using a less expensive model")
                    elif policy.action == "escalate":
                        recommendations.append(f"Avoid '{policy.name}' violation: Consider escalating this task")
                    elif policy.action == "block":
                        recommendations.append(f"Avoid '{policy.name}' violation: Modify task parameters to comply with policy")
                    elif policy.action == "queue_for_batch_processing":
                        recommendations.append(f"Avoid '{policy.name}' violation: Schedule for batch processing instead of immediate execution")
            except Exception as e:
                logger.error(f"Error getting recommendation for policy {policy.name}: {str(e)}")
        
        return recommendations
    
    async def validate_policy_config(self, policy: PolicyConfig) -> List[str]:
        """
        Validate a policy configuration before adding it
        Returns a list of validation errors
        """
        errors = []
        
        # Validate policy name
        if not policy.name or not policy.name.strip():
            errors.append("Policy name is required")
        
        # Validate agent field
        if not policy.agent or not policy.agent.strip():
            errors.append("Agent field is required (use '*' for all agents)")
        
        # Validate condition
        if not policy.condition or not policy.condition.strip():
            errors.append("Condition is required")
        else:
            # Try to parse the condition to make sure it's valid
            try:
                # This is a basic check - try to match the condition pattern
                pattern = r'(\w+)\s*([=!<>]+|in|not in|contains|matches)\s*(.+)'
                if not re.match(pattern, policy.condition.strip()):
                    errors.append(f"Condition format is invalid: {policy.condition}")
            except:
                errors.append(f"Condition format is invalid: {policy.condition}")
        
        # Validate action
        valid_actions = ["block", "downgrade_model", "escalate", "queue_for_batch_processing", "log", "notify"]
        if policy.action not in valid_actions:
            errors.append(f"Invalid action '{policy.action}'. Valid actions: {valid_actions}")
        
        # Validate description
        if not policy.description or not policy.description.strip():
            errors.append("Description is required")
        
        return errors
    
    async def apply_policy_action(self, violation: PolicyViolation, context: Dict[str, Any] = None) -> bool:
        """
        Apply the action specified in a policy violation
        Returns True if the action was successfully applied, False otherwise
        """
        action = violation.action
        logger.info(f"Applying policy action '{action}' for violation: {violation.policy_name}")
        
        if action == "block":
            logger.warning(f"Blocking action due to policy violation: {violation.policy_name}")
            return False  # Action should be blocked
        elif action == "downgrade_model":
            logger.info(f"Downgrading model due to policy violation: {violation.policy_name}")
            # In a real implementation, this would modify the model selection
            if context:
                context['model_downgraded'] = True
            return True
        elif action == "escalate":
            logger.info(f"Elevating task due to policy violation: {violation.policy_name}")
            # In a real implementation, this would trigger escalation
            if context:
                context['escalated'] = True
            return True
        elif action == "queue_for_batch_processing":
            logger.info(f"Queuing for batch processing due to policy violation: {violation.policy_name}")
            # In a real implementation, this would modify task scheduling
            if context:
                context['batch_processed'] = True
            return True
        elif action == "log":
            logger.info(f"Logging action due to policy violation: {violation.policy_name}")
            return True
        elif action == "notify":
            logger.info(f"Notifying stakeholders due to policy violation: {violation.policy_name}")
            # In a real implementation, this would send notifications
            return True
        else:
            logger.warning(f"Unknown policy action: {action}")
            return True  # Default to allowing the action for unknown actions
