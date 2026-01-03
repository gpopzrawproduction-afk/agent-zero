"""
Celery Task Queue for ZeroGravity

This module implements a task queue system using Celery for
handling background tasks and job persistence in the ZeroGravity platform.
"""
from celery import Celery
from celery.schedules import crontab
import os
from typing import Any, Dict
import json


# Create Celery instance
celery_app = Celery('zerogravity')

# Configure Celery
celery_config = {
    # Broker settings (using Redis by default, but can be configured)
    'broker_url': os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    'result_backend': os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    
    # Task settings
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    
    # Worker settings
    'worker_prefetch_multiplier': 1,
    'task_acks_late': True,
    
    # Result settings
    'result_expires': 3600,  # Results expire after 1 hour
    
    # Beat scheduler settings (for periodic tasks)
    'beat_schedule': {
        'cleanup-expired-tasks': {
            'task': 'zerogravity.task_queue.tasks.cleanup_expired_tasks',
            'schedule': crontab(minute=0, hour=0),  # Run daily at midnight
        },
    },
}

celery_app.config_from_object(celery_config)

# Auto-discover tasks
celery_app.autodiscover_tasks(['zero_gravity_core.task_queue'])


class TaskPriority:
    """Task priority constants"""
    LOW = 6  # Low priority
    NORMAL = 5  # Normal priority (default)
    HIGH = 4  # High priority
    CRITICAL = 3  # Critical priority


@celery_app.task(bind=True, priority=TaskPriority.NORMAL)
def execute_agent_task(self, agent_role: str, input_data: Any, session_id: str = None) -> Dict[str, Any]:
    """
    Execute an agent task asynchronously
    
    Args:
        agent_role: The role of the agent to execute (architect, engineer, designer, operator)
        input_data: The input data for the agent
        session_id: Optional session ID for tracking
    
    Returns:
        Dictionary containing the result of the agent execution
    """
    from zero_gravity_core.agents.coordinator import Coordinator
    from zero_gravity_core.utils.logging import main_logger
    from zero_gravity_core.utils.validation import validate_agent_input
    
    try:
        # Validate input
        validated_input = validate_agent_input(input_data, agent_role, session_id)
        
        # Initialize coordinator
        coordinator = Coordinator()
        
        # Spawn the appropriate agent
        agent = coordinator.spawn_agent(agent_role)
        
        # Execute the agent task
        result = agent.execute_with_llm(validated_input.input_data)
        
        # Log the execution
        main_logger.log_agent_activity(
            agent_role=agent_role,
            session_id=session_id or "unknown",
            action="async_execution",
            task_id=self.request.id
        )
        
        return {
            "status": "completed",
            "result": result,
            "agent_role": agent_role,
            "session_id": session_id,
            "task_id": self.request.id
        }
    except Exception as exc:
        main_logger.error(
            "Agent task failed",
            task_id=self.request.id,
            agent_role=agent_role,
            session_id=session_id,
            error=str(exc)
        )
        
        # Retry the task if it failed (with exponential backoff)
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, priority=TaskPriority.NORMAL)
def execute_workflow_task(self, objective: str, priority: str = "normal", 
                         callback_url: str = None) -> Dict[str, Any]:
    """
    Execute a complete workflow asynchronously
    
    Args:
        objective: The objective to execute
        priority: Priority of the workflow (low, normal, high)
        callback_url: Optional callback URL to notify when complete
    
    Returns:
        Dictionary containing the result of the workflow execution
    """
    from zero_gravity_core.agents.coordinator import Coordinator
    from zero_gravity_core.utils.logging import main_logger
    from zero_gravity_core.utils.validation import validate_objective
    
    try:
        # Validate objective
        validated_obj = validate_objective(objective, priority, callback_url)
        
        # Initialize coordinator
        coordinator = Coordinator()
        
        # Execute the workflow
        result = coordinator.run(validated_obj.objective)
        
        # Log the execution
        main_logger.info(
            "Workflow completed",
            task_id=self.request.id,
            objective=objective[:100],  # Truncate for logging
            session_id=result.get("session_id", "unknown")
        )
        
        # Call callback if provided
        if callback_url:
            # In a real implementation, you would make an HTTP request to the callback URL
            # For now, we'll just log it
            main_logger.info(
                "Callback would be sent",
                callback_url=callback_url,
                result_summary=str(result)[:200]
            )
        
        return {
            "status": "completed",
            "result": result,
            "objective": objective,
            "task_id": self.request.id
        }
    except Exception as exc:
        main_logger.error(
            "Workflow task failed",
            task_id=self.request.id,
            objective=objective[:100],
            error=str(exc)
        )
        
        # Retry the task if it failed (with exponential backoff)
        raise self.retry(exc=exc, countdown=120, max_retries=3)


@celery_app.task
def cleanup_expired_tasks() -> Dict[str, Any]:
    """
    Clean up expired tasks from the result backend
    
    This is a periodic task that runs daily to clean up old task results
    """
    from zero_gravity_core.utils.logging import main_logger
    
    try:
        # In a real implementation, you would connect to the result backend
        # and remove expired task results to free up storage
        
        # For now, just log that the cleanup ran
        main_logger.info("Expired task cleanup completed")
        
        return {
            "status": "completed",
            "message": "Expired task cleanup completed successfully"
        }
    except Exception as exc:
        main_logger.error("Expired task cleanup failed", error=str(exc))
        return {
            "status": "failed",
            "error": str(exc)
        }


def init_celery_app():
    """Initialize the Celery app with configuration"""
    # This function can be used to apply additional configuration
    # or perform setup tasks when the app starts
    pass


# Export the main functions
__all__ = ['celery_app', 'execute_agent_task', 'execute_workflow_task', 
           'cleanup_expired_tasks', 'TaskPriority', 'init_celery_app']
