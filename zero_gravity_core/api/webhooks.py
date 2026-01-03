"""
Webhook System for ZeroGravity

This module implements webhook functionality for the ZeroGravity API
to enable callbacks and event notifications.
"""
import asyncio
import aiohttp
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from urllib.parse import urlparse
import logging
from concurrent.futures import ThreadPoolExecutor


class WebhookEventType(Enum):
    """Types of events that can trigger webhooks"""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    TASK_QUEUED = "task.queued"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"


@dataclass
class WebhookConfig:
    """Configuration for a webhook"""
    id: str
    url: str
    event_types: List[WebhookEventType]
    secret: str
    active: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    retries: int = 3
    timeout: int = 30 # seconds
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.headers is None:
            self.headers = {}


@dataclass
class WebhookEvent:
    """Represents a webhook event to be sent"""
    id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    timestamp: datetime
    webhook_id: str
    attempt_count: int = 0
    last_attempt: Optional[datetime] = None
    success: bool = False
    response_status: Optional[int] = None
    response_body: Optional[str] = None


class WebhookSignature:
    """Handles webhook signature verification"""
    
    @staticmethod
    def generate_signature(payload: str, secret: str, timestamp: int) -> str:
        """Generate a signature for webhook payload"""
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str, timestamp: int, 
                        tolerance: int = 300) -> bool:  # 5 minutes tolerance
        """Verify webhook signature"""
        # Check timestamp tolerance
        current_time = int(time.time())
        if abs(current_time - timestamp) > tolerance:
            return False
        
        expected_signature = WebhookSignature.generate_signature(payload, secret, timestamp)
        return hmac.compare_digest(expected_signature, signature)


class WebhookStore:
    """Base class for webhook storage"""
    
    async def save_webhook(self, webhook: WebhookConfig) -> WebhookConfig:
        """Save a webhook configuration"""
        raise NotImplementedError
    
    async def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a webhook configuration by ID"""
        raise NotImplementedError
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook configuration"""
        raise NotImplementedError
    
    async def get_webhooks_by_event_type(self, event_type: WebhookEventType) -> List[WebhookConfig]:
        """Get all webhooks that listen for a specific event type"""
        raise NotImplementedError
    
    async def save_event(self, event: WebhookEvent) -> WebhookEvent:
        """Save a webhook event"""
        raise NotImplementedError
    
    async def get_event(self, event_id: str) -> Optional[WebhookEvent]:
        """Get a webhook event by ID"""
        raise NotImplementedError


class InMemoryWebhookStore(WebhookStore):
    """In-memory webhook store for development"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.events: Dict[str, WebhookEvent] = {}
    
    async def save_webhook(self, webhook: WebhookConfig) -> WebhookConfig:
        self.webhooks[webhook.id] = webhook
        return webhook
    
    async def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        return self.webhooks.get(webhook_id)
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    async def get_webhooks_by_event_type(self, event_type: WebhookEventType) -> List[WebhookConfig]:
        return [
            webhook for webhook in self.webhooks.values()
            if event_type in webhook.event_types and webhook.active
        ]
    
    async def save_event(self, event: WebhookEvent) -> WebhookEvent:
        self.events[event.id] = event
        return event
    
    async def get_event(self, event_id: str) -> Optional[WebhookEvent]:
        return self.events.get(event_id)


class WebhookSender:
    """Handles sending webhooks to registered URLs"""
    
    def __init__(self, store: WebhookStore = None):
        self.store = store or InMemoryWebhookStore()
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def initialize(self):
        """Initialize the webhook sender"""
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the webhook sender"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def send_webhook(self, webhook: WebhookConfig, event: WebhookEvent) -> bool:
        """Send a single webhook"""
        if not webhook.active:
            return False
        
        try:
            payload_str = json.dumps(event.payload)
            timestamp = int(event.timestamp.timestamp())
            
            # Generate signature
            signature = WebhookSignature.generate_signature(payload_str, webhook.secret, timestamp)
            
            # Prepare headers
            headers = webhook.headers.copy()
            headers.update({
                'Content-Type': 'application/json',
                'X-ZeroGravity-Event': event.event_type.value,
                'X-ZeroGravity-Timestamp': str(timestamp),
                'X-ZeroGravity-Signature': signature,
                'X-ZeroGravity-Event-ID': event.id
            })
            
            # Send the webhook
            async with self.session.post(
                webhook.url,
                json=event.payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=webhook.timeout)
            ) as response:
                response_body = await response.text()
                
                # Update event with response info
                event.response_status = response.status
                event.response_body = response_body
                event.success = 200 <= response.status < 300
                
                await self.store.save_event(event)
                
                return event.success
                
        except Exception as e:
            logging.error(f"Failed to send webhook to {webhook.url}: {str(e)}")
            
            # Update event with error info
            event.response_status = 0
            event.response_body = str(e)
            event.success = False
            
            await self.store.save_event(event)
            
            return False
    
    async def send_webhooks_for_event(self, event_type: WebhookEventType, 
                                    payload: Dict[str, Any]) -> List[bool]:
        """Send webhooks for a specific event type"""
        webhooks = await self.store.get_webhooks_by_event_type(event_type)
        
        if not webhooks:
            return []
        
        # Create event object
        event = WebhookEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        # Save the event
        await self.store.save_event(event)
        
        # Send webhooks concurrently
        results = []
        for webhook in webhooks:
            success = await self.send_webhook(webhook, event)
            results.append(success)
        
        return results
    
    async def send_webhook_with_retry(self, webhook: WebhookConfig, 
                                    event: WebhookEvent) -> bool:
        """Send a webhook with retry logic"""
        for attempt in range(webhook.retries):
            success = await self.send_webhook(webhook, event)
            if success:
                return True
            
            # Wait before retry (exponential backoff)
            if attempt < webhook.retries - 1:
                wait_time = min(2 ** attempt, 60)  # Max 60 seconds
                await asyncio.sleep(wait_time)
        
        return False


class WebhookManager:
    """Main class for managing webhooks"""
    
    def __init__(self, store: WebhookStore = None):
        self.store = store or InMemoryWebhookStore()
        self.sender = WebhookSender(store)
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}
    
    async def initialize(self):
        """Initialize the webhook manager"""
        await self.sender.initialize()
    
    async def close(self):
        """Close the webhook manager"""
        await self.sender.close()
    
    async def register_webhook(self, url: str, event_types: List[WebhookEventType], 
                             secret: str = None, active: bool = True) -> WebhookConfig:
        """Register a new webhook"""
        if secret is None:
            secret = str(uuid.uuid4())
        
        webhook = WebhookConfig(
            id=str(uuid.uuid4()),
            url=url,
            event_types=event_types,
            secret=secret,
            active=active
        )
        
        return await self.store.save_webhook(webhook)
    
    async def unregister_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook"""
        return await self.store.delete_webhook(webhook_id)
    
    async def trigger_event(self, event_type: WebhookEventType, 
                          payload: Dict[str, Any]) -> List[bool]:
        """Trigger a webhook event"""
        results = await self.sender.send_webhooks_for_event(event_type, payload)
        
        # Call any registered event handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, payload)
                    else:
                        # Run sync function in thread pool
                        await asyncio.get_event_loop().run_in_executor(
                            self.sender.executor, handler, event_type, payload
                        )
                except Exception as e:
                    logging.error(f"Event handler failed: {str(e)}")
        
        return results
    
    def register_event_handler(self, event_type: WebhookEventType, 
                             handler: Callable[[WebhookEventType, Dict[str, Any]], 
                                             Awaitable[None] | None]):
        """Register a function to handle events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def verify_webhook_signature(self, payload: str, signature: str, 
                                    secret: str, timestamp: int) -> bool:
        """Verify the signature of an incoming webhook"""
        return WebhookSignature.verify_signature(payload, signature, secret, timestamp)
    
    async def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get a webhook by ID"""
        return await self.store.get_webhook(webhook_id)
    
    async def update_webhook(self, webhook_id: str, **updates) -> Optional[WebhookConfig]:
        """Update a webhook configuration"""
        webhook = await self.store.get_webhook(webhook_id)
        if not webhook:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(webhook, key):
                setattr(webhook, key, value)
        
        webhook.updated_at = datetime.utcnow()
        
        return await self.store.save_webhook(webhook)


# Global webhook manager instance
webhook_manager = WebhookManager()


# Initialize the webhook manager
async def init_webhooks():
    """Initialize webhooks system"""
    await webhook_manager.initialize()


# Cleanup function
async def cleanup_webhooks():
    """Cleanup webhooks system"""
    await webhook_manager.close()


# Convenience functions
async def register_webhook(url: str, event_types: List[str], 
                          secret: str = None, active: bool = True) -> WebhookConfig:
    """Convenience function to register a webhook"""
    event_type_enums = [WebhookEventType[event_type.upper().replace('.', '_')] 
                       for event_type in event_types]
    return await webhook_manager.register_webhook(url, event_type_enums, secret, active)


async def trigger_event(event_type: str, payload: Dict[str, Any]) -> List[bool]:
    """Convenience function to trigger a webhook event"""
    event_type_enum = WebhookEventType[event_type.upper().replace('.', '_')]
    return await webhook_manager.trigger_event(event_type_enum, payload)


async def verify_webhook_signature(payload: str, signature: str, 
                                 secret: str, timestamp: int) -> bool:
    """Convenience function to verify webhook signature"""
    return await webhook_manager.verify_webhook_signature(payload, signature, secret, timestamp)


# Predefined event payload builders
def build_workflow_started_payload(workflow_id: str, objective: str, 
                                 user_id: str = None) -> Dict[str, Any]:
    """Build payload for workflow started event"""
    return {
        "workflow_id": workflow_id,
        "objective": objective,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }


def build_workflow_completed_payload(workflow_id: str, result: Any, 
                                   user_id: str = None) -> Dict[str, Any]:
    """Build payload for workflow completed event"""
    return {
        "workflow_id": workflow_id,
        "result": result,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }


def build_workflow_failed_payload(workflow_id: str, error: str, 
                                user_id: str = None) -> Dict[str, Any]:
    """Build payload for workflow failed event"""
    return {
        "workflow_id": workflow_id,
        "error": error,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }


def build_agent_event_payload(agent_role: str, workflow_id: str, 
                            input_data: Any, output_data: Any = None, 
                            error: str = None, user_id: str = None) -> Dict[str, Any]:
    """Build payload for agent events"""
    payload = {
        "agent_role": agent_role,
        "workflow_id": workflow_id,
        "input_data": input_data,
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if output_data is not None:
        payload["output_data"] = output_data
    
    if error is not None:
        payload["error"] = error
    
    return payload
