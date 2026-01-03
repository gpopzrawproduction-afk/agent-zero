"""
Audit Trails System for ZeroGravity

This module implements comprehensive audit trails for compliance,
tracking all important actions, decisions, and data access in the
ZeroGravity platform.
"""
import json
import hashlib
import hmac
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
from contextlib import contextmanager
import logging


class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    OBJECTIVE_SUBMITTED = "objective_submitted"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    AGENT_EXECUTED = "agent_executed"
    LLM_CALL_MADE = "llm_call_made"
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    CONFIG_CHANGED = "config_changed"
    API_CALL = "api_call"
    FILE_UPLOADED = "file_uploaded"
    FILE_DOWNLOADED = "file_downloaded"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    SYSTEM_ERROR = "system_error"
    SECURITY_EVENT = "security_event"


@dataclass
class AuditEvent:
    """Represents an audit event"""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    action: str
    resource: str
    metadata: Dict[str, Any]
    signature: str  # Digital signature for integrity
    source: str = "ZeroGravity"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "action": self.action,
            "resource": self.resource,
            "metadata": self.metadata,
            "signature": self.signature,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create audit event from dictionary"""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=AuditEventType(data["event_type"]),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            ip_address=data.get("ip_address"),
            action=data["action"],
            resource=data["resource"],
            metadata=data["metadata"],
            signature=data["signature"],
            source=data.get("source", "ZeroGravity")
        )


class AuditTrailIntegrity:
    """Handles integrity verification for audit trails"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
    
    def generate_signature(self, event: AuditEvent) -> str:
        """Generate digital signature for an audit event"""
        # Create a string representation of the event (excluding signature)
        event_data = {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "ip_address": event.ip_address,
            "action": event.action,
            "resource": event.resource,
            "metadata": event.metadata,
            "source": event.source
        }
        
        message = json.dumps(event_data, sort_keys=True, separators=(',', ':')).encode()
        
        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key,
            message,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, event: AuditEvent) -> bool:
        """Verify the digital signature of an audit event"""
        # Generate expected signature
        expected_signature = self.generate_signature(event)
        
        # Compare signatures securely
        return hmac.compare_digest(expected_signature, event.signature)


class AuditDatabase:
    """Database interface for audit trails"""
    
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize the audit database"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    metadata TEXT,
                    signature TEXT NOT NULL,
                    source TEXT NOT NULL
                )
            ''')
            
            # Create indexes for faster queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_resource ON audit_events(resource)')
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper handling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def insert_event(self, event: AuditEvent) -> bool:
        """Insert an audit event into the database"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO audit_events (
                        id, timestamp, event_type, user_id, session_id, 
                        ip_address, action, resource, metadata, signature, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.user_id,
                    event.session_id,
                    event.ip_address,
                    event.action,
                    event.resource,
                    json.dumps(event.metadata),
                    event.signature,
                    event.source
                ))
            return True
        except Exception as e:
            logging.error(f"Failed to insert audit event: {e}")
            return False
    
    def get_events(self, limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """Retrieve audit events with pagination"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            events = []
            for row in cursor.fetchall():
                event_data = dict(row)
                event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                event_data["event_type"] = AuditEventType(event_data["event_type"])
                event_data["metadata"] = json.loads(event_data["metadata"])
                
                # Create event but without signature verification for now
                event = AuditEvent(
                    id=event_data["id"],
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    user_id=event_data["user_id"],
                    session_id=event_data["session_id"],
                    ip_address=event_data["ip_address"],
                    action=event_data["action"],
                    resource=event_data["resource"],
                    metadata=event_data["metadata"],
                    signature=event_data["signature"],
                    source=event_data["source"]
                )
                events.append(event)
            
            return events
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[AuditEvent]:
        """Retrieve audit events for a specific user"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                WHERE user_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (user_id, limit))
            
            events = []
            for row in cursor.fetchall():
                event_data = dict(row)
                event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                event_data["event_type"] = AuditEventType(event_data["event_type"])
                event_data["metadata"] = json.loads(event_data["metadata"])
                
                event = AuditEvent(
                    id=event_data["id"],
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    user_id=event_data["user_id"],
                    session_id=event_data["session_id"],
                    ip_address=event_data["ip_address"],
                    action=event_data["action"],
                    resource=event_data["resource"],
                    metadata=event_data["metadata"],
                    signature=event_data["signature"],
                    source=event_data["source"]
                )
                events.append(event)
            
            return events
    
    def get_events_by_type(self, event_type: AuditEventType, limit: int = 100) -> List[AuditEvent]:
        """Retrieve audit events of a specific type"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                WHERE event_type = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (event_type.value, limit))
            
            events = []
            for row in cursor.fetchall():
                event_data = dict(row)
                event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                event_data["event_type"] = AuditEventType(event_data["event_type"])
                event_data["metadata"] = json.loads(event_data["metadata"])
                
                event = AuditEvent(
                    id=event_data["id"],
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    user_id=event_data["user_id"],
                    session_id=event_data["session_id"],
                    ip_address=event_data["ip_address"],
                    action=event_data["action"],
                    resource=event_data["resource"],
                    metadata=event_data["metadata"],
                    signature=event_data["signature"],
                    source=event_data["source"]
                )
                events.append(event)
            
            return events
    
    def search_events(self, query: str, limit: int = 100) -> List[AuditEvent]:
        """Search audit events by text query"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM audit_events 
                WHERE action LIKE ? OR resource LIKE ? OR user_id LIKE ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            events = []
            for row in cursor.fetchall():
                event_data = dict(row)
                event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                event_data["event_type"] = AuditEventType(event_data["event_type"])
                event_data["metadata"] = json.loads(event_data["metadata"])
                
                event = AuditEvent(
                    id=event_data["id"],
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    user_id=event_data["user_id"],
                    session_id=event_data["session_id"],
                    ip_address=event_data["ip_address"],
                    action=event_data["action"],
                    resource=event_data["resource"],
                    metadata=event_data["metadata"],
                    signature=event_data["signature"],
                    source=event_data["source"]
                )
                events.append(event)
            
            return events


class AuditTrailManager:
    """Main manager for audit trails"""
    
    def __init__(self, secret_key: str, db_path: str = "audit.db"):
        self.integrity = AuditTrailIntegrity(secret_key)
        self.database = AuditDatabase(db_path)
        self.logger = logging.getLogger(__name__)
    
    def log_event(self, 
                  event_type: AuditEventType, 
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  action: str = "",
                  resource: str = "",
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Log an audit event"""
        try:
            # Create event ID using timestamp and hash
            timestamp = datetime.utcnow()
            event_id_data = f"{timestamp.isoformat()}{event_type.value}{user_id or ''}{action}{resource}"
            event_id = hashlib.sha256(event_id_data.encode()).hexdigest()
            
            # Create event object
            event = AuditEvent(
                id=event_id,
                timestamp=timestamp,
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                action=action,
                resource=resource,
                metadata=metadata or {},
                signature="",  # Will be set after creating the signature
                source="ZeroGravity"
            )
            
            # Generate signature
            event.signature = self.integrity.generate_signature(event)
            
            # Insert into database
            success = self.database.insert_event(event)
            
            if success:
                self.logger.info(f"Audit event logged: {event_type.value} for user {user_id}")
            else:
                self.logger.error(f"Failed to log audit event: {event_type.value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error logging audit event: {e}")
            return False
    
    def verify_event_integrity(self, event: AuditEvent) -> bool:
        """Verify the integrity of a single audit event"""
        return self.integrity.verify_signature(event)
    
    def verify_trail_integrity(self, events: List[AuditEvent]) -> Dict[str, bool]:
        """Verify the integrity of multiple audit events"""
        results = {}
        for event in events:
            results[event.id] = self.verify_event_integrity(event)
        return results
    
    def get_events(self, limit: int = 100, offset: int = 0) -> List[AuditEvent]:
        """Get audit events with pagination"""
        return self.database.get_events(limit, offset)
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[AuditEvent]:
        """Get audit events for a specific user"""
        return self.database.get_events_by_user(user_id, limit)
    
    def get_events_by_type(self, event_type: AuditEventType, limit: int = 100) -> List[AuditEvent]:
        """Get audit events of a specific type"""
        return self.database.get_events_by_type(event_type, limit)
    
    def search_events(self, query: str, limit: int = 100) -> List[AuditEvent]:
        """Search audit events by text query"""
        return self.database.search_events(query, limit)
    
    def export_events(self, filepath: str, event_type: Optional[AuditEventType] = None) -> bool:
        """Export audit events to a file"""
        try:
            if event_type:
                events = self.database.get_events_by_type(event_type, limit=10000)  # Large limit for export
            else:
                events = self.database.get_events(limit=10000) # Large limit for export
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + '\n')
            
            self.logger.info(f"Exported {len(events)} audit events to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting audit events: {e}")
            return False
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report"""
        total_events = len(self.database.get_events(limit=1))  # Just get a count by attempting to get 1
        
        # Get counts by event type
        event_counts = {}
        for event_type in AuditEventType:
            count = len(self.database.get_events_by_type(event_type, limit=1))  # Just get a count
            event_counts[event_type.value] = count
        
        # Get recent events
        recent_events = self.database.get_events(limit=10)
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_events": total_events,
            "event_counts": event_counts,
            "recent_events": [event.to_dict() for event in recent_events]
        }
        
        return report


class AuditTrailMiddleware:
    """Middleware for automatically logging important events"""
    
    def __init__(self, audit_manager: AuditTrailManager):
        self.audit_manager = audit_manager
    
    def log_request(self, 
                    user_id: Optional[str], 
                    ip_address: str, 
                    method: str, 
                    endpoint: str,
                    status_code: int,
                    request_data: Optional[Dict[str, Any]] = None) -> bool:
        """Log an API request"""
        metadata = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "request_size": len(json.dumps(request_data)) if request_data else 0
        }
        
        if request_data:
            # Don't log sensitive data, just record that it existed
            if "password" in request_data or "token" in request_data or "key" in request_data:
                metadata["has_sensitive_data"] = True
                # Remove sensitive data from metadata
                safe_data = {k: v for k, v in request_data.items() 
                           if k not in ["password", "token", "key", "secret"]}
                metadata["request_data"] = safe_data
            else:
                metadata["request_data"] = request_data
        
        return self.audit_manager.log_event(
            event_type=AuditEventType.API_CALL,
            user_id=user_id,
            ip_address=ip_address,
            action=f"{method} {endpoint}",
            resource=endpoint,
            metadata=metadata
        )
    
    def log_user_login(self, user_id: str, ip_address: str) -> bool:
        """Log user login event"""
        return self.audit_manager.log_event(
            event_type=AuditEventType.USER_LOGIN,
            user_id=user_id,
            ip_address=ip_address,
            action="User login",
            resource=f"user:{user_id}",
            metadata={"ip_address": ip_address}
        )
    
    def log_user_logout(self, user_id: str, session_id: str) -> bool:
        """Log user logout event"""
        return self.audit_manager.log_event(
            event_type=AuditEventType.USER_LOGOUT,
            user_id=user_id,
            session_id=session_id,
            action="User logout",
            resource=f"user:{user_id}",
            metadata={"session_id": session_id}
        )
    
    def log_objective_submission(self, user_id: str, objective: str, workflow_id: str) -> bool:
        """Log objective submission"""
        return self.audit_manager.log_event(
            event_type=AuditEventType.OBJECTIVE_SUBMITTED,
            user_id=user_id,
            action="Objective submitted",
            resource=f"workflow:{workflow_id}",
            metadata={
                "objective_summary": objective[:100],  # Truncate for privacy
                "workflow_id": workflow_id
            }
        )
    
    def log_agent_execution(self, 
                           user_id: str, 
                           agent_role: str, 
                           workflow_id: str,
                           input_data: Any,
                           output_data: Any = None) -> bool:
        """Log agent execution"""
        return self.audit_manager.log_event(
            event_type=AuditEventType.AGENT_EXECUTED,
            user_id=user_id,
            action=f"Agent {agent_role} executed",
            resource=f"agent:{agent_role}",
            metadata={
                "workflow_id": workflow_id,
                "agent_role": agent_role,
                "input_summary": str(input_data)[:200] if input_data else "",
                "output_summary": str(output_data)[:200] if output_data else ""
            }
        )
    
    def log_security_event(self, 
                          user_id: Optional[str], 
                          event_description: str,
                          severity: str = "medium",
                          ip_address: Optional[str] = None) -> bool:
        """Log security-related events"""
        return self.audit_manager.log_event(
            event_type=AuditEventType.SECURITY_EVENT,
            user_id=user_id,
            ip_address=ip_address,
            action=event_description,
            resource="security",
            metadata={
                "severity": severity,
                "ip_address": ip_address
            }
        )


# Global audit manager instance
audit_manager: Optional[AuditTrailManager] = None
audit_middleware: Optional[AuditTrailMiddleware] = None


def init_audit_system(secret_key: str, db_path: str = "audit.db") -> AuditTrailManager:
    """Initialize the audit system"""
    global audit_manager, audit_middleware
    
    audit_manager = AuditTrailManager(secret_key, db_path)
    audit_middleware = AuditTrailMiddleware(audit_manager)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    return audit_manager


def get_audit_manager() -> Optional[AuditTrailManager]:
    """Get the global audit manager instance"""
    return audit_manager


def get_audit_middleware() -> Optional[AuditTrailMiddleware]:
    """Get the global audit middleware instance"""
    return audit_middleware


def log_audit_event(event_type: AuditEventType, 
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   action: str = "",
                   resource: str = "",
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to log an audit event"""
    if audit_manager is None:
        # Initialize if not already done
        init_audit_system("default_secret_key")
    
    return audit_manager.log_event(
        event_type=event_type,
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        action=action,
        resource=resource,
        metadata=metadata
    )


def search_audit_events(query: str, limit: int = 100) -> List[AuditEvent]:
    """Convenience function to search audit events"""
    if audit_manager is None:
        return []
    
    return audit_manager.search_events(query, limit)


def generate_compliance_report() -> Dict[str, Any]:
    """Convenience function to generate a compliance report"""
    if audit_manager is None:
        return {}
    
    return audit_manager.generate_compliance_report()


def export_audit_events(filepath: str, event_type: Optional[AuditEventType] = None) -> bool:
    """Convenience function to export audit events"""
    if audit_manager is None:
        return False
    
    return audit_manager.export_events(filepath, event_type)


# Example usage and initialization
if __name__ == "__main__":
    # Initialize audit system
    audit_mgr = init_audit_system("my_secret_key_123")
    
    print("ZeroGravity Audit Trail System initialized")
    
    # Log some example events
    success1 = log_audit_event(
        event_type=AuditEventType.USER_LOGIN,
        user_id="user123",
        ip_address="192.168.1.1",
        action="User logged in",
        resource="auth",
        metadata={"login_method": "password"}
    )
    
    success2 = log_audit_event(
        event_type=AuditEventType.OBJECTIVE_SUBMITTED,
        user_id="user123",
        action="Submitted objective for processing",
        resource="workflow:abc123",
        metadata={
            "objective": "Create a web application",
            "priority": "high"
        }
    )
    
    print(f"Events logged successfully: {success1 and success2}")
    
    # Get recent events
    recent_events = audit_mgr.get_events(limit=5)
    print(f"\nRecent audit events: {len(recent_events)}")
    for event in recent_events:
        print(f"  {event.event_type.value} - {event.action} - {event.timestamp}")
    
    # Generate compliance report
    report = generate_compliance_report()
    print(f"\nCompliance report generated at: {report['generated_at']}")
    print(f"Total events: {report['total_events']}")
    
    # Export events to file
    export_success = export_audit_events("audit_export.json")
    print(f"\nAudit events exported: {export_success}")
