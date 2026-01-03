"""
Authentication and Authorization System for ZeroGravity

This module implements a comprehensive authentication and authorization system
for the ZeroGravity platform, including user management, role-based access control,
and secure token management.
"""
import asyncio
import hashlib
import secrets
import jwt
import bcrypt
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import logging
from contextlib import contextmanager
import re


class UserRole(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE_ACCOUNT = "service_account"


class Permission(Enum):
    """Permissions available in the system"""
    READ_OBJECTIVES = "read:objectives"
    WRITE_OBJECTIVES = "write:objectives"
    READ_JOBS = "read:jobs"
    WRITE_JOBS = "write:jobs"
    MANAGE_USERS = "manage:users"
    READ_WORKFLOWS = "read:workflows"
    WRITE_WORKFLOWS = "write:workflows"
    ACCESS_ADMIN_PANEL = "access:admin_panel"
    READ_SYSTEM_METRICS = "read:system_metrics"
    MANAGE_PLUGINS = "manage:plugins"


@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    password_hash: str
    salt: str
    api_key: str
    refresh_token: Optional[str] = None
    password_reset_token: Optional[str] = None
    password_reset_expires: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_active": self.is_active
        }


@dataclass
class TokenPayload:
    """Payload for JWT tokens"""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    exp: datetime
    iat: datetime
    token_type: str  # 'access' or 'refresh'


class PasswordHasher:
    """Handles password hashing and verification"""
    
    def __init__(self, rounds: int = 12):
        self.rounds = rounds
    
    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash a password and return hash and salt"""
        salt = bcrypt.gensalt(rounds=self.rounds)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


class TokenManager:
    """Manages JWT token creation and validation"""
    
    def __init__(self, secret_key: str, access_token_ttl: int = 3600, 
                 refresh_token_ttl: int = 86400):
        self.secret_key = secret_key
        self.access_token_ttl = access_token_ttl  # 1 hour default
        self.refresh_token_ttl = refresh_token_ttl  # 24 hours default
        self.logger = logging.getLogger("TokenManager")
    
    def create_access_token(self, user: User) -> str:
        """Create an access token for a user"""
        now = datetime.utcnow()
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": (now + timedelta(seconds=self.access_token_ttl)).timestamp(),
            "iat": now.timestamp(),
            "token_type": "access"
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def create_refresh_token(self, user: User) -> str:
        """Create a refresh token for a user"""
        now = datetime.utcnow()
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "exp": (now + timedelta(seconds=self.refresh_token_ttl)).timestamp(),
            "iat": now.timestamp(),
            "token_type": "refresh"
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
    
    def is_token_expired(self, token: str) -> bool:
        """Check if a token is expired"""
        try:
            payload = self.decode_token(token)
            exp = payload.get("exp")
            if exp:
                return datetime.utcnow().timestamp() > exp
            return True
        except ValueError:
            return True


class UserDatabase:
    """Database interface for user management"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.password_hasher = PasswordHasher()
        self.lock = asyncio.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize the user database"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL,
                    refresh_token TEXT,
                    password_reset_token TEXT,
                    password_reset_expires TEXT
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_api_key ON users(api_key)')
            
            # Create default admin user if none exists
            cursor = conn.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            if count == 0:
                self._create_default_admin_user(conn)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper handling"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _create_default_admin_user(self, conn):
        """Create a default admin user"""
        user_id = "admin_" + secrets.token_hex(8)
        username = "admin"
        email = "admin@zerogravity.ai"
        role = UserRole.ADMIN.value
        permissions = [p.value for p in Permission]
        created_at = datetime.utcnow().isoformat()
        password_hash, salt = self.password_hasher.hash_password("admin123")
        api_key = "sk-" + secrets.token_urlsafe(32)
        
        conn.execute('''
            INSERT INTO users (
                user_id, username, email, role, permissions, created_at, 
                is_active, password_hash, salt, api_key
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, username, email, role, 
            ",".join(permissions), created_at, 
            True, password_hash, salt, api_key
        ))
    
    def create_user(self, username: str, email: str, password: str, 
                    role: UserRole = UserRole.USER) -> Optional[User]:
        """Create a new user"""
        password_hash, salt = self.password_hasher.hash_password(password)
        user_id = "user_" + secrets.token_hex(8)
        created_at = datetime.utcnow()
        api_key = "sk-" + secrets.token_urlsafe(32)
        
        # Set default permissions based on role
        if role == UserRole.ADMIN:
            permissions = [p for p in Permission]
        elif role == UserRole.USER:
            permissions = [
                Permission.READ_OBJECTIVES,
                Permission.WRITE_OBJECTIVES,
                Permission.READ_JOBS,
                Permission.READ_WORKFLOWS
            ]
        elif role == UserRole.GUEST:
            permissions = [Permission.READ_OBJECTIVES]
        else:  # SERVICE_ACCOUNT
            permissions = [
                Permission.READ_OBJECTIVES,
                Permission.WRITE_OBJECTIVES,
                Permission.READ_JOBS
            ]
        
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO users (
                        user_id, username, email, role, permissions, created_at, 
                        is_active, password_hash, salt, api_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, username, email, role.value, 
                    ",".join([p.value for p in permissions]), 
                    created_at.isoformat(), True, password_hash, salt, api_key
                ))
            
            return User(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                created_at=created_at,
                last_login=None,
                is_active=True,
                password_hash=password_hash,
                salt=salt,
                api_key=api_key
            )
        except sqlite3.IntegrityError:
            return None  # Username or email already exists
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM users WHERE email = ?
            ''', (email,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get a user by API key"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM users WHERE api_key = ?
            ''', (api_key,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        user = self.get_user_by_username(username)
        if user and self.password_hasher.verify_password(password, user.password_hash):
            # Update last login
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE users SET last_login = ? WHERE user_id = ?
                ''', (datetime.utcnow().isoformat(), user.user_id))
            
            # Refresh the user object
            return self.get_user_by_username(username)
        return None
    
    def update_user_permissions(self, user_id: str, permissions: List[Permission]) -> bool:
        """Update user permissions"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE users SET permissions = ? WHERE user_id = ?
                ''', (",".join([p.value for p in permissions]), user_id))
            return True
        except Exception:
            return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE users SET is_active = 0 WHERE user_id = ?
                ''', (user_id,))
            return True
        except Exception:
            return False
    
    def activate_user(self, user_id: str) -> bool:
        """Activate a user account"""
        try:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE users SET is_active = 1 WHERE user_id = ?
                ''', (user_id,))
            return True
        except Exception:
            return False
    
    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert a database row to a User object"""
        permissions = [Permission(p) for p in row['permissions'].split(',') if p]
        
        return User(
            user_id=row['user_id'],
            username=row['username'],
            email=row['email'],
            role=UserRole(row['role']),
            permissions=permissions,
            created_at=datetime.fromisoformat(row['created_at']),
            last_login=datetime.fromisoformat(row['last_login']) if row['last_login'] else None,
            is_active=bool(row['is_active']),
            password_hash=row['password_hash'],
            salt=row['salt'],
            api_key=row['api_key'],
            refresh_token=row['refresh_token']
        )


class AuthorizationManager:
    """Manages authorization and permission checks"""
    
    def __init__(self, user_db: UserDatabase):
        self.user_db = user_db
        self.logger = logging.getLogger("AuthorizationManager")
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a specific permission"""
        return permission in user.permissions
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """Check if a user has any of the specified permissions"""
        user_perms = set(user.permissions)
        req_perms = set(permissions)
        return bool(user_perms.intersection(req_perms))
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if a user has all of the specified permissions"""
        return all(perm in user.permissions for perm in permissions)
    
    def check_resource_access(self, user: User, resource_type: str, 
                            resource_id: str, action: str) -> bool:
        """Check if a user can access a specific resource with an action"""
        # Map resource types to permissions
        resource_permission_map = {
            "objective": {
                "read": Permission.READ_OBJECTIVES,
                "write": Permission.WRITE_OBJECTIVES
            },
            "job": {
                "read": Permission.READ_JOBS,
                "write": Permission.WRITE_JOBS
            },
            "workflow": {
                "read": Permission.READ_WORKFLOWS,
                "write": Permission.WRITE_WORKFLOWS
            },
            "user": {
                "read": Permission.MANAGE_USERS,
                "write": Permission.MANAGE_USERS
            },
            "system": {
                "read": Permission.READ_SYSTEM_METRICS,
                "write": Permission.ACCESS_ADMIN_PANEL
            }
        }
        
        if resource_type not in resource_permission_map:
            return False
        
        if action not in resource_permission_map[resource_type]:
            return False
        
        required_permission = resource_permission_map[resource_type][action]
        return self.has_permission(user, required_permission)
    
    def filter_accessible_resources(self, user: User, resources: List[Dict[str, Any]], 
                                  resource_type: str, action: str) -> List[Dict[str, Any]]:
        """Filter resources based on user permissions"""
        if not self.check_resource_access(user, resource_type, "*", action):
            return []  # User doesn't have permission to access any of this resource type
        
        # For now, return all resources (in a real implementation, you'd check ownership, etc.)
        return resources


class AuthSystem:
    """Main authentication and authorization system"""
    
    def __init__(self, secret_key: str, db_path: str = "users.db"):
        self.secret_key = secret_key
        self.user_db = UserDatabase(db_path)
        self.token_manager = TokenManager(secret_key)
        self.authz_manager = AuthorizationManager(self.user_db)
        self.logger = logging.getLogger("AuthSystem")
    
    def register_user(self, username: str, email: str, password: str, 
                     role: UserRole = UserRole.USER) -> Optional[User]:
        """Register a new user"""
        # Validate input
        if not self._is_valid_username(username):
            raise ValueError("Invalid username format")
        
        if not self._is_valid_email(email):
            raise ValueError("Invalid email format")
        
        if not self._is_valid_password(password):
            raise ValueError("Password does not meet requirements")
        
        # Create user
        user = self.user_db.create_user(username, email, password, role)
        if user:
            self.logger.info(f"New user registered: {username} ({user.user_id})")
            return user
        else:
            raise ValueError("Username or email already exists")
    
    def authenticate(self, username: str, password: str) -> Optional[Tuple[User, str, str]]:
        """Authenticate user and return user object with access and refresh tokens"""
        user = self.user_db.authenticate_user(username, password)
        if user:
            access_token = self.token_manager.create_access_token(user)
            refresh_token = self.token_manager.create_refresh_token(user)
            
            # Update refresh token in database
            with self.user_db._get_connection() as conn:
                conn.execute('''
                    UPDATE users SET refresh_token = ? WHERE user_id = ?
                ''', (refresh_token, user.user_id))
            
            self.logger.info(f"User authenticated: {username} ({user.user_id})")
            return user, access_token, refresh_token
        
        return None
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify a JWT token and return the user"""
        try:
            payload = self.token_manager.decode_token(token)
            user_id = payload.get("sub")
            
            # Get user from database
            # Since we don't have direct access to user_db.get_user_by_id method,
            # we'll get user by username to verify they still exist
            username = payload.get("username")
            user = self.user_db.get_user_by_username(username)
            
            if user and user.user_id == user_id and user.is_active:
                return user
        except ValueError:
            pass  # Invalid token
        
        return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Refresh an access token using a refresh token"""
        try:
            payload = self.token_manager.decode_token(refresh_token)
            token_type = payload.get("token_type")
            
            if token_type != "refresh":
                return None
            
            user_id = payload.get("sub")
            username = payload.get("username")
            
            # Verify user still exists and is active
            user = self.user_db.get_user_by_username(username)
            if user and user.user_id == user_id and user.is_active:
                # Check if refresh token matches stored one
                if user.refresh_token == refresh_token:
                    # Create new tokens
                    new_access_token = self.token_manager.create_access_token(user)
                    new_refresh_token = self.token_manager.create_refresh_token(user)
                    
                    # Update refresh token in database
                    with self.user_db._get_connection() as conn:
                        conn.execute('''
                            UPDATE users SET refresh_token = ? WHERE user_id = ?
                        ''', (new_refresh_token, user.user_id))
                    
                    return new_access_token, new_refresh_token
        except ValueError:
            pass  # Invalid token
        
        return None
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a specific permission"""
        return self.authz_manager.has_permission(user, permission)
    
    def check_resource_access(self, user: User, resource_type: str, 
                            resource_id: str, action: str) -> bool:
        """Check if a user can access a specific resource with an action"""
        return self.authz_manager.check_resource_access(user, resource_type, resource_id, action)
    
    def _is_valid_username(self, username: str) -> bool:
        """Validate username format"""
        return bool(re.match(r'^[a-zA-Z0-9_-]{3,30}$', username))
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _is_valid_password(self, password: str) -> bool:
        """Validate password strength"""
        # At least 8 characters, one uppercase, one lowercase, one digit
        if len(password) < 8:
            return False
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        return True
    
    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get a user by their API key"""
        return self.user_db.get_user_by_api_key(api_key)
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change a user's password"""
        # Get the user
        user = self.user_db.get_user_by_username(self.user_db.get_user_by_api_key(api_key).username)
        
        # In a real implementation, we'd need to get the user differently
        # For now, let's assume we have a method to get user by ID
        with self.user_db._get_connection() as conn:
            cursor = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            stored_hash = row['password_hash']
            if not self.user_db.password_hasher.verify_password(current_password, stored_hash):
                return False
            
            # Validate new password
            if not self._is_valid_password(new_password):
                return False
            
            # Hash new password
            new_hash, new_salt = self.user_db.password_hasher.hash_password(new_password)
            
            # Update password
            conn.execute('''
                UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?
            ''', (new_hash, new_salt, user_id))
            
            return True


# Global auth system instance
auth_system: Optional[AuthSystem] = None


def init_auth_system(secret_key: str, db_path: str = "users.db") -> AuthSystem:
    """Initialize the authentication system"""
    global auth_system
    auth_system = AuthSystem(secret_key, db_path)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    return auth_system


def get_auth_system() -> Optional[AuthSystem]:
    """Get the global auth system instance"""
    return auth_system


def authenticate_user(username: str, password: str) -> Optional[Tuple[User, str, str]]:
    """Authenticate a user"""
    if not auth_system:
        return None
    return auth_system.authenticate(username, password)


def verify_token(token: str) -> Optional[User]:
    """Verify a JWT token"""
    if not auth_system:
        return None
    return auth_system.verify_token(token)


def check_permission(user: User, permission: Permission) -> bool:
    """Check if a user has a specific permission"""
    if not auth_system:
        return False
    return auth_system.check_permission(user, permission)


def register_new_user(username: str, email: str, password: str, 
                     role: UserRole = UserRole.USER) -> Optional[User]:
    """Register a new user"""
    if not auth_system:
        return None
    return auth_system.register_user(username, email, password, role)


# Example usage and testing
async def test_auth_system():
    """Test the authentication system"""
    print("Testing ZeroGravity Authentication System...")
    
    # Initialize auth system
    auth_sys = init_auth_system("my_super_secret_key_123")
    
    print("Auth system initialized")
    
    # Test user registration
    try:
        user = register_new_user("testuser", "test@example.com", "SecurePass123", UserRole.USER)
        print(f"User registered: {user.username if user else 'FAILED'}")
    except Exception as e:
        print(f"Registration failed: {e}")
    
    # Test authentication
    auth_result = authenticate_user("testuser", "SecurePass123")
    if auth_result:
        user, access_token, refresh_token = auth_result
        print(f"Authentication successful for {user.username}")
        print(f"Access token: {access_token[:20]}...")
        print(f"Refresh token: {refresh_token[:20]}...")
    else:
        print("Authentication failed")
    
    # Test token verification
    verified_user = verify_token(access_token) if auth_result else None
    if verified_user:
        print(f"Token verification successful for {verified_user.username}")
    else:
        print("Token verification failed")
    
    # Test permissions
    if verified_user:
        has_perm = check_permission(verified_user, Permission.READ_OBJECTIVES)
        print(f"User has READ_OBJECTIVES permission: {has_perm}")
    
    # Test API key lookup
    if auth_result:
        api_key_user = auth_sys.get_user_by_api_key(auth_result[0].api_key)
        print(f"API key lookup successful: {api_key_user.username if api_key_user else 'FAILED'}")


if __name__ == "__main__":
    # For testing purposes
    print("Starting ZeroGravity Authentication System example...")
    # asyncio.run(test_auth_system())
