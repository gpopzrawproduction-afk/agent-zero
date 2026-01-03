"""
Deployment Configuration for ZeroGravity

This module contains comprehensive deployment configurations
for different environments (development, staging, production)
for the ZeroGravity platform.
"""
import os
import json
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ServiceType(Enum):
    """Types of services in the platform"""
    API_GATEWAY = "api_gateway"
    AGENT_COORDINATOR = "agent_coordinator"
    LLM_PROVIDER = "llm_provider"
    TASK_QUEUE = "task_queue"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"
    AUTH_SERVICE = "auth_service"


@dataclass
class ServiceConfig:
    """Configuration for a specific service"""
    name: str
    type: ServiceType
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "500m", "memory": "512Mi"}
    })
    env_vars: Dict[str, str] = field(default_factory=dict)
    ports: List[Dict[str, int]] = field(default_factory=list)
    health_check: Dict[str, Any] = field(default_factory=lambda: {
        "path": "/health",
        "port": 8000,
        "interval": 30,
        "timeout": 10
    })
    readiness_probe: Optional[Dict[str, Any]] = None
    liveness_probe: Optional[Dict[str, Any]] = None


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "zerogravity"
    user: str = "zerogravity_user"
    password: str = "zerogravity_pass"
    ssl_mode: str = "prefer"
    connection_pool_size: int = 20
    max_overflow: int = 30
    echo_sql: bool = False
    
    def get_database_url(self) -> str:
        """Get the database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration for caching and queues"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 20
    socket_keepalive: bool = True
    socket_connect_timeout: int = 5
    socket_timeout: int = 10


@dataclass
class LLMProviderConfig:
    """Configuration for LLM providers"""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    default_model: str = "gpt-4"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60  # seconds


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    grafana_enabled: bool = True
    grafana_port: int = 3000
    jaeger_enabled: bool = True
    jaeger_port: int = 16686
    logging_level: str = "INFO"
    log_format: str = "json"  # json or text
    metrics_retention_days: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = "your-jwt-secret-key-here"
    api_key_prefix: str = "zk_"
    password_hash_rounds: int = 12
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds


@dataclass
class DeploymentConfig:
    """Main deployment configuration"""
    environment: Environment
    project_name: str = "zerogravity"
    version: str = "1.0.0"
    domain: str = "zerogravity.ai"
    api_version: str = "v1"
    
    # Service configurations
    api_gateway: ServiceConfig = None
    agent_coordinator: ServiceConfig = None
    task_queue: ServiceConfig = None
    monitoring: ServiceConfig = None
    auth_service: ServiceConfig = None
    
    # Infrastructure configs
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm_providers: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Deployment settings
    replica_count: int = 1
    image_pull_policy: str = "IfNotPresent"
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)
    affinity: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    cpu_request: str = "100m"
    memory_request: str = "256Mi"
    
    def __post_init__(self):
        """Initialize default service configurations"""
        if self.api_gateway is None:
            self.api_gateway = ServiceConfig(
                name="api-gateway",
                type=ServiceType.API_GATEWAY,
                replicas=self.replica_count,
                ports=[{"name": "http", "port": 8000}],
                env_vars={
                    "ENVIRONMENT": self.environment.value,
                    "PROJECT_NAME": self.project_name,
                    "VERSION": self.version
                }
            )
        
        if self.agent_coordinator is None:
            self.agent_coordinator = ServiceConfig(
                name="agent-coordinator",
                type=ServiceType.AGENT_COORDINATOR,
                replicas=self.replica_count,
                ports=[{"name": "http", "port": 8001}],
                env_vars={
                    "ENVIRONMENT": self.environment.value
                }
            )
        
        if self.task_queue is None:
            self.task_queue = ServiceConfig(
                name="task-queue",
                type=ServiceType.TASK_QUEUE,
                replicas=1,  # Usually just one for coordination
                ports=[{"name": "http", "port": 8002}],
                env_vars={
                    "ENVIRONMENT": self.environment.value
                }
            )
        
        if self.monitoring is None:
            self.monitoring = ServiceConfig(
                name="monitoring",
                type=ServiceType.MONITORING,
                replicas=1,
                ports=[{"name": "http", "port": 9090}],
                env_vars={
                    "ENVIRONMENT": self.environment.value
                }
            )
        
        if self.auth_service is None:
            self.auth_service = ServiceConfig(
                name="auth-service",
                type=ServiceType.AUTH_SERVICE,
                replicas=self.replica_count,
                ports=[{"name": "http", "port": 8003}],
                env_vars={
                    "ENVIRONMENT": self.environment.value
                }
            )
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get all environment variables for the deployment"""
        env_vars = {
            # General
            "ENVIRONMENT": self.environment.value,
            "PROJECT_NAME": self.project_name,
            "VERSION": self.version,
            "DOMAIN": self.domain,
            
            # Database
            "DATABASE_URL": self.database.get_database_url(),
            "DB_HOST": self.database.host,
            "DB_PORT": str(self.database.port),
            "DB_NAME": self.database.name,
            "DB_USER": self.database.user,
            "DB_PASSWORD": self.database.password,
            
            # Redis
            "REDIS_URL": f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}",
            "REDIS_HOST": self.redis.host,
            "REDIS_PORT": str(self.redis.port),
            "REDIS_DB": str(self.redis.db),
            
            # LLM Providers
            "OPENAI_API_KEY": self.llm_providers.openai_api_key or "",
            "ANTHROPIC_API_KEY": self.llm_providers.anthropic_api_key or "",
            "OPENAI_BASE_URL": self.llm_providers.openai_base_url,
            "ANTHROPIC_BASE_URL": self.llm_providers.anthropic_base_url,
            "DEFAULT_MODEL": self.llm_providers.default_model,
            
            # Security
            "JWT_SECRET": self.security.jwt_secret,
            "API_KEY_PREFIX": self.security.api_key_prefix,
            
            # Monitoring
            "LOGGING_LEVEL": self.security.logging_level,
        }
        
        # Add any custom environment variables from services
        for service in [self.api_gateway, self.agent_coordinator, self.task_queue, 
                       self.monitoring, self.auth_service]:
            env_vars.update(service.env_vars)
        
        return env_vars
    
    def validate(self) -> List[str]:
        """Validate the configuration and return a list of errors"""
        errors = []
        
        # Validate environment-specific requirements
        if self.environment == Environment.PRODUCTION:
            if not self.security.jwt_secret or self.security.jwt_secret == "your-jwt-secret-key-here":
                errors.append("JWT secret must be set for production environment")
            
            if not self.database.password or self.database.password == "zerogravity_pass":
                errors.append("Database password must be set for production environment")
            
            if not self.llm_providers.openai_api_key:
                errors.append("OpenAI API key must be set for production environment")
        
        # Validate database configuration
        if not self.database.host:
            errors.append("Database host must be specified")
        
        # Validate Redis configuration
        if not self.redis.host:
            errors.append("Redis host must be specified")
        
        # Validate service configurations
        if self.api_gateway.replicas < 1:
            errors.append("API Gateway must have at least 1 replica")
        
        if self.agent_coordinator.replicas < 1:
            errors.append("Agent Coordinator must have at least 1 replica")
        
        return errors


class ConfigManager:
    """Manages deployment configurations"""
    
    def __init__(self):
        self.configs: Dict[Environment, DeploymentConfig] = {}
        self.current_config: Optional[DeploymentConfig] = None
    
    def load_config(self, environment: Environment, config_path: Optional[str] = None) -> DeploymentConfig:
        """Load configuration for a specific environment"""
        if config_path:
            # Load from file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create config from data (simplified - in reality you'd have a more complex mapping)
            config = self._create_config_from_data(environment, config_data)
        else:
            # Create default config for environment
            config = self._create_default_config(environment)
        
        self.configs[environment] = config
        self.current_config = config
        return config
    
    def _create_default_config(self, environment: Environment) -> DeploymentConfig:
        """Create default configuration for environment"""
        config = DeploymentConfig(environment=environment)
        
        if environment == Environment.DEVELOPMENT:
            # Development-specific defaults
            config.database = DatabaseConfig(
                host="localhost",
                port=5432,
                name="zerogravity_dev",
                user="dev_user",
                password="dev_pass"
            )
            config.redis = RedisConfig(host="localhost", port=6379)
            config.security = SecurityConfig(
                jwt_secret="dev-jwt-secret-key-change-in-production",
                cors_origins=["http://localhost:3000", "http://127.0.0.1:3000"]
            )
            config.replica_count = 1
            config.security.logging_level = "DEBUG"
        
        elif environment == Environment.STAGING:
            # Staging-specific defaults
            config.database = DatabaseConfig(
                host="staging-db.zerogravity.internal",
                port=5432,
                name="zerogravity_staging",
                user="staging_user",
                password=os.getenv("STAGING_DB_PASSWORD", "")
            )
            config.redis = RedisConfig(
                host="staging-redis.zerogravity.internal", 
                port=6379
            )
            config.security = SecurityConfig(
                jwt_secret=os.getenv("STAGING_JWT_SECRET", ""),
                cors_origins=["https://staging.zerogravity.ai"]
            )
            config.replica_count = 2
        
        elif environment == Environment.PRODUCTION:
            # Production-specific defaults
            config.database = DatabaseConfig(
                host=os.getenv("PROD_DB_HOST", "prod-db.zerogravity.internal"),
                port=5432,
                name="zerogravity_prod",
                user="prod_user",
                password=os.getenv("PROD_DB_PASSWORD", "")
            )
            config.redis = RedisConfig(
                host=os.getenv("PROD_REDIS_HOST", "prod-redis.zerogravity.internal"), 
                port=6379
            )
            config.security = SecurityConfig(
                jwt_secret=os.getenv("PROD_JWT_SECRET", ""),
                cors_origins=["https://zerogravity.ai"],
                ssl_enabled=True
            )
            config.replica_count = 3
            config.llm_providers.openai_api_key = os.getenv("OPENAI_API_KEY")
            config.llm_providers.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        else:  # TESTING
            config.database = DatabaseConfig(
                host="localhost",
                port=5432,
                name="zerogravity_test",
                user="test_user",
                password="test_pass"
            )
            config.redis = RedisConfig(host="localhost", port=6379)
            config.security = SecurityConfig(
                jwt_secret="test-jwt-secret",
                cors_origins=["*"]
            )
            config.replica_count = 1
        
        return config
    
    def _create_config_from_data(self, environment: Environment, data: Dict[str, Any]) -> DeploymentConfig:
        """Create configuration from data dictionary"""
        # This would involve mapping the data to the DeploymentConfig structure
        # For brevity, we'll just return a default config with environment
        return self._create_default_config(environment)
    
    def get_config(self, environment: Environment) -> Optional[DeploymentConfig]:
        """Get configuration for environment"""
        return self.configs.get(environment)
    
    def get_current_config(self) -> Optional[DeploymentConfig]:
        """Get the currently active configuration"""
        return self.current_config
    
    def validate_config(self, environment: Environment) -> bool:
        """Validate configuration for environment"""
        config = self.get_config(environment)
        if not config:
            return False
        
        errors = config.validate()
        return len(errors) == 0


# Global config manager instance
config_manager = ConfigManager()


def get_deployment_config(environment: str = None) -> Optional[DeploymentConfig]:
    """Get deployment configuration for environment"""
    env = Environment(environment) if environment else Environment.DEVELOPMENT
    return config_manager.get_config(env)


def load_environment_config() -> DeploymentConfig:
    """Load configuration based on environment variable"""
    env_str = os.getenv("ENVIRONMENT", "development")
    environment = Environment(env_str.lower())
    
    config = config_manager.load_config(environment)
    
    # Validate config
    if not config_manager.validate_config(environment):
        errors = config.validate()
        raise ValueError(f"Invalid configuration: {errors}")
    
    return config


def create_kubernetes_manifests(config: DeploymentConfig) -> Dict[str, Any]:
    """Create Kubernetes manifests from deployment config"""
    manifests = {
        "namespace": {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": config.project_name
            }
        },
        "api_gateway": _create_api_gateway_deployment(config),
        "agent_coordinator": _create_agent_coordinator_deployment(config),
        "database": _create_database_deployment(config),
        "redis": _create_redis_deployment(config),
        "service_accounts": _create_service_accounts(config),
        "config_maps": _create_config_maps(config),
        "secrets": _create_secrets(config),
        "services": _create_services(config),
        "ingresses": _create_ingress(config)
    }
    
    return manifests


def _create_api_gateway_deployment(config: DeploymentConfig) -> Dict[str, Any]:
    """Create API Gateway deployment manifest"""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{config.project_name}-api-gateway",
            "namespace": config.project_name,
            "labels": {
                "app": f"{config.project_name}-api-gateway",
                "version": config.version
            }
        },
        "spec": {
            "replicas": config.api_gateway.replicas,
            "selector": {
                "matchLabels": {
                    "app": f"{config.project_name}-api-gateway"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": f"{config.project_name}-api-gateway",
                        "version": config.version
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "api-gateway",
                        "image": f"{config.project_name}/api-gateway:{config.version}",
                        "ports": [{"containerPort": 8000}],
                        "env": [{"name": k, "value": v} for k, v in config.get_environment_variables().items()],
                        "resources": config.api_gateway.resources,
                        "livenessProbe": {
                            "httpGet": {
                                "path": config.api_gateway.health_check["path"],
                                "port": config.api_gateway.health_check["port"]
                            },
                            "initialDelaySeconds": 30,
                            "periodSeconds": config.api_gateway.health_check["interval"]
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": config.api_gateway.health_check["path"],
                                "port": config.api_gateway.health_check["port"]
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10
                        }
                    }],
                    "imagePullSecrets": [{"name": "regcred"}] if config.image_pull_policy == "Always" else []
                }
            }
        }
    }


def _create_agent_coordinator_deployment(config: DeploymentConfig) -> Dict[str, Any]:
    """Create Agent Coordinator deployment manifest"""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{config.project_name}-agent-coordinator",
            "namespace": config.project_name,
            "labels": {
                "app": f"{config.project_name}-agent-coordinator",
                "version": config.version
            }
        },
        "spec": {
            "replicas": config.agent_coordinator.replicas,
            "selector": {
                "matchLabels": {
                    "app": f"{config.project_name}-agent-coordinator"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": f"{config.project_name}-agent-coordinator",
                        "version": config.version
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "agent-coordinator",
                        "image": f"{config.project_name}/agent-coordinator:{config.version}",
                        "ports": [{"containerPort": 8001}],
                        "env": [{"name": k, "value": v} for k, v in config.get_environment_variables().items()],
                        "resources": config.agent_coordinator.resources,
                        "livenessProbe": {
                            "httpGet": {
                                "path": config.agent_coordinator.health_check["path"],
                                "port": config.agent_coordinator.health_check["port"]
                            },
                            "initialDelaySeconds": 30,
                            "periodSeconds": config.agent_coordinator.health_check["interval"]
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": config.agent_coordinator.health_check["path"],
                                "port": config.agent_coordinator.health_check["port"]
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10
                        }
                    }]
                }
            }
        }
    }


def _create_database_deployment(config: DeploymentConfig) -> Dict[str, Any]:
    """Create database deployment manifest"""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{config.project_name}-database",
            "namespace": config.project_name,
            "labels": {
                "app": f"{config.project_name}-database",
                "version": config.version
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": f"{config.project_name}-database"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": f"{config.project_name}-database",
                        "version": config.version
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "postgres",
                        "image": "postgres:13",
                        "ports": [{"containerPort": 5432}],
                        "env": [
                            {"name": "POSTGRES_DB", "value": config.database.name},
                            {"name": "POSTGRES_USER", "value": config.database.user},
                            {"name": "POSTGRES_PASSWORD", "valueFrom": {
                                "secretKeyRef": {
                                    "name": f"{config.project_name}-db-secrets",
                                    "key": "password"
                                }
                            }}
                        ],
                        "volumeMounts": [{
                            "name": "postgres-storage",
                            "mountPath": "/var/lib/postgresql/data",
                            "subPath": "postgres"
                        }],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "500m", "memory": "1Gi"}
                        }
                    }],
                    "volumes": [{
                        "name": "postgres-storage",
                        "persistentVolumeClaim": {
                            "claimName": f"{config.project_name}-postgres-pvc"
                        }
                    }]
                }
            }
        }
    }


def _create_redis_deployment(config: DeploymentConfig) -> Dict[str, Any]:
    """Create Redis deployment manifest"""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{config.project_name}-redis",
            "namespace": config.project_name,
            "labels": {
                "app": f"{config.project_name}-redis",
                "version": config.version
            }
        },
        "spec": {
            "replicas": 1,
            "selector": {
                "matchLabels": {
                    "app": f"{config.project_name}-redis"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": f"{config.project_name}-redis",
                        "version": config.version
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "redis",
                        "image": "redis:6-alpine",
                        "ports": [{"containerPort": 6379}],
                        "resources": {
                            "requests": {"cpu": "50m", "memory": "64Mi"},
                            "limits": {"cpu": "200m", "memory": "256Mi"}
                        }
                    }]
                }
            }
        }
    }


def _create_service_accounts(config: DeploymentConfig) -> Dict[str, Any]:
    """Create service accounts"""
    return {
        "apiVersion": "v1",
        "kind": "ServiceAccount",
        "metadata": {
            "name": f"{config.project_name}-service-account",
            "namespace": config.project_name
        }
    }


def _create_config_maps(config: DeploymentConfig) -> Dict[str, Any]:
    """Create config maps"""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": f"{config.project_name}-config",
            "namespace": config.project_name
        },
        "data": {
            "APP_ENV": config.environment.value,
            "PROJECT_NAME": config.project_name,
            "VERSION": config.version,
            "LOG_LEVEL": config.security.logging_level
        }
    }


def _create_secrets(config: DeploymentConfig) -> Dict[str, Any]:
    """Create secrets"""
    import base64
    
    secrets_data = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": f"{config.project_name}-secrets",
            "namespace": config.project_name
        },
        "type": "Opaque",
        "data": {
            "jwt-secret": base64.b64encode(config.security.jwt_secret.encode()).decode(),
            "db-password": base64.b64encode(config.database.password.encode()).decode(),
            "openai-api-key": base64.b64encode((config.llm_providers.openai_api_key or "").encode()).decode(),
            "anthropic-api-key": base64.b64encode((config.llm_providers.anthropic_api_key or "").encode()).decode()
        }
    }
    
    return secrets_data


def _create_services(config: DeploymentConfig) -> Dict[str, Any]:
    """Create services"""
    services = []
    
    # API Gateway Service
    services.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{config.project_name}-api-gateway-svc",
            "namespace": config.project_name
        },
        "spec": {
            "selector": {
                "app": f"{config.project_name}-api-gateway"
            },
            "ports": [{
                "protocol": "TCP",
                "port": 80,
                "targetPort": 8000,
                "name": "http"
            }],
            "type": "ClusterIP"
        }
    })
    
    # Agent Coordinator Service
    services.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{config.project_name}-agent-coordinator-svc",
            "namespace": config.project_name
        },
        "spec": {
            "selector": {
                "app": f"{config.project_name}-agent-coordinator"
            },
            "ports": [{
                "protocol": "TCP",
                "port": 80,
                "targetPort": 8001,
                "name": "http"
            }],
            "type": "ClusterIP"
        }
    })
    
    # Database Service
    services.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{config.project_name}-database-svc",
            "namespace": config.project_name
        },
        "spec": {
            "selector": {
                "app": f"{config.project_name}-database"
            },
            "ports": [{
                "protocol": "TCP",
                "port": 5432,
                "targetPort": 5432,
                "name": "postgres"
            }],
            "type": "ClusterIP"
        }
    })
    
    # Redis Service
    services.append({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{config.project_name}-redis-svc",
            "namespace": config.project_name
        },
        "spec": {
            "selector": {
                "app": f"{config.project_name}-redis"
            },
            "ports": [{
                "protocol": "TCP",
                "port": 6379,
                "targetPort": 6379,
                "name": "redis"
            }],
            "type": "ClusterIP"
        }
    })
    
    return services


def _create_ingress(config: DeploymentConfig) -> Dict[str, Any]:
    """Create ingress"""
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": f"{config.project_name}-ingress",
            "namespace": config.project_name,
            "annotations": {
                "kubernetes.io/ingress.class": "nginx",
                "cert-manager.io/cluster-issuer": "letsencrypt-prod"
            }
        },
        "spec": {
            "tls": [{
                "hosts": [config.domain],
                "secretName": f"{config.project_name}-tls"
            }],
            "rules": [{
                "host": config.domain,
                "http": {
                    "paths": [{
                        "path": "/api",
                        "pathType": "Prefix",
                        "backend": {
                            "service": {
                                "name": f"{config.project_name}-api-gateway-svc",
                                "port": {"number": 80}
                            }
                        }
                    }]
                }
            }]
        }
    }


def generate_docker_compose(config: DeploymentConfig) -> Dict[str, Any]:
    """Generate Docker Compose configuration"""
    compose_config = {
        "version": "3.8",
        "services": {
            "api_gateway": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.api"
                },
                "ports": ["8000:8000"],
                "environment": config.get_environment_variables(),
                "depends_on": ["database", "redis"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": f"{config.api_gateway.health_check['interval']}s",
                    "timeout": f"{config.api_gateway.health_check['timeout']}s",
                    "retries": 3
                }
            },
            "agent_coordinator": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.agent"
                },
                "ports": ["8001:8001"],
                "environment": config.get_environment_variables(),
                "depends_on": ["database", "redis"],
                "restart": "unless-stopped"
            },
            "database": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": config.database.name,
                    "POSTGRES_USER": config.database.user,
                    "POSTGRES_PASSWORD": config.database.password
                },
                "ports": ["5432:5432"],
                "volumes": ["./data/postgres:/var/lib/postgresql/data"],
                "restart": "unless-stopped"
            },
            "redis": {
                "image": "redis:6-alpine",
                "ports": ["6379:6379"],
                "restart": "unless-stopped"
            },
            "task_queue": {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile.task_queue"
                },
                "environment": config.get_environment_variables(),
                "depends_on": ["redis"],
                "restart": "unless-stopped"
            }
        },
        "volumes": {
            "postgres_data": {}
        }
    }
    
    return compose_config


# Example usage and initialization
def init_deployment(environment: str = "development") -> DeploymentConfig:
    """Initialize deployment configuration"""
    env = Environment(environment)
    config = config_manager.load_config(env)
    
    # Validate the configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed: {errors}")
    
    print(f"Deployment configuration loaded for {environment} environment")
    print(f"Project: {config.project_name}")
    print(f"Version: {config.version}")
    print(f"Environment: {config.environment.value}")
    print(f"Services: {len([s for s in [config.api_gateway, config.agent_coordinator, config.task_queue]])}")
    
    return config


if __name__ == "__main__":
    # For testing purposes
    print("Initializing ZeroGravity deployment configuration...")
    
    try:
        # Load config based on environment variable or default to development
        config = load_environment_config()
        
        print(f"Configuration loaded for {config.environment.value} environment")
        print(f"Database: {config.database.host}:{config.database.port}/{config.database.name}")
        print(f"Redis: {config.redis.host}:{config.redis.port}")
        print(f"Replicas: {config.replica_count}")
        
        # Validate config
        errors = config.validate()
        if errors:
            print(f"Configuration errors: {errors}")
        else:
            print("Configuration is valid")
        
        # Generate Kubernetes manifests
        print("\nGenerating Kubernetes manifests...")
        manifests = create_kubernetes_manifests(config)
        print(f"Generated {len(manifests)} manifests")
        
        # Generate Docker Compose
        print("\nGenerating Docker Compose configuration...")
        compose = generate_docker_compose(config)
        print(f"Docker Compose services: {list(compose['services'].keys())}")
        
    except Exception as e:
        print(f"Error initializing deployment: {e}")
        import traceback
        traceback.print_exc()
