# Configuration for AI Ops Agent
# Defines configuration settings for the AI Ops system

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class AIOpsConfig:
    """
    Configuration for the AI Ops Agent system
    """
    # General settings
    enabled: bool = True
    log_level: str = "INFO"
    storage_path: str = "zero_gravity_core/ai_ops_data"
    
    # Telemetry settings
    telemetry_enabled: bool = True
    telemetry_buffer_size: int = 1000
    telemetry_flush_interval: int = 30  # seconds
    telemetry_storage_path: str = "zero_gravity_core/monitoring/telemetry_data"
    
    # Decision engine settings
    decision_engine_enabled: bool = True
    model_cost_map: Dict[str, float] = None
    performance_thresholds: Dict[str, float] = None
    
    # Policy engine settings
    policy_engine_enabled: bool = True
    policy_storage_path: str = "zero_gravity_core/policies"
    
    # Optimization engine settings
    optimization_engine_enabled: bool = True
    optimization_storage_path: str = "zero_gravity_core/optimization_data"
    optimization_thresholds: Dict[str, float] = None
    
    # Escalation controller settings
    escalation_controller_enabled: bool = True
    escalation_storage_path: str = "zero_gravity_core/escalation_data"
    escalation_triggers: Dict[str, Any] = None
    
    # Integration settings
    agent_integration_enabled: bool = True
    workflow_integration_enabled: bool = True
    
    def __post_init__(self):
        """Set default values for complex fields"""
        if self.model_cost_map is None:
            self.model_cost_map = {
                "gpt-3.5-turbo": 0.002,
                "gpt-4": 0.03,
                "gpt-4-32k": 0.06,
                "claude-2": 0.01102,
                "claude-instant-1": 0.00163
            }
        
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                "latency_ms": 500,
                "cost_per_task": 1.0,
                "success_rate": 0.85,
                "quality_score": 0.8
            }
        
        if self.optimization_thresholds is None:
            self.optimization_thresholds = {
                "low_success_rate": 0.7,
                "high_cost_per_task": 0.5,
                "high_latency": 5000,
                "low_quality": 0.7
            }
        
        if self.escalation_triggers is None:
            self.escalation_triggers = {
                "policy_violation": {
                    "critical": ["block", "escalate"],
                    "high_threshold": 1,
                    "medium_threshold": 3
                },
                "repeated_failures": {
                    "threshold": 3
                },
                "high_cost": {
                    "threshold": 5.0
                },
                "legal_financial_risk": {
                    "keywords": ["legal", "financial", "compliance", "regulatory", "audit", "contract", "agreement"]
                },
                "ambiguous_output": {
                    "low_quality_threshold": 0.3
                },
                "high_value_decisions": {
                    "high_priority_threshold": 9,
                    "high_impact_threshold": 1000
                }
            }

class AIOpsConfigManager:
    """
    Manages configuration for the AI Ops system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "zero_gravity_core/config/ai_ops_config.json"
        self.config = self._get_default_config()
    
    def _get_default_config(self) -> AIOpsConfig:
        """Get default configuration"""
        return AIOpsConfig()
    
    async def load_config(self) -> AIOpsConfig:
        """Load configuration from file"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Create config object from dict
                # For simplicity, we'll directly assign values
                config = AIOpsConfig()
                
                # Update config with values from file
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                
                self.config = config
                return config
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
                # Return default config if loading fails
                self.config = self._get_default_config()
                return self.config
        else:
            # Create default config file if it doesn't exist
            await self.save_config(self.config)
            return self.config
    
    async def save_config(self, config: AIOpsConfig = None) -> bool:
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        try:
            config_file = Path(self.config_path)
            
            # Create directory if it doesn't exist
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save config as JSON
            with open(config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving config to {self.config_path}: {e}")
            return False
    
    async def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            # Update current config with new values
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Save updated config
            return await self.save_config(self.config)
        except Exception as e:
            print(f"Error updating config: {e}")
            return False
    
    def get_config(self) -> AIOpsConfig:
        """Get current configuration"""
        return self.config

# Global configuration instance
ai_ops_config_manager = AIOpsConfigManager()

async def get_ai_ops_config() -> AIOpsConfig:
    """Get the AI Ops configuration"""
    return await ai_ops_config_manager.load_config()

async def update_ai_ops_config(updates: Dict[str, Any]) -> bool:
    """Update the AI Ops configuration"""
    return await ai_ops_config_manager.update_config(updates)
