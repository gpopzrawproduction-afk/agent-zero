"""
Plugin Architecture System for ZeroGravity

This module implements a flexible plugin architecture that allows
for extending the ZeroGravity platform with custom agents, tools, and functionality.
"""
import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import yaml
from datetime import datetime


class PluginType(Enum):
    """Types of plugins supported by ZeroGravity"""
    AGENT = "agent"
    TOOL = "tool"
    WORKFLOW = "workflow"
    LLM_PROVIDER = "llm_provider"
    MIDDLEWARE = "middleware"
    EXTENSION = "extension"


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    description: str
    author: str
    license: str
    dependencies: List[str]
    type: PluginType
    created_at: datetime
    updated_at: datetime
    config_schema: Optional[Dict[str, Any]] = None


class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metadata = self.get_metadata()
        self.initialized = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality"""
        pass
    
    def cleanup(self) -> bool:
        """Clean up resources when plugin is unloaded"""
        return True


class AgentPlugin(BasePlugin):
    """Base class for agent plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.role = self.config.get("role", "custom_agent")
        self.system_prompt = self.config.get("system_prompt", "")
    
    @abstractmethod
    def process_input(self, input_data: Any) -> Any:
        """Process input data and return result"""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the agent plugin"""
        input_data = kwargs.get("input_data") or (args[0] if args else None)
        return self.process_input(input_data)


class ToolPlugin(BasePlugin):
    """Base class for tool plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = self.config.get("name", "custom_tool")
        self.description = self.config.get("description", "")
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Run the tool with given parameters"""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool plugin"""
        return self.run(*args, **kwargs)


class WorkflowPlugin(BasePlugin):
    """Base class for workflow plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = self.config.get("name", "custom_workflow")
    
    @abstractmethod
    def execute_workflow(self, objective: str) -> Dict[str, Any]:
        """Execute the custom workflow"""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the workflow plugin"""
        objective = kwargs.get("objective") or (args[0] if args else None)
        return self.execute_workflow(objective)


class LLMProviderPlugin(BasePlugin):
    """Base class for LLM provider plugins"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.provider_name = self.config.get("provider_name", "custom_llm")
    
    @abstractmethod
    def call(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """Call the LLM provider"""
        pass
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the LLM provider plugin"""
        messages = kwargs.get("messages") or (args[0] if args else None)
        model = kwargs.get("model", "default")
        return self.call(messages, model, **kwargs)


class PluginManager:
    """Manages loading, registering, and executing plugins"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_types: Dict[PluginType, Dict[str, BasePlugin]] = {
            PluginType.AGENT: {},
            PluginType.TOOL: {},
            PluginType.WORKFLOW: {},
            PluginType.LLM_PROVIDER: {},
            PluginType.MIDDLEWARE: {},
            PluginType.EXTENSION: {}
        }
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory"""
        discovered_plugins = []
        
        if not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True, exist_ok=True)
            return discovered_plugins
        
        # Look for Python files in the plugins directory
        for plugin_file in self.plugins_dir.rglob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            plugin_name = plugin_file.stem
            if self._is_plugin_file(plugin_file):
                discovered_plugins.append(str(plugin_file))
        
        return discovered_plugins
    
    def _is_plugin_file(self, file_path: Path) -> bool:
        """Check if a file is a valid plugin file"""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None:
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module contains plugin classes
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin
                ):
                    return True
        except Exception:
            pass
        
        return False
    
    def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> Optional[BasePlugin]:
        """Load a plugin from a file path"""
        try:
            plugin_path = Path(plugin_path)
            spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class in the module
            plugin_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin
                ):
                    plugin_class = obj
                    break
            
            if plugin_class is None:
                raise ValueError(f"No plugin class found in {plugin_path}")
            
            # Create plugin instance
            plugin = plugin_class(config or {})
            
            # Initialize plugin
            if plugin.initialize():
                # Register plugin
                self.register_plugin(plugin)
                return plugin
            else:
                raise ValueError(f"Failed to initialize plugin: {plugin.metadata.name}")
                
        except Exception as e:
            print(f"Error loading plugin {plugin_path}: {e}")
            return None
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin with the manager"""
        try:
            plugin_name = plugin.metadata.name
            
            # Check if plugin with same name already exists
            if plugin_name in self.plugins:
                print(f"Plugin {plugin_name} already registered")
                return False
            
            # Register in main plugins dict
            self.plugins[plugin_name] = plugin
            self.plugin_metadata[plugin_name] = plugin.metadata
            
            # Register in type-specific dict
            plugin_type = plugin.metadata.type
            if plugin_type in self.plugin_types:
                self.plugin_types[plugin_type][plugin_name] = plugin
            
            print(f"Plugin {plugin_name} registered successfully")
            return True
            
        except Exception as e:
            print(f"Error registering plugin {plugin.metadata.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        
        # Clean up plugin
        plugin.cleanup()
        
        # Remove from main dict
        del self.plugins[plugin_name]
        del self.plugin_metadata[plugin_name]
        
        # Remove from type-specific dict
        plugin_type = plugin.metadata.type
        if plugin_type in self.plugin_types and plugin_name in self.plugin_types[plugin_type]:
            del self.plugin_types[plugin_type][plugin_name]
        
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, BasePlugin]:
        """Get all plugins of a specific type"""
        return self.plugin_types.get(plugin_type, {}).copy()
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin by name"""
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        return plugin.execute(*args, **kwargs)
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins"""
        results = {}
        discovered = self.discover_plugins()
        
        for plugin_path in discovered:
            plugin_name = Path(plugin_path).stem
            plugin = self.load_plugin(plugin_path)
            results[plugin_name] = plugin is not None
        
        return results
    
    def validate_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration against schema"""
        metadata = self.plugin_metadata.get(plugin_name)
        if not metadata or not metadata.config_schema:
            return True  # No schema to validate against
        
        # Simple validation based on schema structure
        schema = metadata.config_schema
        for key, expected_type in schema.items():
            if key in config:
                actual_value = config[key]
                expected_python_type = self._map_json_type_to_python(expected_type)
                
                if not isinstance(actual_value, expected_python_type):
                    return False
        
        return True
    
    def _map_json_type_to_python(self, json_type: str) -> type:
        """Map JSON schema types to Python types"""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_mapping.get(json_type, object)


class PluginRegistry:
    """Registry for plugin metadata and configuration"""
    
    def __init__(self, registry_file: str = "plugin_registry.json"):
        self.registry_file = Path(registry_file)
        self.plugins: Dict[str, PluginMetadata] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load plugin registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, metadata_dict in data.items():
                        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                        metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                        metadata_dict['type'] = PluginType(metadata_dict['type'])
                        self.plugins[name] = PluginMetadata(**metadata_dict)
            except Exception as e:
                print(f"Error loading plugin registry: {e}")
    
    def save_registry(self):
        """Save plugin registry to file"""
        try:
            # Convert PluginMetadata objects to dictionaries
            data = {}
            for name, metadata in self.plugins.items():
                metadata_dict = metadata.__dict__.copy()
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['updated_at'] = metadata.updated_at.isoformat()
                metadata_dict['type'] = metadata.type.value
                data[name] = metadata_dict
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving plugin registry: {e}")
    
    def register_plugin(self, metadata: PluginMetadata):
        """Register a plugin in the registry"""
        self.plugins[metadata.name] = metadata
        self.save_registry()
    
    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin from the registry"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.save_registry()
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin"""
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, PluginMetadata]:
        """Get all registered plugins"""
        return self.plugins.copy()


class PluginConfig:
    """Configuration management for plugins"""
    
    def __init__(self, config_file: str = "plugin_config.yaml"):
        self.config_file = Path(config_file)
        self.config: Dict[str, Dict[str, Any]] = {}
        self.load_config()
    
    def load_config(self):
        """Load plugin configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading plugin config: {e}")
                self.config = {}
    
    def save_config(self):
        """Save plugin configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving plugin config: {e}")
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin"""
        return self.config.get(plugin_name, {})
    
    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]):
        """Set configuration for a specific plugin"""
        self.config[plugin_name] = config
        self.save_config()
    
    def update_plugin_config(self, plugin_name: str, updates: Dict[str, Any]):
        """Update configuration for a specific plugin"""
        if plugin_name not in self.config:
            self.config[plugin_name] = {}
        
        self.config[plugin_name].update(updates)
        self.save_config()


# Global plugin manager instance
plugin_manager = PluginManager()
plugin_registry = PluginRegistry()
plugin_config = PluginConfig()


def create_plugin_template(plugin_type: PluginType, name: str, output_dir: str = "plugins") -> str:
    """Create a template for a new plugin"""
    output_path = Path(output_dir) / f"{name}.py"
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select template based on plugin type
    if plugin_type == PluginType.AGENT:
        template = f'''"""
{name.capitalize()} Agent Plugin for ZeroGravity

This is a template for a custom agent plugin.
"""

from zero_gravity_core.plugin_system import AgentPlugin, PluginMetadata, PluginType
from datetime import datetime


class {name.capitalize()}Agent(AgentPlugin):
    """Custom agent plugin template"""
    
    def get_metadata(self):
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            description="A custom agent plugin",
            author="Your Name",
            license="MIT",
            dependencies=[],
            type=PluginType.AGENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config_schema={{
                "parameter1": "string",
                "parameter2": "number"
            }}
        )
    
    def initialize(self):
        """Initialize the agent plugin"""
        # Add initialization code here
        self.initialized = True
        return True
    
    def process_input(self, input_data):
        """Process input and return result"""
        # Add your agent logic here
        return f"Processed: {{input_data}}"
'''
    
    elif plugin_type == PluginType.TOOL:
        template = f'''"""
{name.capitalize()} Tool Plugin for ZeroGravity

This is a template for a custom tool plugin.
"""

from zero_gravity_core.plugin_system import ToolPlugin, PluginMetadata, PluginType
from datetime import datetime


class {name.capitalize()}Tool(ToolPlugin):
    """Custom tool plugin template"""
    
    def get_metadata(self):
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            description="A custom tool plugin",
            author="Your Name",
            license="MIT",
            dependencies=[],
            type=PluginType.TOOL,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config_schema={{
                "parameter1": "string",
                "parameter2": "number"
            }}
        )
    
    def initialize(self):
        """Initialize the tool plugin"""
        # Add initialization code here
        self.initialized = True
        return True
    
    def run(self, *args, **kwargs):
        """Run the tool with given parameters"""
        # Add your tool logic here
        return {{"result": "Tool executed successfully", "args": args, "kwargs": kwargs}}
'''
    
    elif plugin_type == PluginType.WORKFLOW:
        template = f'''"""
{name.capitalize()} Workflow Plugin for ZeroGravity

This is a template for a custom workflow plugin.
"""

from zero_gravity_core.plugin_system import WorkflowPlugin, PluginMetadata, PluginType
from datetime import datetime


class {name.capitalize()}Workflow(WorkflowPlugin):
    """Custom workflow plugin template"""
    
    def get_metadata(self):
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            description="A custom workflow plugin",
            author="Your Name",
            license="MIT",
            dependencies=[],
            type=PluginType.WORKFLOW,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config_schema={{
                "parameter1": "string",
                "parameter2": "number"
            }}
        )
    
    def initialize(self):
        """Initialize the workflow plugin"""
        # Add initialization code here
        self.initialized = True
        return True
    
    def execute_workflow(self, objective):
        """Execute the custom workflow"""
        # Add your workflow logic here
        return {{
            "status": "completed",
            "result": f"Workflow completed for objective: {{objective}}",
            "steps": ["step1", "step2", "step3"]
        }}
'''
    
    else:
        # Default template
        template = f'''"""
{name.capitalize()} Plugin for ZeroGravity

This is a template for a custom plugin.
"""

from zero_gravity_core.plugin_system import BasePlugin, PluginMetadata, PluginType
from datetime import datetime


class {name.capitalize()}Plugin(BasePlugin):
    """Custom plugin template"""
    
    def get_metadata(self):
        return PluginMetadata(
            name="{name}",
            version="1.0.0",
            description="A custom plugin",
            author="Your Name",
            license="MIT",
            dependencies=[],
            type=PluginType.EXTENSION,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            config_schema={{
                "parameter1": "string",
                "parameter2": "number"
            }}
        )
    
    def initialize(self):
        """Initialize the plugin"""
        # Add initialization code here
        self.initialized = True
        return True
    
    def execute(self, *args, **kwargs):
        """Execute the plugin's main functionality"""
        # Add your plugin logic here
        return "Plugin executed successfully"
'''
    
    # Write template to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return str(output_path)


def install_plugin_from_url(url: str, plugin_manager_instance: PluginManager = None) -> bool:
    """
    Install a plugin from a URL (simplified implementation)
    
    In a real implementation, this would download and install
    a plugin from a remote source.
    """
    if plugin_manager_instance is None:
        plugin_manager_instance = plugin_manager
    
    print(f"Installing plugin from URL: {url}")
    # In a real implementation, you would:
    # 1. Download the plugin package
    # 2. Extract it to the plugins directory
    # 3. Load and register the plugin
    
    # For now, just return True to indicate success
    return True


def list_available_plugins() -> List[str]:
    """List all available plugins in the plugins directory"""
    return plugin_manager.discover_plugins()


def get_plugin_info(plugin_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific plugin"""
    plugin = plugin_manager.get_plugin(plugin_name)
    if plugin:
        metadata = plugin.metadata
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "type": metadata.type.value,
            "initialized": plugin.initialized
        }
    
    # Check if it's a registered plugin that's not loaded
    metadata = plugin_registry.get_plugin_metadata(plugin_name)
    if metadata:
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "type": metadata.type.value,
            "initialized": False
        }
    
    return None


# Initialize the plugin system
def init_plugin_system(plugins_dir: str = "plugins"):
    """Initialize the plugin system"""
    global plugin_manager
    plugin_manager = PluginManager(plugins_dir)
    
    # Load all available plugins
    load_results = plugin_manager.load_all_plugins()
    print(f"Loaded plugins: {load_results}")
    
    return plugin_manager


# Example usage
if __name__ == "__main__":
    # Initialize plugin system
    pm = init_plugin_system()
    
    # Create a sample plugin template
    template_path = create_plugin_template(PluginType.AGENT, "sample_agent")
    print(f"Created plugin template at: {template_path}")
    
    # List available plugins
    available = list_available_plugins()
    print(f"Available plugins: {available}")
    
    # Load and execute a plugin (if any exist)
    if available:
        plugin_name = Path(available[0]).stem
        print(f"Loading plugin: {plugin_name}")
        
        # Try to load the plugin
        plugin = pm.load_plugin(available[0])
        if plugin:
            print(f"Plugin loaded: {plugin.metadata.name}")
            result = plugin.execute(input_data="Hello, plugin!")
            print(f"Plugin result: {result}")
