"""
Client SDK Generation System for ZeroGravity

This module generates client SDKs for different programming languages
to interact with the ZeroGravity API.
"""
import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import jinja2
from jinja2 import Template


class Language(Enum):
    """Supported programming languages for SDK generation"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"


@dataclass
class APISpec:
    """API specification for SDK generation"""
    title: str
    description: str
    version: str
    base_url: str
    endpoints: List[Dict[str, Any]]
    auth: Dict[str, Any]


class SDKTemplateProvider:
    """Provides templates for different programming languages"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[Language, Dict[str, str]]:
        """Load all SDK templates"""
        templates = {}
        
        # Python templates
        templates[Language.PYTHON] = {
            "client": """import requests
import json
from typing import Dict, Any, Optional


class ZeroGravityClient:
    def __init__(self, base_url: str = \"{{ spec.base_url }}\", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
    
    {% for endpoint in spec.endpoints %}
    def {{ endpoint.name }}(self, {% for param in endpoint.parameters %}{{ param.name }}: {{ param.type }}{% if not loop.last %}, {% endif %}{% endfor %}) -> Dict[str, Any]:
        url = f\"{self.base_url}{{ endpoint.path }}\"
        {% if endpoint.method == 'GET' %}
        params = {
            {% for param in endpoint.parameters %}
            \"{{ param.name }}\": {{ param.name }}{% if not loop.last %},{% endif %}
            {% endfor %}
        }
        response = self.session.{{ endpoint.method.lower() }}(url, params=params)
        {% else %}
        data = {
            {% for param in endpoint.parameters %}
            \"{{ param.name }}\": {{ param.name }}{% if not loop.last %},{% endif %}
            {% endfor %}
        }
        response = self.session.{{ endpoint.method.lower() }}(url, json=data)
        {% endif %}
        
        response.raise_for_status()
        return response.json()
    
    {% endfor %}
    
    def close(self):
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
""",
            "setup": """from setuptools import setup, find_packages

setup(
    name=\"zerogravity-client\",
    version=\"{{ spec.version }}\",
    packages=find_packages(),
    install_requires=[
        \"requests>=2.25.0\",
    ],
    author=\"ZeroGravity\",
    author_email=\"support@zerogravity.ai\",
    description=\"Client library for ZeroGravity API\",
    long_description=open(\"README.md\").read(),
    long_description_content_type=\"text/markdown\",
    url=\"https://github.com/zerogravity/client-python\",
    classifiers=[
        \"Programming Language :: Python :: 3\",
        \"License :: OSI Approved :: MIT License\",
        \"Operating System :: OS Independent\",
    ],
    python_requires='>=3.7',
)
""",
            "readme": """# ZeroGravity Python Client

Python client library for the ZeroGravity API.

## Installation

```bash
pip install zerogravity-client
```

## Usage

```python
from zerogravity import ZeroGravityClient

client = ZeroGravityClient(
    base_url=\"{{ spec.base_url }}\",
    api_key=\"your-api-key\"
)

# Example usage
result = client.submit_objective(objective=\"Build a web application\")
print(result)
```
"""
        }
        
        # JavaScript templates
        templates[Language.JAVASCRIPT] = {
            "client": """class ZeroGravityClient {
    constructor(baseUrl = \"{{ spec.base_url }}\", apiKey = null) {
        this.baseUrl = baseUrl.replace(/\\/$/, '');
        this.apiKey = apiKey;
        this.headers = {
            'Content-Type': 'application/json'
        };
        
        if (apiKey) {
            this.headers['Authorization'] = `Bearer ${apiKey}`;
        }
    }
    
    async _request(method, path, data = null, params = null) {
        let url = `${this.baseUrl}${path}`;
        
        if (params) {
            const searchParams = new URLSearchParams(params);
            url += `?${searchParams}`;
        }
        
        const options = {
            method,
            headers: this.headers
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: \${response.status}`);
        }
        
        return await response.json();
    }
    
    {% for endpoint in spec.endpoints %}
    async {{ endpoint.name }}({% for param in endpoint.parameters %}{{ param.name }}{% if not loop.last %}, {% endif %}{% endfor %}) {
        {% if endpoint.method == 'GET' %}
        const params = {
            {% for param in endpoint.parameters %}
            {{ param.name }}: {{ param.name }}{% if not loop.last %},{% endif %}
            {% endfor %}
        };
        return await this._request('{{ endpoint.method }}', '{{ endpoint.path }}', null, params);
        {% else %}
        const data = {
            {% for param in endpoint.parameters %}
            {{ param.name }}: {{ param.name }}{% if not loop.last %},{% endif %}
            {% endfor %}
        };
        return await this._request('{{ endpoint.method }}', '{{ endpoint.path }}', data);
        {% endif %}
    }
    
    {% endfor %}
}

module.exports = ZeroGravityClient;
""",
            "package": """{
  "name": "zerogravity-client",
  "version": "{{ spec.version }}",
 "description": "{{ spec.description }}",
 "main": "index.js",
  "scripts": {
    "test": "jest"
  },
  "keywords": ["zerogravity", "ai", "api"],
  "author": "ZeroGravity",
  "license": "MIT",
  "dependencies": {
    "node-fetch": "^2.6.0"
  },
  "devDependencies": {
    "jest": "^27.0.0"
  }
}
""",
            "readme": """# ZeroGravity JavaScript Client

JavaScript client library for the ZeroGravity API.

## Installation

```bash
npm install zerogravity-client
```

## Usage

```javascript
const ZeroGravityClient = require('zerogravity-client');

const client = new ZeroGravityClient(
    '{{ spec.base_url }}',
    'your-api-key'
);

// Example usage
const result = await client.submitObjective({ objective: 'Build a web application' });
console.log(result);
```
"""
        }
        
        # TypeScript templates
        templates[Language.TYPESCRIPT] = {
            "client": """interface RequestOptions {
    method: string;
    headers: Record<string, string>;
    body?: string;
}

interface {{ spec.title.replace(' ', '') }}Config {
    baseUrl?: string;
    apiKey?: string;
}

{% for endpoint in spec.endpoints %}
interface {{ endpoint.name | title }}Request {
    {% for param in endpoint.parameters %}
    {{ param.name }}: {{ param.ts_type }};
    {% endfor %}
}

interface {{ endpoint.name | title }}Response {
    // Define response structure based on your API
    [key: string]: any;
}
{% endfor %}

class {{ spec.title.replace(' ', '') }}Client {
    private baseUrl: string;
    private apiKey: string | null;
    private headers: Record<string, string>;

    constructor(config: {{ spec.title.replace(' ', '') }}Config = {}) {
        this.baseUrl = (config.baseUrl || \"{{ spec.base_url }}\").replace(/\\/$/, '');
        this.apiKey = config.apiKey || null;
        this.headers = {
            'Content-Type': 'application/json'
        };

        if (this.apiKey) {
            this.headers['Authorization'] = `Bearer \${this.apiKey}`;
        }
    }

    private async request<T>(method: string, path: string, data: any = null, params: any = null): Promise<T> {
        let url = `\${this.baseUrl}\${path}`;

        if (params) {
            const searchParams = new URLSearchParams();
            for (const [key, value] of Object.entries(params)) {
                searchParams.append(key, String(value));
            }
            url += `?\${searchParams}`;
        }

        const options: RequestOptions = {
            method,
            headers: this.headers
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error(`HTTP error! status: \${response.status}`);
        }

        return await response.json() as T;
    }

    {% for endpoint in spec.endpoints %}
    async {{ endpoint.name }}(request: {{ endpoint.name | title }}Request): Promise<{{ endpoint.name | title }}Response> {
        {% if endpoint.method == 'GET' %}
        return await this.request<{{ endpoint.name | title }}Response>('{{ endpoint.method }}', '{{ endpoint.path }}', null, request);
        {% else %}
        return await this.request<{{ endpoint.name | title }}Response>('{{ endpoint.method }}', '{{ endpoint.path }}', request);
        {% endif %}
    }

    {% endfor %}
}

export default {{ spec.title.replace(' ', '') }}Client;
""",
            "package": """{
  "name": "@zerogravity/client",
  "version": "{{ spec.version }}",
  "description": "{{ spec.description }}",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  },
  "keywords": ["zerogravity", "ai", "api", "typescript"],
  "author": "ZeroGravity",
  "license": "MIT",
  "dependencies": {
    "node-fetch": "^2.6.0"
  },
 "devDependencies": {
    "@types/node": "^16.0.0",
    "typescript": "^4.5.0",
    "jest": "^27.0.0"
  }
}
""",
            "tsconfig": """{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020", "DOM"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "types": ["node", "jest"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "**/*.test.ts"]
}
"""
        }
        
        return templates
    
    def get_template(self, language: Language, template_name: str) -> Template:
        """Get a specific template for a language"""
        if language not in self.templates:
            raise ValueError(f"Language {language} not supported")
        
        if template_name not in self.templates[language]:
            raise ValueError(f"Template {template_name} not available for {language}")
        
        return Template(self.templates[language][template_name])


class SDKGenerator:
    """Generates client SDKs for different programming languages"""
    
    def __init__(self):
        self.template_provider = SDKTemplateProvider()
    
    def generate_sdk(self, spec: APISpec, language: Language) -> bytes:
        """Generate SDK for the specified language"""
        templates = self.template_provider.templates[language]
        
        # Create a temporary directory for the SDK files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Generate the main client file
            client_template = Template(templates["client"])
            client_content = client_template.render(spec=spec)
            
            if language == Language.PYTHON:
                client_file = temp_path / "zerogravity" / "client.py"
                client_file.parent.mkdir(exist_ok=True)
                client_file.write_text(client_content)
                
                # Generate setup.py
                setup_template = Template(templates["setup"])
                setup_content = setup_template.render(spec=spec)
                (temp_path / "setup.py").write_text(setup_content)
                
                # Generate __init__.py
                (temp_path / "zerogravity" / "__init__.py").write_text("from .client import ZeroGravityClient\n\n__all__ = ['ZeroGravityClient']")
                
                # Generate README
                readme_template = Template(templates["readme"])
                readme_content = readme_template.render(spec=spec)
                (temp_path / "README.md").write_text(readme_content)
                
            elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                client_file = temp_path / "index.js" if language == Language.JAVASCRIPT else temp_path / "src" / "index.ts"
                if language == Language.TYPESCRIPT:
                    client_file.parent.mkdir(exist_ok=True)
                client_file.write_text(client_content)
                
                # Generate package.json
                package_template = Template(templates["package"])
                package_content = package_template.render(spec=spec)
                (temp_path / "package.json").write_text(package_content)
                
                # Generate README
                readme_template = Template(templates["readme"])
                readme_content = readme_template.render(spec=spec)
                (temp_path / "README.md").write_text(readme_content)
                
                # For TypeScript, also generate tsconfig.json
                if language == Language.TYPESCRIPT and "tsconfig" in templates:
                    tsconfig_template = Template(templates["tsconfig"])
                    tsconfig_content = tsconfig_template.render(spec=spec)
                    (temp_path / "tsconfig.json").write_text(tsconfig_content)
            
            # Create a zip file containing the SDK
            zip_path = temp_path / f"zerogravity-sdk-{language.value}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file() and file_path != zip_path:
                        arc_name = file_path.relative_to(temp_path)
                        zip_file.write(file_path, arc_name)
            
            # Read the zip file and return as bytes
            return zip_path.read_bytes()
    
    def generate_all_sdks(self, spec: APISpec) -> Dict[Language, bytes]:
        """Generate SDKs for all supported languages"""
        sdks = {}
        for language in self.template_provider.templates.keys():
            sdks[language] = self.generate_sdk(spec, language)
        return sdks


class OpenAPISpecConverter:
    """Converts OpenAPI specifications to internal format"""
    
    @staticmethod
    def from_openapi(openapi_spec: Dict[str, Any]) -> APISpec:
        """Convert OpenAPI spec to internal APISpec format"""
        endpoints = []
        
        for path, methods in openapi_spec.get("paths", {}).items():
            for method, details in methods.items():
                endpoint = {
                    "name": details.get("operationId", f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}"),
                    "path": path,
                    "method": method.upper(),
                    "description": details.get("description", ""),
                    "parameters": []
                }
                
                # Extract parameters
                for param in details.get("parameters", []):
                    param_info = {
                        "name": param["name"],
                        "type": OpenAPISpecConverter._map_type(param.get("schema", {}).get("type", "string")),
                        "required": param.get("required", False),
                        "description": param.get("description", "")
                    }
                    endpoint["parameters"].append(param_info)
                
                # If no parameters from the 'parameters' section, check the request body
                if not endpoint["parameters"] and "requestBody" in details:
                    content = details["requestBody"].get("content", {})
                    if "application/json" in content:
                        schema = content["application/json"].get("schema", {})
                        if "properties" in schema:
                            for prop_name, prop_details in schema["properties"].items():
                                param_info = {
                                    "name": prop_name,
                                    "type": OpenAPISpecConverter._map_type(prop_details.get("type", "string")),
                                    "required": prop_name in schema.get("required", []),
                                    "description": prop_details.get("description", "")
                                }
                                endpoint["parameters"].append(param_info)
                
                endpoints.append(endpoint)
        
        return APISpec(
            title=openapi_spec.get("info", {}).get("title", "ZeroGravity API"),
            description=openapi_spec.get("info", {}).get("description", ""),
            version=openapi_spec.get("info", {}).get("version", "1.0.0"),
            base_url=openapi_spec.get("servers", [{}])[0].get("url", "https://api.zerogravity.ai"),
            endpoints=endpoints,
            auth={}  # Extract auth info from OpenAPI spec if available
        )
    
    @staticmethod
    def _map_type(openapi_type: str) -> str:
        """Map OpenAPI types to language-specific types"""
        type_mapping = {
            "string": "str",
            "integer": "int", 
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict"
        }
        return type_mapping.get(openapi_type, "str")


# Global SDK generator instance
sdk_generator = SDKGenerator()


def generate_sdk(spec: APISpec, language: Language) -> bytes:
    """Convenience function to generate an SDK"""
    return sdk_generator.generate_sdk(spec, language)


def generate_all_sdks(spec: APISpec) -> Dict[Language, bytes]:
    """Convenience function to generate SDKs for all languages"""
    return sdk_generator.generate_all_sdks(spec)


def create_default_spec() -> APISpec:
    """Create a default API specification for ZeroGravity"""
    return APISpec(
        title="ZeroGravity API",
        description="ZeroGravity Multi-Agent AI Platform API",
        version="1.0.0",
        base_url="https://api.zerogravity.ai",
        endpoints=[
            {
                "name": "submit_objective",
                "path": "/api/v1/objective",
                "method": "POST",
                "description": "Submit a new objective for processing",
                "parameters": [
                    {"name": "objective", "type": "str", "ts_type": "string", "required": True, "description": "The objective to process"},
                    {"name": "priority", "type": "str", "ts_type": "string", "required": False, "description": "Priority level (low, normal, high)"},
                    {"name": "callback_url", "type": "str", "ts_type": "string", "required": False, "description": "Callback URL for results"}
                ]
            },
            {
                "name": "get_job_status",
                "path": "/api/v1/job/{job_id}",
                "method": "GET",
                "description": "Get the status of a submitted job",
                "parameters": [
                    {"name": "job_id", "type": "str", "ts_type": "string", "required": True, "description": "The job ID to check"}
                ]
            },
            {
                "name": "list_jobs",
                "path": "/api/v1/jobs",
                "method": "GET",
                "description": "List all jobs",
                "parameters": []
            }
        ],
        auth={"type": "bearer", "description": "JWT Bearer token"}
    )
