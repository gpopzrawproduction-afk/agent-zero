"""
Setup configuration for ZeroGravity

This module contains the setup configuration for the ZeroGravity
multi-agent AI platform, defining package metadata, dependencies,
and installation parameters.
"""
import os
from setuptools import setup, find_packages
from pathlib import Path


# Read the contents of README file
def read_long_description():
    """Read the long description from README.md"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "ZeroGravity - Multi-Agent AI Platform"


# Read the requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f.readlines():
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#") and not line.startswith("--"):
                    # Handle -r requirement files by reading them recursively
                    if line.startswith("-r "):
                        sub_req_file = Path(__file__).parent / line[3:].strip()
                        if sub_req_file.exists():
                            with open(sub_req_file, "r", encoding="utf-8") as sub_f:
                                for sub_line in sub_f.readlines():
                                    sub_line = sub_line.strip()
                                    if sub_line and not sub_line.startswith("#") and not sub_line.startswith("--"):
                                        requirements.append(sub_line)
                    else:
                        requirements.append(line)
        return requirements
    return []


# Read the development requirements
def read_dev_requirements():
    """Read development requirements from requirements.dev.txt"""
    dev_requirements_path = Path(__file__).parent / "requirements.dev.txt"
    if dev_requirements_path.exists():
        with open(dev_requirements_path, "r", encoding="utf-8") as f:
            dev_requirements = []
            for line in f.readlines():
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith("#") and not line.startswith("--"):
                    dev_requirements.append(line)
        return dev_requirements
    return []


# Get the version from the source code
def get_version():
    """Extract version from the package"""
    version_file = Path(__file__).parent / "zero_gravity_core" / "__init__.py"
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.startswith("__version__"):
                    # Extract version from line like "__version__ = "1.0.0""
                    return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"  # Default version if not found


# Get package information
package_name = "zerogravity-platform"
package_version = get_version()
package_description = "ZeroGravity - Production-Ready Multi-Agent AI Platform"
long_description = read_long_description()
requirements = read_requirements()
dev_requirements = read_dev_requirements()


setup(
    name=package_name,
    version=package_version,
    author="ZeroGravity Team",
    author_email="contact@zerogravity.ai",
    description=package_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zerogravity/zerogravity",
    project_urls={
        "Documentation": "https://zerogravity.ai/docs",
        "Source Code": "https://github.com/zerogravity/zerogravity",
        "Issue Tracker": "https://github.com/zerogravity/issues",
        "Changelog": "https://github.com/zerogravity/releases",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    package_data={
        "zero_gravity_core": [
            "prompts/*.system.md",
            "prompts/*.md",
            "*.md",
            "config/*.yaml",
            "config/*.yml",
            "deployment/*.py",
            "deployment/config/*.py",
            "api_gateway/templates/*.html",
            "api_gateway/static/*.css",
            "api_gateway/static/*.js",
        ],
        "": ["*.md", "LICENSE", "NOTICE", ".env.example"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Framework :: FastAPI",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "coverage>=7.0.0",
            "hypothesis>=6.0.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.5.0",
            "tiktoken>=0.5.0",
            "cohere>=5.0.0",
            "google-generativeai>=0.3.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "sentry-sdk>=1.0.0",
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
        "database": [
            "sqlalchemy>=2.0.0",
            "asyncpg>=0.28.0",
            "psycopg2-binary>=2.9.0",
            "aiosqlite>=0.19.0",
        ],
        "caching": [
            "aiocache>=0.12.0",
            "cachetools>=5.3.0",
            "diskcache>=5.6.0",
        ],
        "queue": [
            "celery>=5.3.0",
            "redis>=4.6.0",
            "rq>=1.15.0",
        ],
        "security": [
            "cryptography>=41.0.0",
            "pyjwt>=2.7.0",
            "bcrypt>=4.0.0",
            "passlib>=1.7.0",
            "secure>=0.3.0",
        ],
        "tools": [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "pyarrow>=13.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zerogravity=zero_gravity_core.cli:main",
            "zerogravity-api=zero_gravity_core.api_gateway.gateway:run_api_gateway",
            "zerogravity-worker=zero_gravity_core.task_queue.celery_app:run_worker",
            "zerogravity-demo=zero_gravity_core.demo:run_demo",
        ],
        "zerogravity.plugins": [
            # Plugin entry points would go here
            # "custom_agent = my_package.my_agent:MyCustomAgent",
            # "custom_tool = my_package.my_tool:MyCustomTool",
        ],
    },
    keywords=[
        "ai", "artificial-intelligence", "multi-agent", "llm", 
        "gpt", "chatgpt", "claude", "autonomous-agents",
        "workflow", "orchestration", "automation",
        "openai", "anthropic", "framework"
    ],
    license="MIT",
    license_files=["LICENSE"],
    zip_safe=False,
    platforms=["any"],
    # Optional: Enable type checking
    options={
        "bdist_wheel": {
            "universal": True,
        },
    },
)
