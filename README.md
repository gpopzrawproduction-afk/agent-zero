# ZeroGravity Multi-Agent AI Platform

ZeroGravity is a production-ready, multi-agent AI platform designed to decompose complex objectives into actionable workflows through coordinated agent collaboration.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Intelligent Agents**: Specialized agents (Architect, Engineer, Designer, Operator) handle different aspects of workflow execution
- **LLM Integration**: Supports multiple LLM providers (OpenAI, Anthropic, custom) with provider abstraction
- **Workflow Orchestration**: Parallel execution with dependency management and workflow graphs
- **API Gateway**: Production-ready API with rate limiting, authentication, and streaming
- **Security**: Comprehensive authentication, authorization, and input validation
- **Monitoring**: Prometheus-based metrics and observability
- **Extensibility**: Plugin architecture for custom functionality

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  Coordinator    │────│   Agents        │
│                 │    │                 │
│ • Rate Limiting │    │ • Workflow      │    │ • Architect     │
│ • Auth/Z        │    │ • Task Queue    │    │ • Engineer      │
│ • Streaming     │    │ • Dependency    │    │ • Designer      │
└─────────────────┘    │   Management    │    │ • Operator      │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │  LLM Manager    │────│   Tools         │
                       │                 │    │                 │
                       │ • Provider      │    │ • File Reader   │
                       │   Abstraction   │    │ • Command Exec  │
                       │ • Caching       │    │ • Web Scraper   │
                       │ • Streaming     │    │ • Calculator    │
                       └─────────────────┘    └─────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Redis (for caching and task queues)
- PostgreSQL or SQLite (for persistent storage)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-organization/zerogravity.git
cd zerogravity
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env to add your API keys and configuration
```

### Configuration

The platform can be configured through environment variables:

```bash
# API Configuration
ZEROGRAVITY_API_KEY=your_api_key
ZEROGRAVITY_JWT_SECRET=your_jwt_secret

# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
DATABASE_URL=postgresql://user:pass@localhost/db

# Redis
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## Usage

### Running Locally

1. Start the services with Docker Compose:
```bash
docker-compose up -d
```

2. The API will be available at `http://localhost:8000`

3. Submit a test objective:
```bash
curl -X POST http://localhost:8000/api/v1/objective \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "objective": "Build a simple web application",
    "priority": "normal"
  }'
```

### Running with Python (Development)

1. Set up your environment variables
2. Run the coordinator directly:
```bash
python -m uvicorn zero_gravity_core.api_gateway.gateway:run_api_gateway --reload
```

## Deployment

### Docker Compose Deployment

For staging and production deployments, use the provided `docker-compose.yml`:

1. Update the environment variables in `docker-compose.yml` with your production values
2. Deploy:
```bash
docker-compose up -d --scale api-gateway=3 --scale agent-coordinator=2
```

### Kubernetes Deployment

For Kubernetes deployments, the platform includes manifests generation:

```bash
# Generate Kubernetes manifests
python -c "from deployment.config.deployment_config import create_kubernetes_manifests, load_environment_config; config = load_environment_config(); manifests = create_kubernetes_manifests(config); print(manifests)"
```

### Production Configuration

For production environments, ensure you have:

- SSL certificates configured
- Proper API keys for LLM providers
- Scaled database and Redis instances
- Monitoring and alerting configured
- Backup and disaster recovery procedures

## Monitoring

The platform includes comprehensive monitoring capabilities:

- **Metrics**: Prometheus metrics available at `/metrics`
- **Tracing**: Distributed tracing with Jaeger
- **Logging**: Structured logging with log levels
- **Health Checks**: Health endpoints at `/health`

### Dashboard Access

- Grafana: `http://localhost:3000` (admin/admin)
- Prometheus: `http://localhost:9090`
- Jaeger UI: `http://localhost:16686`

## API Documentation

The API is documented using OpenAPI/Swagger. After starting the service, visit:

- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

### Available Endpoints

- `POST /api/v1/objective` - Submit a new objective
- `GET /api/v1/job/{job_id}` - Get job status
- `GET /api/v1/jobs` - List jobs
- `POST /api/v1/stream/workflow` - Stream workflow results
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Prometheus metrics

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest`
6. Commit your changes: `git commit -m 'Add my feature'`
7. Push to the branch: `git push origin feature/my-feature`
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 coding standards
- Write comprehensive tests for new features
- Document public APIs with docstrings
- Use type hints for all function parameters and return values

## Security

The platform includes several security measures:

- Input validation and sanitization
- JWT-based authentication
- Rate limiting
- SQL injection prevention
- XSS protection
- Secure session management

For security issues, please contact us directly rather than opening a public issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository or contact our team.

---

*ZeroGravity - Transforming complex objectives into intelligent, automated workflows*
