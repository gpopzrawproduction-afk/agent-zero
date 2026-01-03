"""
Production-Ready API Gateway for ZeroGravity

This module implements a production-ready API gateway that handles
routing, authentication, rate limiting, and other cross-cutting concerns
for the ZeroGravity platform.
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import logging
from pathlib import Path
import jwt
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis
import aioredis


class APIEndpoint(str, Enum):
    """Available API endpoints"""
    SUBMIT_OBJECTIVE = "/api/v1/objective"
    GET_JOB_STATUS = "/api/v1/job/{job_id}"
    LIST_JOBS = "/api/v1/jobs"
    STREAM_WORKFLOW = "/api/v1/stream/workflow"
    HEALTH_CHECK = "/api/v1/health"
    METRICS = "/api/v1/metrics"
    WEBHOOK_CALLBACK = "/api/v1/webhook/callback"


@dataclass
class APIConfig:
    """Configuration for the API gateway"""
    api_key: str
    jwt_secret: str
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    cors_origins: List[str] = None
    debug: bool = False
    redis_url: str = "redis://localhost:6379/0"
    timeout: int = 30
    max_concurrent_requests: int = 100
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


class APIRequest(BaseModel):
    """Model for API requests"""
    objective: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field(default="normal", regex=r"^(low|normal|high)$")
    callback_url: Optional[str] = None
    timeout: int = Field(default=300, ge=1, le=3600)  # 5 minutes to 1 hour
    stream_tokens: bool = False


class APIResponse(BaseModel):
    """Model for API responses"""
    job_id: str
    status: str
    created_at: str
    objective: str
    priority: str


class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, redis_url: str, requests: int, window: int):
        self.redis_url = redis_url
        self.requests = requests
        self.window = window
        self.redis = None
    
    async def init_redis(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed based on rate limits"""
        if not self.redis:
            await self.init_redis()
        
        key = f"rate_limit:{identifier}"
        current = await self.redis.get(key)
        
        if current is None:
            # First request, set counter
            await self.redis.setex(key, self.window, 1)
            return True
        
        count = int(current)
        if count >= self.requests:
            return False
        
        # Increment counter
        await self.redis.incr(key)
        return True
    
    async def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for an identifier"""
        if not self.redis:
            await self.init_redis()
        
        key = f"rate_limit:{identifier}"
        current = await self.redis.get(key)
        
        if current is None:
            return self.requests
        
        count = int(current)
        return max(0, self.requests - count)


class JWTAuth:
    """JWT authentication handler"""
    
    def __init__(self, secret: str):
        self.secret = secret
    
    def create_token(self, user_id: str, expires_delta: timedelta = None) -> str:
        """Create a JWT token"""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
        
        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "sub": user_id,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp()
        }
        encoded_jwt = jwt.encode(to_encode, self.secret, algorithm="HS256")
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify a JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")


class RequestLogger:
    """Log API requests"""
    
    def __init__(self):
        self.logger = logging.getLogger("APIGateway.RequestLogger")
    
    async def log_request(self, request: Request, response: Response, 
                         execution_time: float, user_id: Optional[str] = None):
        """Log API request details"""
        self.logger.info(
            f"API Request: {request.method} {request.url.path} "
            f"User: {user_id or 'anonymous'} "
            f"IP: {self.get_client_ip(request)} "
            f"Status: {response.status_code} "
            f"Time: {execution_time:.3f}s"
        )
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host


class APIGateway:
    """Main API Gateway class"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.app = FastAPI(
            title="ZeroGravity API Gateway",
            description="Production-ready API gateway for ZeroGravity platform",
            version="1.0.0"
        )
        self.rate_limiter = RateLimiter(
            config.redis_url, 
            config.rate_limit_requests, 
            config.rate_limit_window
        )
        self.jwt_auth = JWTAuth(config.jwt_secret)
        self.request_logger = RequestLogger()
        self.security = HTTPBearer()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Setup middleware
        self.setup_middleware()
        
        # Setup routes
        self.setup_routes()
    
    def setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # Expose custom headers
            expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
        )
        
        # Custom middleware for rate limiting and logging
        self.app.add_middleware(APIGatewayMiddleware, gateway=self)
    
    def setup_routes(self):
        """Setup API routes"""
        @self.app.get(APIEndpoint.HEALTH_CHECK.value)
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.post(APIEndpoint.SUBMIT_OBJECTIVE.value, response_model=APIResponse)
        async def submit_objective(request: APIRequest, 
                                 credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Submit a new objective for processing"""
            # Verify token
            token_payload = self.jwt_auth.verify_token(credentials.credentials)
            user_id = token_payload.get("sub")
            
            # Create job ID
            job_id = str(uuid.uuid4())
            
            # Store job details
            self.active_jobs[job_id] = {
                "job_id": job_id,
                "objective": request.objective,
                "priority": request.priority,
                "callback_url": request.callback_url,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "stream_tokens": request.stream_tokens
            }
            
            # In a real implementation, this would trigger background processing
            # For now, we'll simulate immediate processing
            await self.process_job(job_id)
            
            return APIResponse(
                job_id=job_id,
                status="queued",
                created_at=self.active_jobs[job_id]["created_at"],
                objective=request.objective,
                priority=request.priority
            )
        
        @self.app.get(APIEndpoint.GET_JOB_STATUS.value)
        async def get_job_status(job_id: str,
                               credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Get the status of a submitted job"""
            token_payload = self.jwt_auth.verify_token(credentials.credentials)
            user_id = token_payload.get("sub")
            
            if job_id not in self.active_jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            
            job = self.active_jobs[job_id]
            
            # Verify user owns the job
            if job["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "result": job.get("result"),
                "error": job.get("error")
            }
        
        @self.app.get(APIEndpoint.LIST_JOBS.value)
        async def list_jobs(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """List jobs for the authenticated user"""
            token_payload = self.jwt_auth.verify_token(credentials.credentials)
            user_id = token_payload.get("sub")
            
            user_jobs = [
                job for job in self.active_jobs.values() 
                if job["user_id"] == user_id
            ]
            
            return {
                "jobs": user_jobs,
                "count": len(user_jobs)
            }
        
        @self.app.post(APIEndpoint.STREAM_WORKFLOW.value)
        async def stream_workflow(request: APIRequest,
                                credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            """Stream workflow results"""
            token_payload = self.jwt_auth.verify_token(credentials.credentials)
            user_id = token_payload.get("sub")
            
            # Create job ID
            job_id = str(uuid.uuid4())
            
            # Store job details
            self.active_jobs[job_id] = {
                "job_id": job_id,
                "objective": request.objective,
                "priority": request.priority,
                "callback_url": request.callback_url,
                "status": "processing",
                "created_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "stream_tokens": True
            }
            
            # Create streaming response
            async def event_generator():
                # Simulate streaming response
                yield f"data: {json.dumps({'event': 'started', 'job_id': job_id})}\n\n"
                
                # Simulate processing steps
                steps = ["architect", "engineer", "designer", "operator"]
                for i, step in enumerate(steps):
                    yield f"data: {json.dumps({'event': 'step', 'step': step, 'progress': (i+1)/len(steps)})}\n\n"
                    await asyncio.sleep(0.5)  # Simulate processing time
                
                # Complete
                result = f"Workflow completed for objective: {request.objective}"
                yield f"data: {json.dumps({'event': 'completed', 'result': result})}\n\n"
            
            return StreamingResponse(event_generator(), media_type="text/event-stream")
    
    async def process_job(self, job_id: str):
        """Process a job in the background"""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]
        job["status"] = "processing"
        
        try:
            # In a real implementation, this would call the actual ZeroGravity workflow
            # For now, we'll simulate the processing
            await asyncio.sleep(2)  # Simulate processing time
            
            # Update job status
            job["status"] = "completed"
            job["result"] = f"Processed: {job['objective']}"
            job["completed_at"] = datetime.utcnow().isoformat()
            
            # If callback URL is provided, send notification
            if job.get("callback_url"):
                await self.send_callback(job)
                
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.utcnow().isoformat()
    
    async def send_callback(self, job: Dict[str, Any]):
        """Send callback notification for job completion"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    job["callback_url"],
                    json={
                        "job_id": job["job_id"],
                        "status": job["status"],
                        "result": job.get("result"),
                        "error": job.get("error")
                    },
                    timeout=10
                )
        except Exception as e:
            logging.error(f"Failed to send callback for job {job['job_id']}: {e}")


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """Custom middleware for the API gateway"""
    
    def __init__(self, app, gateway: APIGateway):
        super().__init__(app)
        self.gateway = gateway
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get client IP for rate limiting
        client_ip = self.get_client_ip(request)
        
        # Check rate limit
        is_allowed = await self.gateway.rate_limiter.is_allowed(client_ip)
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={
                    "X-RateLimit-Limit": str(self.gateway.config.rate_limit_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.gateway.config.rate_limit_window)
                }
            )
        
        # Add rate limit headers
        remaining = await self.gateway.rate_limiter.get_remaining_requests(client_ip)
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(self.gateway.config.rate_limit_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.gateway.config.rate_limit_window)
        
        # Log the request
        execution_time = time.time() - start_time
        await self.gateway.request_logger.log_request(request, response, execution_time)
        
        return response
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host


class APIClient:
    """Client for interacting with the ZeroGravity API"""
    
    def __init__(self, base_url: str, api_key: str = None, jwt_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.headers = {"Content-Type": "application/json"}
        
        if self.jwt_token:
            self.headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self.api_key:
            self.headers["X-API-Key"] = self.api_key
    
    async def submit_objective(self, objective: str, priority: str = "normal", 
                             callback_url: str = None) -> Dict[str, Any]:
        """Submit an objective to the API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/objective",
                json={
                    "objective": objective,
                    "priority": priority,
                    "callback_url": callback_url
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a job"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/job/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
    
    async def stream_workflow(self, objective: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream workflow results"""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/v1/stream/workflow",
                json={"objective": objective},
                headers=self.headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data.strip():
                            yield json.loads(data)


# Global API gateway instance
api_gateway: Optional[APIGateway] = None


def init_api_gateway(config: APIConfig) -> APIGateway:
    """Initialize the API gateway"""
    global api_gateway
    api_gateway = APIGateway(config)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not config.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    return api_gateway


def get_api_gateway() -> Optional[APIGateway]:
    """Get the global API gateway instance"""
    return api_gateway


def create_default_config() -> APIConfig:
    """Create a default configuration for the API gateway"""
    import os
    return APIConfig(
        api_key=os.getenv("ZEROGRAVITY_API_KEY", "default_api_key"),
        jwt_secret=os.getenv("ZEROGRAVITY_JWT_SECRET", "default_jwt_secret"),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        debug=os.getenv("DEBUG", "False").lower() == "true",
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        timeout=int(os.getenv("API_TIMEOUT", "30"))
    )


# Example usage
async def run_api_gateway():
    """Run the API gateway"""
    import uvicorn
    
    # Create default config
    config = create_default_config()
    
    # Initialize gateway
    gateway = init_api_gateway(config)
    
    # Run the server
    uvicorn_config = uvicorn.Config(
        gateway.app,
        host="0.0.0.0",
        port=8000,
        log_level="info" if not config.debug else "debug"
    )
    server = uvicorn.Server(uvicorn_config)
    
    print("ZeroGravity API Gateway starting...")
    print(f"API endpoints available at: http://localhost:8000")
    print(f"Health check: http://localhost:8000/api/v1/health")
    print(f"Rate limit: {config.rate_limit_requests} requests per {config.rate_limit_window} seconds")
    
    await server.serve()


if __name__ == "__main__":
    # For testing purposes
    print("Starting ZeroGravity API Gateway...")
    # asyncio.run(run_api_gateway())
