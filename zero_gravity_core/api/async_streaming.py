"""
Async Streaming API for ZeroGravity

This module implements an asynchronous streaming API that allows
for real-time communication with the ZeroGravity platform,
including streaming responses from agents and workflows.
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from uuid import uuid4


class StreamEventType(Enum):
    """Types of events in the streaming API"""
    CONNECTION_ESTABLISHED = "connection_established"
    WORKFLOW_STARTED = "workflow_started"
    AGENT_STARTED = "agent_started"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETED = "agent_completed"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_ERROR = "workflow_error"
    TOKEN_STREAM = "token_stream"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """Represents an event in the streaming API"""
    event_type: StreamEventType
    data: Dict[str, Any]
    timestamp: datetime = None
    event_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.event_id is None:
            self.event_id = str(uuid4())
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        event_dict = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
        return json.dumps(event_dict)


class StreamingRequest(BaseModel):
    """Request model for streaming API"""
    objective: str
    stream_tokens: bool = False
    include_progress: bool = True
    priority: str = "normal"
    timeout: int = 300  # 5 minutes default timeout


class StreamingAgent:
    """Streaming agent that can emit progress events"""
    
    def __init__(self, role: str, coordinator):
        self.role = role
        self.coordinator = coordinator
        self.logger = logging.getLogger(f"StreamingAgent.{role}")
    
    async def execute_with_streaming(self, input_data: Any, 
                                   websocket: WebSocket = None,
                                   stream_tokens: bool = False) -> Any:
        """Execute agent with optional streaming"""
        # Send agent started event
        if websocket:
            await self._send_event(
                websocket, 
                StreamEventType.AGENT_STARTED, 
                {"role": self.role, "input_summary": str(input_data)[:100]}
            )
        
        try:
            # Execute with LLM and stream if requested
            if stream_tokens:
                result = await self._execute_with_token_streaming(input_data, websocket)
            else:
                result = await self._execute_normally(input_data)
            
            # Send completion event
            if websocket:
                await self._send_event(
                    websocket,
                    StreamEventType.AGENT_COMPLETED,
                    {"role": self.role, "result_summary": str(result)[:200]}
                )
            
            return result
            
        except Exception as e:
            # Send error event
            if websocket:
                await self._send_event(
                    websocket,
                    StreamEventType.WORKFLOW_ERROR,
                    {"role": self.role, "error": str(e)}
                )
            raise
    
    async def _execute_normally(self, input_data: Any) -> Any:
        """Execute agent normally without streaming"""
        # Get the agent from coordinator
        agent = self.coordinator.spawn_agent(self.role)
        return agent.execute_with_llm(input_data)
    
    async def _execute_with_token_streaming(self, input_data: Any, 
                                          websocket: WebSocket) -> str:
        """Execute agent with token streaming"""
        # Get the agent from coordinator
        agent = self.coordinator.spawn_agent(self.role)
        
        # Use the agent's streaming method if available
        if hasattr(agent, 'execute_with_llm_streaming'):
            result_parts = []
            async for token in agent.execute_with_llm_streaming(input_data):
                result_parts.append(token)
                
                # Send token to client
                if websocket:
                    await self._send_event(
                        websocket,
                        StreamEventType.TOKEN_STREAM,
                        {"role": self.role, "token": token}
                    )
                
                # Yield control to event loop
                await asyncio.sleep(0)
            
            return "".join(result_parts)
        else:
            # Fallback to normal execution
            result = agent.execute_with_llm(input_data)
            
            # Send as a single token event
            if websocket:
                await self._send_event(
                    websocket,
                    StreamEventType.TOKEN_STREAM,
                    {"role": self.role, "token": result}
                )
            
            return result
    
    async def _send_event(self, websocket: WebSocket, 
                         event_type: StreamEventType, data: Dict[str, Any]):
        """Send an event to the websocket"""
        try:
            event = StreamEvent(event_type, data)
            await websocket.send_text(event.to_json())
        except Exception as e:
            self.logger.error(f"Failed to send event to websocket: {e}")


class StreamingWorkflow:
    """Streaming workflow that can emit progress events"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger("StreamingWorkflow")
    
    async def execute_with_streaming(self, objective: str, 
                                   websocket: WebSocket = None,
                                   stream_tokens: bool = False,
                                   include_progress: bool = True) -> Dict[str, Any]:
        """Execute workflow with optional streaming and progress"""
        # Send workflow started event
        if websocket:
            await self._send_event(
                websocket,
                StreamEventType.WORKFLOW_STARTED,
                {"objective": objective}
            )
        
        try:
            # Initialize workflow components
            result = {
                "objective": objective,
                "result": None,
                "history": [],
                "execution_summary": {}
            }
            
            # Execute each phase with progress updates
            phases = ["architect", "engineer", "designer", "operator"]
            
            for i, role in enumerate(phases):
                if websocket and include_progress:
                    await self._send_event(
                        websocket,
                        StreamEventType.WORKFLOW_PROGRESS,
                        {
                            "current_phase": role,
                            "phase_number": i + 1,
                            "total_phases": len(phases),
                            "progress": (i + 1) / len(phases)
                        }
                    )
                
                # Create streaming agent for this phase
                agent = StreamingAgent(role, self.coordinator)
                
                # Execute with streaming
                phase_result = await agent.execute_with_streaming(
                    objective if i == 0 else result["result"],
                    websocket,
                    stream_tokens
                )
                
                # Update result
                result["result"] = phase_result
                result["history"].append({
                    "role": role,
                    "output": phase_result,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Send completion event
            if websocket:
                await self._send_event(
                    websocket,
                    StreamEventType.WORKFLOW_COMPLETED,
                    {
                        "result_summary": str(result["result"])[:500],
                        "total_phases": len(phases)
                    }
                )
            
            return result
            
        except Exception as e:
            # Send error event
            if websocket:
                await self._send_event(
                    websocket,
                    StreamEventType.WORKFLOW_ERROR,
                    {"error": str(e), "objective": objective}
                )
            raise
    
    async def _send_event(self, websocket: WebSocket, 
                         event_type: StreamEventType, data: Dict[str, Any]):
        """Send an event to the websocket"""
        try:
            event = StreamEvent(event_type, data)
            await websocket.send_text(event.to_json())
        except Exception as e:
            self.logger.error(f"Failed to send event to websocket: {e}")


class StreamingAPI:
    """Main streaming API class"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.logger = logging.getLogger("StreamingAPI")
        self.active_connections: Dict[str, WebSocket] = {}
        self.streaming_workflow = StreamingWorkflow(coordinator)
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket endpoint for streaming"""
        await websocket.accept()
        
        # Generate connection ID
        connection_id = str(uuid4())
        self.active_connections[connection_id] = websocket
        
        try:
            # Send connection established event
            await self._send_event(
                websocket,
                StreamEventType.CONNECTION_ESTABLISHED,
                {"connection_id": connection_id}
            )
            
            # Listen for messages
            while True:
                try:
                    data = await websocket.receive_text()
                    request_data = json.loads(data)
                    
                    # Process request
                    await self._handle_websocket_request(websocket, request_data)
                    
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    await self._send_error(websocket, f"Error processing request: {str(e)}")
                    break
        
        except WebSocketDisconnect:
            pass
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def _handle_websocket_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Handle incoming websocket request"""
        request_type = request_data.get("type", "")
        
        if request_type == "execute_workflow":
            objective = request_data.get("objective", "")
            stream_tokens = request_data.get("stream_tokens", False)
            include_progress = request_data.get("include_progress", True)
            
            if not objective:
                await self._send_error(websocket, "Missing objective")
                return
            
            # Execute workflow with streaming
            await self.streaming_workflow.execute_with_streaming(
                objective=objective,
                websocket=websocket,
                stream_tokens=stream_tokens,
                include_progress=include_progress
            )
        elif request_type == "ping":
            # Send heartbeat response
            await self._send_event(
                websocket,
                StreamEventType.HEARTBEAT,
                {"timestamp": datetime.utcnow().isoformat()}
            )
        else:
            await self._send_error(websocket, f"Unknown request type: {request_type}")
    
    async def _send_event(self, websocket: WebSocket, 
                         event_type: StreamEventType, data: Dict[str, Any]):
        """Send an event to the websocket"""
        try:
            event = StreamEvent(event_type, data)
            await websocket.send_text(event.to_json())
        except Exception as e:
            self.logger.error(f"Failed to send event to websocket: {e}")
    
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send an error message to the websocket"""
        try:
            await websocket.send_text(json.dumps({
                "event_type": "error",
                "data": {"error": error_message},
                "timestamp": datetime.utcnow().isoformat()
            }))
        except Exception as e:
            self.logger.error(f"Failed to send error to websocket: {e}")
    
    def create_fastapi_app(self) -> FastAPI:
        """Create a FastAPI app with streaming endpoints"""
        app = FastAPI(title="ZeroGravity Streaming API", version="1.0.0")
        
        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            await self.websocket_endpoint(websocket)
        
        @app.post("/api/v1/stream/workflow")
        async def stream_workflow(request: StreamingRequest):
            """HTTP endpoint that streams workflow results"""
            async def event_generator():
                # Create a mock websocket-like interface for the generator
                queue = asyncio.Queue()
                
                # Run workflow in background
                async def run_workflow():
                    try:
                        # This is a simplified version - in reality, you'd need to adapt
                        # the streaming workflow to work with HTTP streaming
                        result = await self.streaming_workflow.execute_with_streaming(
                            objective=request.objective,
                            stream_tokens=request.stream_tokens,
                            include_progress=request.include_progress
                        )
                        await queue.put({"type": "completion", "data": result})
                    except Exception as e:
                        await queue.put({"type": "error", "data": str(e)})
                
                # Start workflow execution
                asyncio.create_task(run_workflow())
                
                # Yield events as they come
                while True:
                    try:
                        # Set timeout for queue get
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                        
                        if event["type"] == "completion":
                            yield f"data: {json.dumps({'event_type': 'workflow_completed', 'data': event['data']})}\n\n"
                            break
                        elif event["type"] == "error":
                            yield f"data: {json.dumps({'event_type': 'workflow_error', 'data': {'error': event['data']}})}\n\n"
                            break
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        yield f"data: {json.dumps({'event_type': 'heartbeat', 'data': {}})}\n\n"
            
            return StreamingResponse(event_generator(), media_type="text/plain")
        
        @app.get("/api/v1/stream/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "active_connections": len(self.active_connections),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return app


# Example usage and initialization
async def create_streaming_api_example():
    """Create an example streaming API instance"""
    from zero_gravity_core.agents.coordinator import Coordinator
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Create streaming API
    streaming_api = StreamingAPI(coordinator)
    
    # Create FastAPI app
    app = streaming_api.create_fastapi_app()
    
    return app, streaming_api


# Standalone server example
async def run_streaming_server():
    """Run the streaming server (for testing purposes)"""
    import uvicorn
    
    app, streaming_api = await create_streaming_api_example()
    
    # Print available endpoints
    print("ZeroGravity Streaming API started")
    print("WebSocket endpoint: ws://localhost:8000/ws/stream")
    print("HTTP streaming endpoint: POST http://localhost:8000/api/v1/stream/workflow")
    print("Health check: GET http://localhost:8000/api/v1/stream/health")
    
    # Run the server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # For testing purposes, run the streaming server
    print("Starting ZeroGravity Streaming API example...")
    # Uncomment the next line to run the server
    # asyncio.run(run_streaming_server())
