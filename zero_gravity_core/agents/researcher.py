# zero_gravity_core/agents/researcher.py

from zero_gravity_core.agents.base import BaseAgent
from zero_gravity_core.agents.ai_ops_integration import (
    request_task_approval,
    monitor_task_execution,
    report_task_metrics,
    enforce_policy
)
from typing import Dict, Any, List
import requests
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    def __init__(self, name="ResearchAgent"):
        super().__init__(name=name, role="ResearchAgent")
        self.search_engines = [
            "google",  # Placeholder - would use actual search API
            "wikipedia",
            "arxiv",
            "pubmed"
        ]
        self.max_sources = 10
        self.start_time = None
        self.resources_used = {'cpu_percent': 0, 'memory_mb': 0}
        self.cost = 0.0
        self.task_id = None

    def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities for registration"""
        return {
            'research_domains': [
                'general_knowledge',
                'scientific_literature',
                'market_research',
                'technical_documentation',
                'historical_data'
            ],
            'research_methods': [
                'web_search',
                'database_query',
                'document_analysis',
                'data_synthesis'
            ],
            'output_formats': [
                'summary',
                'detailed_report',
                'key_points',
                'source_list'
            ]
        }

    def get_resource_requirements(self) -> Dict[str, Any]:
        """Return resource requirements for registration"""
        return {
            'cpu': 2,
            'memory': '1GB',
            'network': True,
            'gpu': False,
            'estimated_runtime_minutes': 10
        }

    def estimate_resources(self, task_description: str) -> Dict[str, Any]:
        """Estimate resources needed for specific task"""
        word_count = len(task_description.split())
        estimated_time = min(300, max(60, word_count * 2))  # 1-5 minutes based on complexity

        return {
            'cpu_percent': 10,
            'memory_mb': 256,
            'network_io_mb': 50,
            'estimated_time_seconds': estimated_time
        }

    def estimate_execution_time(self, task_description: str) -> int:
        """Estimate execution time for task"""
        word_count = len(task_description.split())
        return min(600, max(60, word_count * 3))  # 1-10 minutes based on complexity

    async def execute_research_task(self, task_description: str) -> Dict[str, Any]:
        """
        Main method to execute a research task following the AI Ops lifecycle:
        Register → Request Approval → Execute → Emit Telemetry → Await Evaluation
        """
        # 1. REGISTER with AI Ops
        logger.info(f"ResearchAgent registering with AI Ops for task: {task_description}")
        
        # 2. REQUEST APPROVAL from AI Ops
        logger.info(f"ResearchAgent requesting approval for task: {task_description}")
        approval = await request_task_approval(
            agent_name=self.role,
            task_description=task_description,
            priority=5,
            estimated_cost=0.15,
            estimated_time=self.estimate_execution_time(task_description),
            complexity="medium"
        )

        if not approval.approved:
            raise Exception(f"Task not approved by AI Ops: {approval}")

        logger.info(f"Task approved. Model selection: {approval.model_selection}")
        
        # 3. EXECUTE the research task
        self.start_time = time.time()
        self.resources_used = {'cpu_percent': 0, 'memory_mb': 0}
        
        try:
            # Parse the research request
            research_request = self._parse_research_request(task_description)
            
            # Conduct research
            findings = self._conduct_research(research_request)
            
            # Synthesize results
            synthesis = self._synthesize_findings(findings)
            
            # Format output
            result = {
                'task_description': task_description,
                'research_request': research_request,
                'findings': findings,
                'synthesis': synthesis,
                'source_count': len(findings),
                'confidence_score': synthesis.get('confidence', 0.8),
                'execution_time': time.time() - self.start_time
            }
            
            # Calculate cost
            self.cost = self._calculate_cost(result)
            
            # 4. EMIT TELEMETRY
            await self._emit_telemetry(result, approval)
            
            return result
        except Exception as e:
            # Track error for telemetry
            error_result = {
                'task_description': task_description,
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }
            self.cost = self._calculate_cost(error_result, is_error=True)
            
            # Emit error telemetry
            await self._emit_telemetry(error_result, approval, success=False)
            raise e

    def _parse_research_request(self, task_description: str) -> Dict[str, Any]:
        """Parse the research request to identify key parameters"""
        # Simple parsing - would use NLP in production
        return {
            'topic': task_description,
            'depth': 'comprehensive',  # Default
            'sources_required': 5,     # Default
            'date_range': 'any',       # Default
            'source_types': ['web', 'academic', 'news']  # Default
        }

    def _conduct_research(self, research_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Conduct research using various sources"""
        findings = []
        
        # For each search engine, perform search
        for engine in self.search_engines[:research_request['sources_required']]:
            try:
                sources = self._search_with_engine(engine, research_request['topic'])
                findings.extend(sources)
                
                # Limit to prevent excessive resource usage
                if len(findings) >= self.max_sources:
                    break
            except Exception as e:
                print(f"Error searching with {engine}: {str(e)}")
                continue
        
        return findings

    def _search_with_engine(self, engine: str, topic: str) -> List[Dict[str, Any]]:
        """Search with a specific engine"""
        # Placeholder implementation - would use actual search APIs
        # This would be replaced with real search implementations
        return [{
            'source': engine,
            'topic': topic,
            'url': f'https://{engine}.com/search?q={topic}',
            'summary': f'Result from {engine} for {topic}',
            'relevance_score': 0.8,
            'timestamp': time.time()
        }]

    def _synthesize_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize research findings into coherent results"""
        if not findings:
            return {
                'summary': 'No sources found for the research topic',
                'key_points': [],
                'confidence': 0.0
            }
        
        # Simple synthesis - would use advanced NLP in production
        summary = f"Found {len(findings)} sources for research topic"
        key_points = [f"Source from {f['source']}: {f['summary'][:50]}..." for f in findings[:3]]
        
        return {
            'summary': summary,
            'key_points': key_points,
            'confidence': min(0.95, 0.5 + (len(findings) * 0.1)),  # Confidence based on source count
            'total_sources': len(findings)
        }

    def get_execution_time(self) -> float:
        """Get execution time for telemetry"""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage for telemetry"""
        # In a real implementation, this would track actual resource usage
        return {
            'cpu_percent': 15,  # Placeholder
            'memory_mb': 128,   # Placeholder
            'network_io_mb': 25 # Placeholder
        }

    def calculate_cost(self) -> float:
        """Calculate cost for telemetry"""
        return self.cost

    def _calculate_cost(self, result: Dict[str, Any], is_error: bool = False) -> float:
        """Calculate cost based on execution parameters"""
        if is_error:
            # Minimal cost for errors
            return 0.001
        
        # Base cost
        cost = 0.01
        
        # Add cost based on number of sources searched
        if 'findings' in result:
            cost += len(result['findings']) * 0.002
        
        # Add cost based on execution time
        if 'execution_time' in result:
            cost += (result['execution_time'] / 60) * 0.005  # Per minute
        
        return cost

    async def _emit_telemetry(self, result: Dict[str, Any], approval, success=True):
        """Emit telemetry data to AI Ops"""
        execution_time = result.get('execution_time', time.time() - self.start_time)
        latency_ms = execution_time * 1000
        cost_usd = self.cost
        
        # Determine quality score based on result
        quality_score = result.get('confidence_score', 0.8) if success else 0.1
        
        # Report metrics to AI Ops
        await report_task_metrics(
            agent_name=self.role,
            task_id=self.task_id or f"research_{int(time.time())}",
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            success=success,
            quality_score=quality_score,
            model_used=approval.model_selection,
            error=result.get('error') if not success else None
        )

        logger.info(f"Telemetry emitted for research task. Cost: ${cost_usd}, Quality: {quality_score}")

    async def await_evaluation(self):
        """Await evaluation from AI Ops (placeholder implementation)"""
        # In a full implementation, this would wait for explicit evaluation from AI Ops
        # For now, we'll just log that the agent is awaiting evaluation
        logger.info(f"ResearchAgent awaiting evaluation from AI Ops")
        await asyncio.sleep(0.1)  # Simulate awaiting evaluation
        logger.info(f"ResearchAgent evaluation completed")
