# Research Agent Design (AI Ops–Native)

This document outlines the design for a Research Agent that is native to the AI Ops system from day one, following all governance and lifecycle requirements.

## Agent Overview

### Purpose
The Research Agent is designed to:
- Conduct comprehensive research on topics requested by other agents or users
- Gather information from multiple sources
- Synthesize findings into actionable insights
- Feed research results to other agents in the system
- Operate under strict AI Ops governance

### Position in Architecture
- First new agent designed with AI Ops integration from the ground up
- Serves as a test subject for AI Ops optimization
- Provides high-value information to other agents
- Low operational risk with high signal generation

## Agent Lifecycle Implementation

The Research Agent will strictly follow the mandatory lifecycle:
1. **Register** → Register with AI Ops before any execution
2. **Request Approval** → Request approval for research tasks
3. **Execute** → Perform research with AI Ops oversight
4. **Emit Telemetry** → Emit comprehensive telemetry data
5. **Await Evaluation** → Await evaluation from AI Ops

## Technical Design

### Base Class Inheritance
```python
# zero_gravity_core/agents/researcher.py

from zero_gravity_core.agents.base import BaseAgent
from typing import Dict, Any, List
import requests
import time

class ResearchAgent(BaseAgent):
    def __init__(self, name="ResearchAgent"):
        super().__init__(name)
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
    
    def _execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute the research task"""
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
            
            return result
        except Exception as e:
            # Track error for telemetry
            error_result = {
                'task_description': task_description,
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }
            self.cost = self._calculate_cost(error_result, is_error=True)
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
```

### Research Agent Prompts
```markdown
# zero_gravity_core/prompts/researcher.system.md

# Research Agent System Prompt

## Role
You are a Research Agent in the ZeroGravity AI Ops system. Your primary function is to conduct comprehensive research on topics requested by users or other agents, synthesizing findings into actionable insights.

## Core Principles
1. Follow the AI Ops lifecycle strictly: Register → Request Approval → Execute → Emit Telemetry → Await Evaluation
2. Maintain the highest standards of research accuracy and relevance
3. Prioritize information from reliable and authoritative sources
4. Synthesize information in a clear, concise, and actionable manner

## Capabilities
- Web research across multiple domains
- Academic literature search
- Market research and analysis
- Technical documentation review
- Historical data compilation

## Research Process
1. Parse the research request to identify key parameters
2. Search across multiple sources for relevant information
3. Evaluate source credibility and relevance
4. Synthesize findings into coherent results
5. Present information in requested format

## Output Formats
- Summary: Brief overview of key findings
- Detailed Report: Comprehensive analysis with source attribution
- Key Points: Bullet points of most important findings
- Source List: Curated list of relevant sources

## Constraints
- Do not fabricate information
- Cite sources when possible
- Acknowledge limitations in available information
- Maintain objectivity in analysis
- Respect copyright and intellectual property

## AI Ops Integration
- All research requests must be approved by AI Ops
- Emit comprehensive telemetry after each research task
- Await evaluation from AI Ops before proceeding with next task
- Report any issues or anomalies to AI Ops immediately
```

## AI Ops Optimization Opportunities

### Research Depth vs Cost Optimization
- AI Ops can analyze the relationship between research depth and cost
- Optimize number of sources based on required confidence level
- Adjust search parameters based on historical effectiveness

### Model Routing
- AI Ops can route research tasks to different models based on:
  - Topic complexity
  - Required accuracy
  - Available budget
  - Time constraints

### Result Scoring
- AI Ops can score research results based on:
  - Source quality
  - Information relevance
  - User satisfaction
  - Downstream agent utilization

## Implementation Plan

### Phase 1: Basic Implementation
1. Create ResearchAgent class inheriting from BaseAgent
2. Implement core research functionality
3. Ensure compliance with agent lifecycle
4. Add basic telemetry emission

### Phase 2: Advanced Research Capabilities
1. Integrate with real search APIs
2. Add source evaluation and credibility assessment
3. Implement advanced synthesis algorithms
4. Add support for document analysis

### Phase 3: AI Ops Optimization Integration
1. Enable AI Ops to modify research parameters
2. Implement feedback loops for quality improvement
3. Add cost optimization features
4. Integrate with other agents in the system

## Expected Benefits

### For AI Ops
- First test subject for optimization algorithms
- Clear metrics for research quality vs cost
- Opportunity to validate AI Ops decision-making

### For the System
- Enhanced information gathering capabilities
- Reliable source of research for other agents
- Foundation for knowledge-based decision making

### For Users
- Access to comprehensive research capabilities
- Consistent quality and reliability
- Transparent cost and time estimates

## Success Metrics

- Research accuracy and relevance scores
- Cost per research task
- Time to complete research tasks
- User satisfaction ratings
- Downstream agent utilization of research results
- AI Ops optimization effectiveness
