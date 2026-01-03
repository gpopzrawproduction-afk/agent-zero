#!/usr/bin/env python3
# zero_gravity_core/demo_research_agent.py

"""
Demo script for the Research Agent with AI Ops integration.
This demonstrates the Research Agent following the complete lifecycle:
Register → Request Approval → Execute → Emit Telemetry → Await Evaluation
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from zero_gravity_core.agents.researcher import ResearchAgent
from zero_gravity_core.agents.ai_ops_integration import initialize_ai_ops_agent, shutdown_ai_ops_agent


async def demo_research_agent():
    print("🚀 Starting Research Agent Demo with AI Ops Integration")
    print("=" * 60)
    
    # Initialize AI Ops agent
    print("🔧 Initializing AI Ops agent...")
    ai_ops = await initialize_ai_ops_agent()
    print("✅ AI Ops agent initialized")
    
    # Create Research Agent
    print("\n🤖 Creating Research Agent...")
    research_agent = ResearchAgent()
    print("✅ Research Agent created")
    
    # Define a research task
    research_task = "Analyze the current state of AI research in large language models, focusing on recent developments in 2023-2024, key players, and emerging trends."
    
    print(f"\n📋 Research Task: {research_task}")
    
    try:
        # Execute the research task (this follows the complete lifecycle internally)
        print("\n🔄 Executing research task (following AI Ops lifecycle)...")
        result = await research_agent.execute_research_task(research_task)
        
        print("\n✅ Research task completed successfully!")
        print(f"📊 Results:")
        print(f"   - Task Description: {result['task_description'][:50]}...")
        print(f"   - Source Count: {result['source_count']}")
        print(f"   - Confidence Score: {result['confidence_score']:.2f}")
        print(f"   - Execution Time: {result['execution_time']:.2f} seconds")
        print(f"   - Synthesis Summary: {result['synthesis']['summary']}")
        print(f"   - Key Points: {len(result['synthesis']['key_points'])} points identified")
        
        # Await evaluation from AI Ops
        print("\n🔍 Awaiting evaluation from AI Ops...")
        await research_agent.await_evaluation()
        print("✅ Evaluation completed")
        
    except Exception as e:
        print(f"\n❌ Error during research task: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Shutdown AI Ops agent
    print("\n🛑 Shutting down AI Ops agent...")
    await shutdown_ai_ops_agent()
    print("✅ AI Ops agent shutdown complete")
    
    print("\n🎉 Research Agent Demo Completed!")
    print("=" * 60)


async def demo_multiple_research_tasks():
    print("🚀 Starting Multiple Research Tasks Demo")
    print("=" * 60)
    
    # Initialize AI Ops agent
    print("🔧 Initializing AI Ops agent...")
    ai_ops = await initialize_ai_ops_agent()
    print("✅ AI Ops agent initialized")
    
    # Create Research Agent
    print("\n🤖 Creating Research Agent...")
    research_agent = ResearchAgent()
    print("✅ Research Agent created")
    
    # Define multiple research tasks
    research_tasks = [
        "Analyze the impact of renewable energy on global electricity markets in 2024",
        "Research the latest developments in quantum computing and their potential applications",
        "Study the trends in remote work adoption and their effects on urban development"
    ]
    
    for i, task in enumerate(research_tasks, 1):
        print(f"\n📋 Research Task {i}: {task}")
        
        try:
            # Execute the research task
            print(f"🔄 Executing research task {i}...")
            result = await research_agent.execute_research_task(task)
            
            print(f"✅ Task {i} completed successfully!")
            print(f"📊 Results:")
            print(f"   - Source Count: {result['source_count']}")
            print(f"   - Confidence Score: {result['confidence_score']:.2f}")
            print(f"   - Execution Time: {result['execution_time']:.2f} seconds")
            print(f"   - Summary: {result['synthesis']['summary']}")
            
            # Await evaluation from AI Ops
            print(f"🔍 Awaiting evaluation for task {i}...")
            await research_agent.await_evaluation()
            print(f"✅ Evaluation for task {i} completed")
            
        except Exception as e:
            print(f"\n❌ Error during task {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Shutdown AI Ops agent
    print("\n🛑 Shutting down AI Ops agent...")
    await shutdown_ai_ops_agent()
    print("✅ AI Ops agent shutdown complete")
    
    print("\n🎉 Multiple Research Tasks Demo Completed!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        asyncio.run(demo_multiple_research_tasks())
    else:
        asyncio.run(demo_research_agent())
