# zero_gravity_core/demo.py

"""
ZeroGravity Multi-Agent Platform - Client Demo

This demo showcases the capabilities of the ZeroGravity platform
with a user-friendly interface for demonstrating the multi-agent workflow.
"""

from pathlib import Path
from zero_gravity_core.agents.coordinator import Coordinator
import json


def run_demo():
    """Run the ZeroGravity platform demo"""
    print("=" * 60)
    print("           ZEROGRAVITY MULTI-AGENT PLATFORM DEMO")
    print("=" * 60)
    print()
    print("Welcome to the ZeroGravity multi-agent platform!")
    print("This demo showcases our AI agents working together to")
    print("accomplish complex objectives through coordinated effort.")
    print()
    
    # Initialize the Coordinator
    base_dir = Path(__file__).resolve().parent
    coordinator = Coordinator(base_dir=base_dir)

    # Example objectives to demonstrate the platform
    demo_objectives = [
        "Build a prototype multi-agent AI platform",
        "Create a web application for task management",
        "Design a data analysis pipeline for customer insights"
    ]

    print("DEMO OBJECTIVES:")
    for i, objective in enumerate(demo_objectives, 1):
        print(f"  {i}. {objective}")
    print()

    # Process each demo objective
    for i, objective in enumerate(demo_objectives, 1):
        print(f"{'='*20} DEMO {i}: {objective} {'='*20}")
        print()
        
        # Run the full workflow
        results = coordinator.run(objective)

        # Display formatted results
        print("🎯 OBJECTIVE:")
        print(f"   {results['objective']}")
        print()

        print("📋 WORKFLOW EXECUTION:")
        for entry in results["history"]:
            role = entry["role"].upper()
            if role != "ERROR":  # Skip error entries in demo display
                print(f"   → {role} Agent processed the task")
        print()

        print("📊 EXECUTION SUMMARY:")
        summary = results.get("execution_summary", {})
        print(f"   Status: {summary.get('workflow_status', 'Unknown')}")
        print(f"   Details: {summary.get('execution_summary', 'No details available')}")
        print()

        # Show a sample of the architect's plan
        for entry in results["history"]:
            if entry["role"] == "architect":
                architect_output = entry["output"]
                print("📋 ARCHITECT'S PLAN:")
                print(f"   Estimated Complexity: {architect_output.get('estimated_complexity', 'Unknown')}")
                print(f"   Required Resources: {', '.join(architect_output.get('required_resources', []))}")
                print("   Key Steps:")
                for j, step in enumerate(architect_output.get("steps", [])[:3], 1):  # Show first 3 steps
                    print(f"     {j}. {step}")
                if len(architect_output.get("steps", [])) > 3:
                    print(f"     ... and {len(architect_output['steps']) - 3} more steps")
                print()
                break

        print(f"{'='*60}")
        print()

    # Show available tools
    print("🛠️  AVAILABLE TOOLS:")
    from zero_gravity_core.tools.zerogravity_tools import list_tools
    tools = list_tools()
    for tool in tools:
        print(f"   • {tool}")
    print()

    # Save logs
    log_file = coordinator.save_logs("zerogravity_demo_logs.json")
    print(f"💾 Execution logs saved to: {log_file}")
    
    print()
    print("Thank you for experiencing the ZeroGravity platform!")
    print("This demo showcases how our multi-agent system can")
    print("decompose complex objectives and execute them through")
    print("coordinated AI agent collaboration.")


def run_custom_objective():
    """Allow users to input a custom objective"""
    print("=" * 60)
    print("           CUSTOM OBJECTIVE DEMO")
    print("=" * 60)
    print()
    
    objective = input("Enter your custom objective: ").strip()
    if not objective:
        print("No objective provided. Returning to main demo.")
        return
    
    print(f"\nProcessing custom objective: {objective}")
    print()
    
    # Initialize the Coordinator
    base_dir = Path(__file__).resolve().parent
    coordinator = Coordinator(base_dir=base_dir)
    
    # Run the workflow
    results = coordinator.run(objective)
    
    # Display results
    print("🎯 RESULTS:")
    print(f"   Objective: {results['objective']}")
    print(f"   Status: {results['execution_summary']['workflow_status']}")
    print()
    
    print("📋 AGENT OUTPUTS:")
    for entry in results["history"]:
        role = entry["role"].upper()
        if role != "ERROR":
            print(f"   {role} AGENT:")
            output = entry["output"]
            # Print a summary of the output
            if isinstance(output, dict):
                if "steps" in output:
                    steps = output.get("steps", [])
                    print(f"     Generated {len(steps)} steps")
                elif "implementation_steps" in output:
                    steps = output.get("implementation_steps", [])
                    print(f"     Created {len(steps)} implementation steps")
                elif "elements" in output:
                    elements = output.get("elements", [])
                    print(f"     Created {len(elements)} visualization elements")
                elif "execution_results" in output:
                    results_list = output.get("execution_results", [])
                    completed = sum(1 for r in results_list if r.get("status") == "completed")
                    print(f"     Executed {completed}/{len(results_list)} steps")
            print()


if __name__ == "__main__":
    print("Choose a demo option:")
    print("1. Run standard demo")
    print("2. Run custom objective")
    print()
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        run_custom_objective()
    else:
        run_demo()
