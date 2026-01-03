# zero_gravity_core/test_comprehensive.py

from pathlib import Path
from zero_gravity_core.agents.coordinator import Coordinator

def test_multiple_objectives():
    """Test the ZeroGravity platform with multiple objectives"""
    # Initialize the Coordinator
    base_dir = Path(__file__).resolve().parent
    coordinator = Coordinator(base_dir=base_dir)

    # Define multiple test objectives
    test_objectives = [
        "Build a prototype multi-agent AI platform",
        "Create a web application for task management",
        "Design a data analysis pipeline for customer insights",
        "Develop an automated testing framework"
    ]

    print("=== COMPREHENSIVE ZEROGRAVITY TEST ===\n")
    
    for i, objective in enumerate(test_objectives, 1):
        print(f"--- TEST {i}: {objective} ---\n")
        
        # Run the full workflow
        results = coordinator.run(objective)

        # Print the outputs
        print(f"Objective: {results['objective']}\n")
        
        print("--- HISTORY ---")
        for entry in results["history"]:
            role = entry["role"]
            output = entry["output"]
            print(f"\n[{role.upper()} OUTPUT]")
            print(output)
        
        print(f"\n[EXECUTION SUMMARY]")
        print(results.get("execution_summary", "No execution summary available"))
        
        print(f"\n{'='*50}\n")

    # Save logs
    log_file = coordinator.save_logs()
    print(f"Execution logs saved to: {log_file}")

if __name__ == "__main__":
    test_multiple_objectives()
