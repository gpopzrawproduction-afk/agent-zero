from pathlib import Path
from zero_gravity_core.agents.coordinator import Coordinator

def main():
    # Initialize the Coordinator
    base_dir = Path(__file__).resolve().parent
    coordinator = Coordinator(base_dir=base_dir)

    # Define a test objective
    test_objective = "Build a prototype multi-agent AI platform"

    # Run the full workflow
    results = coordinator.run(test_objective)

    # Print the outputs
    print("\n=== FINAL RESULTS ===")
    print(f"Objective: {results['objective']}\n")
    
    print("--- HISTORY ---")
    for entry in results["history"]:
        role = entry["role"]
        output = entry["output"]
        print(f"\n[{role.upper()} OUTPUT]")
        print(output)

    print("\n=== WORKFLOW COMPLETE ===")

if __name__ == "__main__":
    main()
