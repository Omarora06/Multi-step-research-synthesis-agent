from research_agent.schema import ResearchPlan, ResearchStep
from research_agent.executor import Executor
import json

def test_executor():
    # Define a simple plan
    plan = ResearchPlan(
        research_goal="Compare renewable energy sources",
        assumptions=["Focus on Solar and Wind"],
        steps=[
            ResearchStep(
                id=1,
                type="research",
                description="Research Solar Energy efficiency",
                constraints=["2023 data"]
            ),
            ResearchStep(
                id=2,
                type="research",
                description="Research Wind Energy efficiency",
                constraints=["2023 data"]
            ),
            ResearchStep(
                id=3,
                type="compare",
                description="Compare Solar and Wind efficiency",
                inputs=[1, 2],
                constraints=["efficiency", "cost"]
            )
        ]
    )

    print("Plan defined.")
    
    executor = Executor()
    print("Running executor...")
    execution_log = executor.run(plan)
    
    print("\nExecution Log:")
    for entry in execution_log.log:
        print(f"Step {entry['step_id']}: {entry['status']}")
        if entry['status'] == 'success':
            print(f"  Output Type: {type(entry['output'])}")
            print(f"  Output Content: {entry['output'].dict()}")
        elif entry['status'] == 'error':
            print(f"  Error: {entry['error']}")

if __name__ == "__main__":
    test_executor()
