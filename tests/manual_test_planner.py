from research_agent.planner import Planner
from research_agent.utils import LLMClient
import os

def test_planner():
    # Force mock if needed, or let auto-detect
    # os.environ["GOOGLE_API_KEY"] = "..." # Uncomment to test real key if local
    
    client = LLMClient(provider="auto")
    planner = Planner(client)
    
    query = "Research the impact of quantum computing on cryptography"
    print(f"Testing Planner with query: '{query}'")
    
    plan = planner.plan(query)
    
    if plan:
        print("\nPlan Generated Successfully:")
        print(f"Goal: {plan.research_goal}")
        print("Steps:")
        for step in plan.steps:
            print(f"  {step.id}. [{step.type}] {step.description}")
    else:
        print("Plan generation failed.")

if __name__ == "__main__":
    test_planner()
