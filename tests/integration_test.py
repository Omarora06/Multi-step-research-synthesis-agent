import sys
import os

# Ensure we can import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_agent.utils import LLMClient
from research_agent.planner import Planner
from research_agent.executor import Executor
from research_agent.synthesizer import Synthesizer
from research_agent.verifier import Verifier

def test_full_pipeline():
    print("Initializing components...")
    client = LLMClient(provider="auto") # Will likely mock
    
    planner = Planner(client)
    executor = Executor(client) # Passes client to tools
    synthesizer = Synthesizer(client)
    verifier = Verifier(client)
    
    query = "Compare potential benefits of AI in Healthcare vs Finance"
    print(f"\n1. Planning for query: '{query}'")
    plan = planner.plan(query)
    
    if not plan:
        print("Planning failed.")
        return

    print("Plan created:")
    for step in plan.steps:
        print(f"  - {step.id}: {step.type} ({step.description})")
        
    print("\n2. Executing Plan...")
    execution_log = executor.run(plan)
    
    print("Execution complete. Artifacts generated:")
    for step_id, artifact in execution_log.artifacts.items():
        print(f"  - Step {step_id}: {type(artifact).__name__}")
        
    print("\n3. Synthesizing...")
    synthesis = synthesizer.synthesize(execution_log, query)
    print("Synthesis Output:")
    print(f"  Summary: {synthesis.directional_summary}")
    print(f"  Hypotheses: {synthesis.hypotheses}")
    
    print("\n4. Verifying...")
    verification = verifier.verify(plan, execution_log, synthesis)
    print("Verification Report:")
    print(f"  Status: {verification.status}")
    print(f"  Coverage: {verification.coverage_check}")
    print(f"  Overclaim Detected: {verification.overclaim_detected}")

if __name__ == "__main__":
    test_full_pipeline()
