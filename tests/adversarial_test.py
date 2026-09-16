from research_agent.utils import LLMClient
from research_agent.planner import Planner
from research_agent.verifier import Verifier
from research_agent.executor import Executor
from research_agent.synthesizer import Synthesizer

def test_adversarial():
    client = LLMClient(provider="auto")
    planner = Planner(client)
    
    # 1. Vague Query
    vague_query = "cloud computing"
    print(f"Testing Vague Query: '{vague_query}'")
    plan = planner.plan(vague_query)
    if plan and len(plan.steps) > 0:
        print("  -> Planned successfully (Planner likely refined the scope).")
    else:
        print("  -> Planner correctly refused or failed on vague query.")

    # 2. Impossible/Nonsense Query
    nonsense_query = "Compare the battery life of a rock vs a cloud"
    print(f"\nTesting Nonsense Query: '{nonsense_query}'")
    plan = planner.plan(nonsense_query)
    
    if plan:
        print("  -> Plan generated (System attempted research). Executing...")
        executor = Executor(client)
        log = executor.run(plan)
        
        synthesizer = Synthesizer(client)
        synthesis = synthesizer.synthesize(log, nonsense_query)
        
        verifier = Verifier(client)
        # We expect a warning or low confidence
        verification = verifier.verify(plan, log, synthesis)
        print(f"  -> Verification Status: {verification.status}")
        print(f"  -> Confidence Adjustment: {verification.confidence_adjustment}")
        
        if verification.status == "warn" or verification.confidence_adjustment == "downgrade":
             print("  SUCCESS: System flagged the nonsense/uncertainty.")
        else:
             print("  NOTICE: System treated it as valid (Review needed).")

if __name__ == "__main__":
    test_adversarial()
