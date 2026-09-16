import sys
import os
import json
import time
from typing import List, Dict

# Path setup to import research_agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_agent.utils import LLMClient, setup_logger
from research_agent.planner import Planner
from research_agent.executor import Executor
from research_agent.synthesizer import Synthesizer
from research_agent.verifier import Verifier
from research_agent.schema import ExecutionMode, FinalOutcome

logger = setup_logger("evaluation_runner")

def run_evaluation():
    prompts_path = os.path.join(os.path.dirname(__file__), "prompts.json")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    results = []
    
    # Init components
    client = LLMClient(provider="auto")
    planner = Planner(client)
    # We will instantiate Executor per run to reset state cleanly/set mode
    synthesizer = Synthesizer(client)
    verifier = Verifier(client)

    modes = [ExecutionMode.NORMAL, ExecutionMode.STRESS_TEST]

    print(f"Starting evaluation on {len(prompts)} prompts x {len(modes)} modes...")

    for prompt_data in prompts:
        for mode in modes:
            query = prompt_data["query"]
            result_record = {
                "prompt_id": prompt_data["id"],
                "category": prompt_data["category"],
                "mode": mode.value,
                "query": query,
                "timestamp": time.time(),
                "plan_steps": 0,
                "verification_status": "N/A",
                "final_outcome": "N/A",
                "confidence_adjustment": "N/A"
            }

            print(f"\n--- PROMPT {prompt_data['id']} [{mode.value}]: {query} ---")
            
            try:
                # 1. Plan
                plan = planner.plan(query)
                if not plan:
                    result_record["error"] = "Planning Failed"
                    results.append(result_record)
                    continue
                
                result_record["plan_steps"] = len(plan.steps)
                
                # 2. Execute
                executor = Executor(client) # Fresh instance
                executor.set_mode(mode)
                execution_log = executor.run(plan)
                
                # 3. Synthesize
                synthesis = synthesizer.synthesize(execution_log, query)
                
                # 4. Verify
                verification = verifier.verify(plan, execution_log, synthesis)
                
                result_record["verification_status"] = verification.status
                result_record["final_outcome"] = verification.final_outcome.value
                result_record["confidence_adjustment"] = verification.confidence_adjustment
                result_record["abstention_reason"] = verification.abstention_reason
                
                print(f"Outcome: {verification.final_outcome.value} | Status: {verification.status}")
                if verification.final_outcome == FinalOutcome.ABSTAINED:
                    print(f"Reason: {verification.abstention_reason}")

            except Exception as e:
                logger.error(f"Run failed: {e}")
                result_record["error"] = str(e)
            
            results.append(result_record)

    # Save results
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "results", "latest_run.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEvaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()
