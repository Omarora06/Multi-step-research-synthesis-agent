import time
from typing import List, Dict, Any, Optional
from .schema import (
    ResearchPlan, ResearchStep, ResearchOutput, ComparisonOutput, 
    ResearchInput, ComparisonInput, ExecutionMode
)
from .tools import execute_research, execute_compare
from .utils import setup_logger, LLMClient

logger = setup_logger("executor")

class ExecutionLog:
    def __init__(self):
        self.log: List[Dict[str, Any]] = []
        self.artifacts: Dict[int, Any] = {} # Step ID -> Output Object

    def add_entry(self, step_id: int, status: str, output: Any = None, error: str = None):
        entry = {
            "step_id": step_id,
            "timestamp": time.time(),
            "status": status,
            "output": output if status == "success" else None,
            "error": error
        }
        self.log.append(entry)
        if output:
            self.artifacts[step_id] = output

class Executor:
    def __init__(self, llm_client: LLMClient):
        self.execution_log = ExecutionLog()
        self.llm_client = llm_client
        self.mode = ExecutionMode.NORMAL

    def set_mode(self, mode: ExecutionMode):
        self.mode = mode

    def validate_plan(self, plan: ResearchPlan) -> bool:
        seen_ids = set()
        for step in plan.steps:
            if step.id in seen_ids:
                logger.error(f"Duplicate Step ID found: {step.id}")
                return False
            
            if step.inputs:
                for input_id in step.inputs:
                    if input_id not in seen_ids:
                        logger.error(f"Step {step.id} depends on future or non-existent Step {input_id}")
                        return False
            
            seen_ids.add(step.id)
        return True

    def execute_step(self, step: ResearchStep) -> bool:
        logger.info(f"Executing Step {step.id}: {step.type} - {step.description}")
        
        try:
            if step.type == "research":
                inp = ResearchInput(
                    topic=step.description, 
                    constraints=step.constraints
                )
                # Pass LLM client and Mode
                result = execute_research(inp, self.llm_client, mode=self.mode)
                self.execution_log.add_entry(step.id, "success", output=result)
                
            elif step.type == "compare":
                items_to_compare = {}
                if step.inputs:
                    for input_id in step.inputs:
                        prev_output = self.execution_log.artifacts.get(input_id)
                        if isinstance(prev_output, ResearchOutput):
                            items_to_compare[f"Step_{input_id}"] = prev_output
                        else:
                            logger.warning(f"Step {step.id} input {input_id} is not a ResearchOutput, skipping.")
                
                if not items_to_compare:
                     raise ValueError("Comparison step requires at least one valid ResearchOutput input.")

                inp = ComparisonInput(
                    items=items_to_compare,
                    dimensions=step.constraints if step.constraints else ["general"]
                )
                # Pass LLM client and Mode
                result = execute_compare(inp, self.llm_client, mode=self.mode)
                self.execution_log.add_entry(step.id, "success", output=result)

            elif step.type == "synthesize":
                logger.info("Synthesis step encountered. Marking as complete (processing happens after loop).")
                self.execution_log.add_entry(step.id, "success", output="Synthesis Marker")

            else:
                raise ValueError(f"Unknown step type: {step.type}")

            return True

        except Exception as e:
            logger.error(f"Step {step.id} failed: {e}")
            self.execution_log.add_entry(step.id, "error", error=str(e))
            return False

    def run(self, plan: ResearchPlan) -> ExecutionLog:
        logger.info(f"Starting execution of plan: {plan.research_goal}")
        
        if not self.validate_plan(plan):
            logger.error("Plan validation failed.")
            return self.execution_log

        for step in plan.steps:
            success = self.execute_step(step)
            if not success:
                logger.error("Stopping execution due to step failure.")
                break
        
        logger.info("Execution complete.")
        return self.execution_log
