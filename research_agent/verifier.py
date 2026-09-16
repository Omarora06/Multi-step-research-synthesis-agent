from .schema import VerificationOutput, ResearchPlan, SynthesisOutput, FinalOutcome
from .executor import ExecutionLog
from .utils import LLMClient, setup_logger
import json

logger = setup_logger("verifier")

class Verifier:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def verify(self, plan: ResearchPlan, execution_log: ExecutionLog, synthesis: SynthesisOutput) -> VerificationOutput:
        logger.info("Running verification...")
        
        # 1. Coverage Check (Rule-based)
        coverage = {}
        executed_ids = set(log['step_id'] for log in execution_log.log if log['status'] == 'success')
        for step in plan.steps:
            coverage[step.id] = step.id in executed_ids
            
        all_covered = all(coverage.values())
        coverage_fail_count = list(coverage.values()).count(False)
        
        # 2. Epistemic Check (LLM-based)
        prompt = f"""
        VERIFY RESEARCH INTEGRITY
        
        GOAL: {plan.research_goal}
        
        SYNTHESIS:
        {synthesis.directional_summary}
        
        Are there unsupported claims? Is the confidence level appropriate?
        If the synthesis admits to knowing nothing or major key data is missing, recommend ABSTENTION.
        
        OUTPUT JSON:
        {{
            "overclaim_detected": bool,
            "missing_assumptions": ["str"],
            "required_disclaimers": ["str"],
            "confidence_adjustment": "none" | "downgrade",
            "recommend_abstain": bool,
            "abstain_reason": "str or null"
        }}
        """
        
        response = self.llm.generate(system_prompt="You are a strict Research Auditor. Output JSON only.", user_prompt=prompt)
        
        try:
            clean_json = response.strip()
            if clean_json.startswith("```json"): clean_json = clean_json[7:]
            if clean_json.endswith("```"): clean_json = clean_json[:-3]
            
            data = json.loads(clean_json)
            
            # --- ABSTENTION LOGIC ---
            # We abstain if:
            # 1. LLM recommends it explicitly
            # 2. More than 50% of steps failed (Critical data loss)
            # 3. Confidence is low AND overclaim is detected (High risk of hallucination)
            
            final_outcome = FinalOutcome.ANSWERED
            abstention_reason = None
            status = "pass"
            
            recommend_abstain = data.get("recommend_abstain", False)
            critical_failure = coverage_fail_count > (len(plan.steps) / 2)
            
            if recommend_abstain:
                final_outcome = FinalOutcome.ABSTAINED
                abstention_reason = data.get("abstain_reason", "Verifier recommended abstention due to content analysis.")
                status = "warn" # Abstention is a warning state in this UI, or handled separately
            elif critical_failure:
                final_outcome = FinalOutcome.ABSTAINED
                abstention_reason = "Critical execution failure: Majority of research steps failed."
                status = "fail"
            elif not executed_ids:
                final_outcome = FinalOutcome.ABSTAINED
                abstention_reason = "Total execution failure: No steps completed."
                status = "fail"
                
            # Downgrade logic if not abstaining
            if final_outcome == FinalOutcome.ANSWERED:
                if data.get("overclaim_detected") or not all_covered or data.get("confidence_adjustment") == "downgrade":
                    status = "warn"
            
            return VerificationOutput(
                status=status,
                final_outcome=final_outcome,
                abstention_reason=abstention_reason,
                coverage_check=coverage,
                overclaim_detected=data.get("overclaim_detected", False),
                missing_assumptions=data.get("missing_assumptions", []),
                required_disclaimers=data.get("required_disclaimers", []),
                confidence_adjustment=data.get("confidence_adjustment", "none")
            )
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return VerificationOutput(
                status="fail",
                final_outcome=FinalOutcome.ABSTAINED,
                abstention_reason=f"Verification process failed: {e}",
                coverage_check=coverage,
                overclaim_detected=True,
                missing_assumptions=["Verification process failed"],
                required_disclaimers=["System integrity check failed"],
                confidence_adjustment="downgrade"
            )
