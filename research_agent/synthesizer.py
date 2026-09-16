from .schema import SynthesisOutput
from .utils import LLMClient, setup_logger
from .executor import ExecutionLog
import json

logger = setup_logger("synthesizer")

class Synthesizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def synthesize(self, execution_log: ExecutionLog, original_goal: str) -> SynthesisOutput:
        logger.info("Synthesizing results...")
        
        # Prepare context
        artifacts_text = ""
        for step_id, artifact in execution_log.artifacts.items():
            artifacts_text += f"\n--- Step {step_id} Output ---\n{str(artifact)}\n"
            
        prompt = f"""
        GOAL: {original_goal}
        
        ARTIFACTS:
        {artifacts_text}
        
        Synthesize these findings into a directional summary.
        Be honest about uncertainty.
        
        OUTPUT JSON SCHEMA:
        {{
            "directional_summary": "str",
            "hypotheses": ["str"],
            "supported_by": {{"hypothesis_text": [int]}}, 
            "open_questions": ["str"],
            "confidence": "low" | "medium"
        }}
        EXAMPLE:
        "supported_by": {{"Hypothesis A": [1, 3]}}
        """
        
        response = self.llm.generate(system_prompt="You are a Research Synthesizer. Output JSON only. Step IDs must be INTEGERS.", user_prompt=prompt)
        
        try:
            clean_json = response.strip()
            if clean_json.startswith("```json"): clean_json = clean_json[7:]
            if clean_json.endswith("```"): clean_json = clean_json[:-3]
            
            data = json.loads(clean_json)
            
            # Data Cleaning: Ensure supported_by list contains integers
            if "supported_by" in data:
                cleaned_supported_by = {}
                for hypo, ids in data["supported_by"].items():
                    cleaned_ids = []
                    for i in ids:
                        if isinstance(i, int):
                            cleaned_ids.append(i)
                        elif isinstance(i, str):
                            # Try to extract number from "Step_1" or "Step 1"
                            try:
                                # Remove non-digit chars
                                digit_str = ''.join(filter(str.isdigit, i))
                                if digit_str:
                                    cleaned_ids.append(int(digit_str))
                            except:
                                pass
                    cleaned_supported_by[hypo] = cleaned_ids
                data["supported_by"] = cleaned_supported_by

            return SynthesisOutput(**data)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return SynthesisOutput(
                directional_summary="Failed to synthesize results.",
                hypotheses=[],
                supported_by={},
                open_questions=["System Error during synthesis"],
                confidence="low"
            )
