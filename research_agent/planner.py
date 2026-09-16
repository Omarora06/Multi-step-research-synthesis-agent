import json
from typing import Optional
from .schema import ResearchPlan, ResearchStep
from .utils import setup_logger, LLMClient

logger = setup_logger("planner")

PLANNER_SYSTEM_PROMPT = """
You are a Research Planner for an advanced AI research agent.
Your goal is to decompose a user's research query into a structured, step-by-step execution plan.

**CORE RULES:**
1. You must output ONLY valid JSON matching the `ResearchPlan` schema.
2. No conversational prowess, no preambles, no explanations.
3. The plan must be logical, deterministic, and efficient.
4. Steps must be strictly ordered (lower IDs first).
5. Use "research" type for gathering information.
6. Use "compare" type for comparing gathered information.
7. Use "synthesize" type ONLY as the final step to aggregate everything.

**SCHEMA:**
{
    "research_goal": "str",
    "assumptions": ["str", ...],
    "steps": [
        {
            "id": int,
            "type": "research" | "compare" | "synthesize",
            "description": "str (precise instructions for the tool)",
            "inputs": [int, ...] or null,
            "constraints": ["str", ...] or null
        }
    ],
    "stop_conditions": {"max_steps": int}
}

**EXAMPLE:**
User: "Compare the battery life of iPhone 15 vs Pixel 8"
JSON:
{
    "research_goal": "Compare battery life: iPhone 15 vs Pixel 8",
    "assumptions": ["Assuming standard usage models", "latest OS versions"],
    "steps": [
        {
            "id": 1, 
            "type": "research", 
            "description": "Find battery specs and test results for iPhone 15", 
            "constraints": ["official specs", "tomsguide reviews"]
        },
        {
            "id": 2, 
            "type": "research", 
            "description": "Find battery specs and test results for Pixel 8", 
            "constraints": ["official specs", "tomsguide reviews"]
        },
        {
            "id": 3, 
            "type": "compare", 
            "description": "Compare battery performance metrics", 
            "inputs": [1, 2], 
            "constraints": ["hours of video playback", "mAh capacity"]
        },
        {
            "id": 4,
            "type": "synthesize",
            "description": "Synthesize findings on battery life comparison",
            "inputs": [1, 2, 3]
        }
    ],
    "stop_conditions": {"max_steps": 5}
}
"""

class Planner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, user_query: str) -> Optional[ResearchPlan]:
        logger.info(f"Planning for query: {user_query}")
        
        prompt = f"User Query: {user_query}\n\nGenerate the JSON ResearchPlan:"
        
        try:
            # In a real implementation with a chat model, we'd pass system prompt strictly.
            # Here we combine for simplicity depending on the client.
            full_response = self.llm.generate(system_prompt=PLANNER_SYSTEM_PROMPT, user_prompt=prompt)
            
            # Clean response (remove markdown fences if any)
            clean_json = full_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            
            data = json.loads(clean_json)
            plan = ResearchPlan(**data)
            logger.info("Plan generated successfully.")
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Planner JSON: {e}")
            logger.debug(f"Raw output: {full_response}")
            return None
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return None
