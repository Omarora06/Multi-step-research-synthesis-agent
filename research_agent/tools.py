from .schema import ResearchInput, ResearchOutput, ComparisonInput, ComparisonOutput, ExecutionMode
from .utils import LLMClient, setup_logger
from typing import List
import json
import random

logger = setup_logger("tools")

def execute_research(input_data: ResearchInput, llm_client: LLMClient = None, mode: ExecutionMode = ExecutionMode.NORMAL) -> ResearchOutput:
    """
    Research tool. Supports STRESS_TEST mode to inject noise.
    """
    topic = input_data.topic
    logger.info(f"Researching topic: {topic} | Mode: {mode.value}")
    
    # --- STRESS MODE LOGIC ---
    if mode == ExecutionMode.STRESS_TEST:
        logger.warning(f"Injector: Applying stress to research on '{topic}'")
        
        # Probabilistic failure types
        failure_type = random.choice(["partial", "contradictory", "empty"])
        
        if failure_type == "empty":
             return ResearchOutput(
                topic=topic,
                summary="Insufficient data available to summarize.",
                key_points=[],
                assumptions=["Data source unreachable in stress test"],
                confidence="low",
                gaps=["All information missing due to stress injection"],
                sources=[]
            )
        elif failure_type == "contradictory":
             return ResearchOutput(
                topic=topic,
                summary=f"Conflicting reports found regarding {topic}.",
                key_points=[
                    f"Source A claims {topic} is highly effective.",
                    f"Source B claims {topic} has no measurable impact."
                ],
                assumptions=["Conflicting methodologies in sources"],
                confidence="low",
                gaps=["Unable to reconcile Source A and Source B"],
                sources=["SourceA", "SourceB"]
            )
        # Default partial
        return ResearchOutput(
            topic=topic,
            summary=f"Partial data found for {topic}.",
            key_points=[f"Some evidence suggests relevance, but details are missing."],
            assumptions=["Extrapolating from limited data"],
            confidence="low",
            gaps=["Critical metrics missing"],
            sources=["FragmentedDB"]
        )

    # --- NORMAL MODE ---
    return ResearchOutput(
        topic=topic,
        summary=f"Research Summary for {topic} (Simulated Data)",
        key_points=[
            f"{topic} is a complex subject.",
            f"Recent studies (2024) show significant interest in {topic}."
        ],
        assumptions=["Results are simulated for demonstration"],
        confidence="medium",
        gaps=["Lack of live internet access in this environment"],
        sources=["SimulationDB"]
    )

def execute_compare(input_data: ComparisonInput, llm_client: LLMClient, mode: ExecutionMode = ExecutionMode.NORMAL) -> ComparisonOutput:
    """
    Comparison tool. Supports STRESS_TEST mode.
    """
    logger.info(f"Executing comparison on dimensions: {input_data.dimensions} | Mode: {mode.value}")
    
    if mode == ExecutionMode.STRESS_TEST:
        return ComparisonOutput(
            dimensions=input_data.dimensions,
            contrasts={d: {k: "Ambiguous data prevented clear contrast" for k in input_data.items} for d in input_data.dimensions},
            tradeoffs=["Unable to determine clear tradeoffs due to data noise."],
            uncertainties=["High uncertainty in input data reliability."]
        )

    # Normal Logic
    items_text = ""
    for name, output in input_data.items.items():
        items_text += f"--- ITEM: {name} ---\nSummary: {output.summary}\nKey Points: {output.key_points}\n\n"
        
    prompt = f"""
    Compare the following items based on these dimensions: {input_data.dimensions}.
    
    ITEMS:
    {items_text}
    
    OUTPUT JSON FORMAT:
    {{
        "dimensions": {json.dumps(input_data.dimensions)},
        "contrasts": {{
            "dimension_name": {{
                "item_name": "contrast description"
            }}
        }},
        "tradeoffs": ["string", ...],
        "uncertainties": ["string", ...]
    }}
    """
    
    response = llm_client.generate(system_prompt="You are a precise comparison engine. Output JSON only.", user_prompt=prompt)
    
    try:
        clean_json = response.strip()
        if clean_json.startswith("```json"): clean_json = clean_json[7:]
        if clean_json.endswith("```"): clean_json = clean_json[:-3]
        
        data = json.loads(clean_json)
        return ComparisonOutput(**data)
    except Exception as e:
        logger.error(f"Comparison LLM failed: {e}")
        return ComparisonOutput(
            dimensions=input_data.dimensions,
            contrasts={d: {k: "Error generating contrast" for k in input_data.items} for d in input_data.dimensions},
            tradeoffs=["Comparison generation failed"],
            uncertainties=["LLM Error"]
        )
