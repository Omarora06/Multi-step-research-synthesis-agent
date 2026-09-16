from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Optional, Union
from enum import Enum

# --- V2 Enums ---
class ExecutionMode(str, Enum):
    NORMAL = "normal"
    STRESS_TEST = "stress_test"

class FinalOutcome(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"

# --- Planner Schemas ---

class ResearchStep(BaseModel):
    id: int = Field(..., description="Unique identifier for the step")
    type: Literal["research", "compare", "synthesize"] = Field(..., description="Type of action to perform")
    description: str = Field(..., description="Clear instructions for what this step should achieve")
    inputs: Optional[List[int]] = Field(default=None, description="List of Step IDs whose outputs are required for this step")
    constraints: Optional[List[str]] = Field(default=None, description="Specific constraints or focus areas for this step")

class ResearchPlan(BaseModel):
    research_goal: str = Field(..., description="The overarching goal of the research")
    assumptions: List[str] = Field(default_factory=list, description="Initial assumptions made by the planner")
    steps: List[ResearchStep] = Field(..., description="Ordered list of steps to execute")
    stop_conditions: Dict[str, int] = Field(default_factory=lambda: {"max_steps": 10}, description="Conditions to halt execution")

# --- Tool Schemas ---

class ResearchInput(BaseModel):
    topic: str = Field(..., description="Main topic or query to research")
    scope: Optional[str] = Field(default=None, description="Scope limitation")
    constraints: Optional[List[str]] = Field(default=None, description="Constraints to apply")

class ResearchOutput(BaseModel):
    topic: str
    summary: str
    key_points: List[str]
    assumptions: List[str]
    confidence: Literal["low", "medium", "high"]
    gaps: List[str]
    sources: Optional[List[str]] = Field(default=None, description="List of sources used")

class ComparisonInput(BaseModel):
    items: Dict[str, ResearchOutput] = Field(..., description="Dict of Item Name -> ResearchOutput to compare")
    dimensions: List[str] = Field(..., description="Specific dimensions to compare")

class ComparisonOutput(BaseModel):
    dimensions: List[str]
    contrasts: Dict[str, Dict[str, str]] = Field(..., description="Structure: {dimension: {item_name: value_description}}")
    tradeoffs: List[str]
    uncertainties: List[str]

# --- Synthesis Schemas ---

class SynthesisOutput(BaseModel):
    directional_summary: str = Field(..., description="A non-authoritative summary of the findings")
    hypotheses: List[str]
    supported_by: Dict[str, List[int]] = Field(..., description="Mapping of hypothesis to supporting Step IDs")
    open_questions: List[str]
    confidence: Literal["low", "medium"]

# --- Verification Schemas ---

class VerificationOutput(BaseModel):
    status: Literal["pass", "warn", "fail"]
    final_outcome: FinalOutcome = Field(default=FinalOutcome.ANSWERED, description="Whether the system answered or abstained")
    abstention_reason: Optional[str] = Field(default=None, description="Reason for abstention if applicable")
    coverage_check: Dict[int, bool] = Field(..., description="Mapping of Step ID to whether it was covered")
    overclaim_detected: bool
    missing_assumptions: List[str]
    required_disclaimers: List[str]
    confidence_adjustment: Literal["none", "downgrade"]
