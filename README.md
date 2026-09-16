# Multi-Step Research & Synthesis Agent

A deterministic, resume-grade AI Agent that functions as a Planner-Executor system. It is designed to tackle open-ended research questions by decomposing them into structured steps, executing them, and synthesizing a "directional" answer with strict verification.

**Status**: V2 (Failure-Aware) | **Mode**: Planner-Executor | **Built with**: Python, Streamlit, Pydantic, Google Gemini

---

## Architecture

This is **NOT** a simple chatbot. It is an agentic system with strict separation of concerns:

1.  **Planner (LLM)**: Pure structured reasoning. No tool execution. Outputs a JSON `ResearchPlan`.
2.  **Executor (Python)**: Deterministic state machine. Iterates logic, dispatches tools, and stores artifacts.
3.  **Tools**:
    *   `ResearchTool`: Scoped information retrieval.
    *   `ComparisonTool`: LLM-based structured comparison.
4.  **Synthesizer (LLM)**: Integrates execution artifacts into a directional summary.
5.  **Verifier (Rules + LLM)**: Audits the process for missing steps, overclaims, or low confidence.

## Features (V2)

*   **Stress Test Mode**: A system-level toggle to deliberately inject noise and data degradation. Ensures the system fails safely rather than hallucinating.
*   **Explicit Abstention**: The Verifier has the authority to block an answer ("System Abstained") if data is insufficient or confidence is low.
*   **Evaluation Harness**: Automated testing against adversarial prompts (ambiguous, nonsense, unanswerable questions).
*   **Transparent Execution**: Watch the agent "think" step-by-step in the UI.
*   **Epistemic Honesty**: The agent explicitly lists "Open Questions" and "Hypotheses".

## Installation & Run

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/goblinasaddy/research-agent.git
    cd research-agent
    ```

2.  **Install Dependencies**:
    ```bash
    pip install streamlit pydantic google-generativeai openai
    ```

3.  **Set API Key**:
    The system auto-detects `GOOGLE_API_KEY` or `OPENAI_API_KEY` from your environment.
    ```powershell
    # Windows PowerShell
    $env:GOOGLE_API_KEY = "your-key-here"
    ```
    *Alternatively, you can enter the key in the UI sidebar.*

4.  **Run the UI**:
    ```bash
    streamlit run ui/app.py
    ```

5.  **Run Evaluation Harness**:
    ```bash
    python evaluation/run_evaluation.py
    ```

## Project Structure

```text
research-agent/
├── research_agent/
│   ├── schema.py       # Pydantic models (The "Contract")
│   ├── planner.py      # LLM logic for planning
│   ├── executor.py     # Main loop & logic
│   ├── tools.py        # Research & Compare tools
│   ├── synthesizer.py  # Summarization logic
│   ├── verifier.py     # Integrity check logic
│   └── utils.py        # Logging & LLM Client
├── ui/
│   └── app.py          # Streamlit Interface
├── tests/              # Manual & Integration tests
└── README.md
```

## Verification Report Explained

The UI includes a **Verification Report** to help you trust the output:
*   **PASS**: High confidence, all steps completed.
*   **WARN**: Partial data, missing steps, or potential overclaim.
*   **FAIL**: System error or complete inability to answer.

## Author
**Aditya Kumar Singh**
*   [GitHub](https://github.com/goblinasaddy)
