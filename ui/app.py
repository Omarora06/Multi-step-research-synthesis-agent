import streamlit as st
import sys
import os
import json
import time

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_agent.utils import LLMClient, setup_logger
from research_agent.planner import Planner
from research_agent.executor import Executor
from research_agent.synthesizer import Synthesizer
from research_agent.verifier import Verifier
from research_agent.schema import ResearchPlan, ExecutionMode, FinalOutcome

st.set_page_config(page_title="Agentic Research Assistant (V2)", layout="wide")

# Custom CSS for "Architecture" feel
st.markdown("""
<style>
    .stExpander { border: 1px solid #ddd; border-radius: 5px; }
    .step-box { padding: 10px; margin: 5px 0; border-left: 4px solid #4CAF50; background-color: #f9f9f9; color: black; }
    .warning-box { padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; color: black; }
    .failure-box { padding: 10px; background-color: #f8d7da; border-left: 4px solid #dc3545; color: black; }
    .abstain-box { padding: 15px; background-color: #e2e3e5; border-left: 5px solid #6c757d; color: black; }
    .stress-banner { background-color: #ffcccc; color: #cc0000; padding: 10px; text-align: center; font-weight: bold; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("Research Agent V2 (Stressed)")
st.markdown("**System Status**: `Online` | **Architecture**: `V2 (Failure-Aware)`")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    
    # V2: Execution Mode
    mode_selection = st.radio("Execution Mode", ["Normal", "Stress Test"])
    execution_mode = ExecutionMode.NORMAL if mode_selection == "Normal" else ExecutionMode.STRESS_TEST
    
    if execution_mode == ExecutionMode.STRESS_TEST:
        st.warning("⚠️ STRESS MODE ACTIVE. Tools will return degraded data.")

    api_key = st.text_input("API Key (Google/OpenAI)", type="password", help="Leave empty to try auto-detect or mock")
    provider = st.selectbox("LLM Provider", ["auto", "google", "openai", "mock"])
    
    if st.button("Reset Agent"):
        st.session_state.clear()
        st.experimental_rerun()

    with st.expander("ℹ️ Help & Documentation"):
        st.markdown("""
        ### Verification Status
        - **PASS**: High confidence, all steps completed.
        - **WARN**: Partial data, missing steps, or potential overclaim.
        - **FAIL**: System error or complete inability to answer.
        
        ### V2 Features
        - **Stress Mode**: Deliberately injects noise to test robustness.
        - **Abstention**: The agent can refuse to answer if data is insufficient.
        """)

# Initialize session state
if "execution_log" not in st.session_state:
    st.session_state.execution_log = None
if "plan" not in st.session_state:
    st.session_state.plan = None
if "synthesis" not in st.session_state:
    st.session_state.synthesis = None
if "verification" not in st.session_state:
    st.session_state.verification = None

# Main Input
if execution_mode == ExecutionMode.STRESS_TEST:
    st.markdown("<div class='stress-banner'>⚠️ EXECUTION MODE: STRESS TEST (Failures are intentional)</div>", unsafe_allow_html=True)

query = st.text_area("Research Query", height=100, placeholder="e.g., Compare the economic impact of remote work vs office work in 2024")

if st.button("Start Research"):
    with st.spinner("Initializing Components..."):
        # Init components
        client = LLMClient(provider=provider, api_key=api_key if api_key else None)
        planner = Planner(client)
        executor = Executor(client)
        executor.set_mode(execution_mode) # V2 Mode Set
        synthesizer = Synthesizer(client)
        verifier = Verifier(client)
        
    # 1. PLAN
    with st.status("Planning...", expanded=True) as status:
        st.write("Decomposing query into research steps...")
        plan = planner.plan(query)
        if plan:
            st.session_state.plan = plan
            st.write("Plan generated.")
            status.update(label="Planning Complete", state="complete")
        else:
            status.update(label="Planning Failed", state="error")
            st.stop()
            
    # Display Plan
    with st.expander("Research Plan", expanded=True):
        st.json(plan.model_dump_json())

    # 2. EXECUTE
    st.subheader("Execution Timeline")
    execution_container = st.container()
    
    with st.status("Running Execution Loop...", expanded=True) as exec_status:
        execution_log = executor.execution_log
        
        # We manually iterate to show progress in UI
        if not executor.validate_plan(plan):
            st.error("Plan Validation Failed")
            st.stop()
            
        progress_bar = st.progress(0)
        total_steps = len(plan.steps)
        
        for i, step in enumerate(plan.steps):
            st.write(f"**Step {step.id}**: {step.type} - _{step.description}_")
            success = executor.execute_step(step)
            
            # Update artifacts display in real-time
            if success:
                 entry = executor.execution_log.log[-1]
                 with execution_container:
                     st.markdown(f"<div class='step-box'><b>Step {step.id} Completed</b><br/>Output: {str(entry['output'])[:200]}...</div>", unsafe_allow_html=True)
            else:
                 st.error(f"Step {step.id} Failed")
                 exec_status.update(state="error")
                 break
            
            progress_bar.progress((i + 1) / total_steps)
            time.sleep(0.5) # UX pause
            
        st.session_state.execution_log = executor.execution_log
        exec_status.update(label="Execution Complete", state="complete")

    # 3. SYNTHESIZE
    if st.session_state.execution_log:
        with st.status("Synthesizing...", expanded=True) as synth_status:
            synthesis = synthesizer.synthesize(executor.execution_log, query)
            st.session_state.synthesis = synthesis
            synth_status.update(label="Synthesis Complete", state="complete")

    # 4. VERIFY & DISPLAY
    if st.session_state.synthesis:
        with st.spinner("Verifying..."):
            verification = verifier.verify(plan, executor.execution_log, synthesis)
            st.session_state.verification = verification
            
        st.subheader("Verification Report")
        
        # V2 ABSTENTION DISPLAY
        if verification.final_outcome == FinalOutcome.ABSTAINED:
            st.markdown(f"""
            <div class='abstain-box'>
                <h3>🚫 System Abstained</h3>
                <p><b>Reason:</b> {verification.abstention_reason}</p>
                <p><i>The system determined that it could not provide a reliable answer based on the available data.</i></p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # NORMAL ANSWER DISPLAY
            if verification.status == "pass":
                st.success(f"Status: PASS | Confidence Adjustment: {verification.confidence_adjustment}")
            elif verification.status == "warn":
                st.warning(f"Status: WARN | Confidence Adjustment: {verification.confidence_adjustment}")
            else:
                st.error("Status: FAIL")

            # Show Synthesis only if NOT abstained
            st.subheader("Directional Synthesis")
            st.info(synthesis.directional_summary)
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Hypotheses:**")
                for h in synthesis.hypotheses:
                    st.markdown(f"- {h}")
            with c2:
                st.write("**Open Questions:**")
                for q in synthesis.open_questions:
                    st.markdown(f"- {q}")
            
        with st.expander("Verification Details"):
            st.write("Outcome:", verification.final_outcome.value)
            st.write("Coverage:", verification.coverage_check)
            st.write("Overclaims:", verification.overclaim_detected)
            st.write("Missing Assumptions:", verification.missing_assumptions)

st.markdown("---")
st.caption("Multi-Step Research Agent V2 | Built by Aditya Kumar Singh")
