"""Streamlit Web Application for the Voting Panel Agent.

Provides an interactive web UI with:
- Message input & pre-built preset test cases.
- Parallel juror cards (Strict Guard, Friendly Person, Data Analyst).
- Live heuristic warning signals breakdown.
- Tally and Tie-Breaker Judge resolution view.
- LangGraph flow visualization.
"""

import streamlit as st
from agent import create_voting_panel_graph, format_verdict_badge
from tools import spam_signals

st.set_page_config(
    page_title="Voting Panel Agent — Spam Classifier",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS for styling cards and badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .juror-card {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
        height: 100%;
        margin-bottom: 1rem;
    }
    .badge-spam {
        background-color: #ff4b4b;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-not-spam {
        background-color: #21ba45;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .badge-unsure {
        background-color: #f2711c;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        display: inline-block;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cache graph instance
@st.cache_resource
def load_graph():
    return create_voting_panel_graph()

graph = load_graph()

# Header
st.markdown('<div class="main-title">🛡️ Voting Panel Agent — Spam Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-Agent Ensemble built with <b>LangGraph</b> and <b>Ollama (llama3.2)</b> featuring 3 parallel jurors and a tie-breaker judge.</div>',
    unsafe_allow_html=True,
)

# Preset buttons
st.subheader("1. Enter or Select a Message")

col_btn1, col_btn2, col_btn3 = st.columns(3)
preset_text = None

if col_btn1.button("🚨 Load Obvious Scam Preset"):
    st.session_state["message_input"] = (
        "URGENT! You won a FREE $1000 Walmart Gift Card! Click immediately to claim: http://scam.me/win"
    )

if col_btn2.button("💬 Load Safe Chat Preset"):
    st.session_state["message_input"] = (
        "Hey Sarah, hope you have a nice day! Are we still meeting for lunch tomorrow at 1?"
    )

if col_btn3.button("⚖️ Load Borderline (Judge) Preset"):
    st.session_state["message_input"] = (
        "Urgent reminder: package delivery status at https://courier.com"
    )

if "message_input" not in st.session_state:
    st.session_state["message_input"] = (
        "URGENT! You won a FREE $1000 Walmart Gift Card! Click immediately to claim: http://scam.me/win"
    )

message = st.text_area(
    "Message text:",
    value=st.session_state["message_input"],
    height=100,
    placeholder="Type or paste any SMS / email message here...",
)

analyze_clicked = st.button("🔍 Run Ensemble Analysis", type="primary", use_container_width=True)

if analyze_clicked or "last_result" in st.session_state:
    if analyze_clicked:
        with st.spinner("Invoking parallel jurors in LangGraph..."):
            result = graph.invoke({"message": message})
            st.session_state["last_result"] = result
            st.session_state["last_message"] = message
    else:
        result = st.session_state["last_result"]

    # Display Heuristic Signals Preview
    signals = spam_signals(message)

    st.markdown("---")
    st.subheader("2. Juror Evaluation (Parallel Fan-Out)")

    col1, col2, col3 = st.columns(3)

    def render_badge(vote_str: str) -> str:
        if "NOT_SPAM" in vote_str:
            return '<span class="badge-not-spam">✅ NOT_SPAM</span>'
        elif "SPAM" in vote_str:
            return '<span class="badge-spam">🚫 SPAM</span>'
        elif "UNSURE" in vote_str:
            return '<span class="badge-unsure">❓ UNSURE</span>'
        return f'<span class="badge-unsure">{vote_str}</span>'

    strict_text = result.get("strict", "N/A")
    friendly_text = result.get("friendly", "N/A")
    evidence_text = result.get("evidence", "N/A")

    strict_vote = strict_text.split(" - ")[0] if " - " in strict_text else strict_text
    strict_reason = strict_text.split(" - ")[1] if " - " in strict_text else ""

    friendly_vote = friendly_text.split(" - ")[0] if " - " in friendly_text else friendly_text
    friendly_reason = friendly_text.split(" - ")[1] if " - " in friendly_text else ""

    evidence_vote = evidence_text.split(" - ")[0] if " - " in evidence_text else evidence_text
    evidence_reason = evidence_text.split(" - ")[1] if " - " in evidence_text else ""

    with col1:
        st.markdown(
            f"""
            <div class="juror-card">
                <h4>👮‍♂️ Strict Security Guard</h4>
                <p><b>Rule:</b> Distrusts urgency, links, or payments.</p>
                <p>{render_badge(strict_vote)}</p>
                <p><i>Reason:</i> {strict_reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="juror-card">
                <h4>🤝 Friendly Reviewer</h4>
                <p><b>Rule:</b> Gives benefit of doubt; flags only obvious scams.</p>
                <p>{render_badge(friendly_vote)}</p>
                <p><i>Reason:</i> {friendly_reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="juror-card">
                <h4>📊 Data Analyst (Evidence)</h4>
                <p><b>Rule:</b> Score &ge; 4 &rarr; SPAM, &le; 1 &rarr; NOT_SPAM, 2-3 &rarr; UNSURE.</p>
                <p>{render_badge(evidence_vote)}</p>
                <p><i>Reason:</i> {evidence_reason}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("3. Tally & Verdict (Fan-In Aggregation)")

    counts = result.get("counts", {})
    verdict = result.get("verdict")
    decided_by = result.get("decided_by")

    tally_col, verdict_col = st.columns([1, 1])

    with tally_col:
        st.markdown("#### 🗳️ Vote Tally (ignoring UNSURE)")
        if counts:
            for vote_type, count in counts.items():
                badge = "🚫 SPAM" if vote_type == "SPAM" else "✅ NOT_SPAM"
                st.write(f"- **{badge}**: `{count}` vote(s)")
        else:
            st.info("No decisive votes cast (all jurors were unsure).")

        if decided_by == "majority vote":
            st.success("✅ **Majority Reached!** (&ge; 2 equal votes)")
        else:
            st.warning("⚖️ **No Majority (Split Decision)** &rarr; Directed to Tie-Breaker Judge.")

    with verdict_col:
        st.markdown("#### 🏆 Final Verdict")
        if verdict == "SPAM":
            st.error(f"### 🚫 SPAM\n**Decided by:** `{decided_by}`")
        elif verdict == "NOT_SPAM":
            st.success(f"### ✅ NOT_SPAM\n**Decided by:** `{decided_by}`")
        else:
            st.warning(f"### ❓ {verdict}\n**Decided by:** `{decided_by}`")

    # Heuristic signals breakdown expander
    with st.expander("🔬 View Counted Warning Signs (Evidence Details)", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Links", signals["links"])
        c2.metric("Urgent Words", signals["urgent_words"])
        c3.metric("Money Words", signals["money_words"])
        c4.metric("Exclamations", signals["exclamations"])
        c5.metric("Caps Ratio", f"{signals['caps_ratio'] * 100:.0f}%")
        c6.metric("Total Score", signals["score"])

# Sidebar: Graph Diagram & Instructions
with st.sidebar:
    st.header("LangGraph Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            START([START]) --> strict[👮‍♂️ Strict]
            START --> friendly[🤝 Friendly]
            START --> evidence[📊 Evidence]
            strict --> tally[🗳️ Tally]
            friendly --> tally
            evidence --> tally
            tally -->|Majority| END([END])
            tally -->|Tie / Split| judge[🧑‍⚖️ Judge]
            judge --> END
        ```
        """
    )
    st.info(
        """
        **How it works:**
        1. **Fan-Out:** START sends message to all 3 jurors simultaneously.
        2. **Fan-In:** Tally node collects all 3 juror votes.
        3. **Conditional Routing:** If $\\ge 2$ jurors agree, verdict is finalized. Otherwise, the Tie-Breaker Judge resolves the tie.
        """
    )
