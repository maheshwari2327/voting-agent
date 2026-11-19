"""Voting Panel Agent — Multi-Agent Spam Classifier.

Ensemble architecture using LangChain and LangGraph.
Three parallel jurors (Strict, Friendly, Evidence) classify incoming messages.
Tally node calculates majority vote (>= 2 decisive votes).
Tie-Breaker Judge resolves split decisions.
"""

from collections import Counter
import os
import sys
from typing import Any, Dict, Optional, Tuple, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from tools import parse_reason, parse_vote, spam_signals, tally

# ============================================================================
# SYSTEM PROMPTS (Runtime instructions copied from agent.py)
# ============================================================================

STRICT_PROMPT = """You are a STRICT security guard. You distrust everything: if a message pushes you to click, pay, reply with details or hurry, vote SPAM. Decide if the message is spam.
Reply in exactly two lines:
Reason: <under 12 words>
Vote: SPAM or NOT_SPAM or UNSURE"""

FRIENDLY_PROMPT = """You are a FRIENDLY, trusting person. You give people the benefit of the doubt: normal chat, reminders, receipts and newsletters from known senders are NOT_SPAM. But obvious scams (prizes, free gift cards, 'claim now', strange links) are still SPAM. Decide if the message is spam.
Reply in exactly two lines:
Reason: <under 12 words>
Vote: SPAM or NOT_SPAM or UNSURE"""

EVIDENCE_PROMPT = """You are a DATA ANALYST. You get a message AND counted warning signs. Follow this rule strictly: score 4 or more -> SPAM; score 0 or 1 -> NOT_SPAM; score 2 or 3 -> UNSURE. Quote the score in your reason.
Reply in exactly two lines:
Reason: <under 12 words>
Vote: SPAM or NOT_SPAM or UNSURE"""

JUDGE_PROMPT = """You are the TIE-BREAKER JUDGE. The panel disagreed. Read the message and their reasons and give
the final call. Reply in exactly two lines:
Reason: <one short sentence>
Vote: SPAM or NOT_SPAM"""


# ============================================================================
# STATE DEFINITION (TypedDict)
# ============================================================================

class VotingPanelState(TypedDict, total=False):
    message: str
    strict: str
    friendly: str
    evidence: str
    counts: Dict[str, int]
    verdict: Optional[str]
    decided_by: Optional[str]


# ============================================================================
# MODEL INITIALIZATION & INVOCATION
# ============================================================================

def get_llm():
    """Initializes the Ollama LLM with llama3.2 and temperature 0."""
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model_name, temperature=0, base_url=base_url)


def invoke_juror_llm(
    system_prompt: str,
    user_prompt: str,
    role: str = "strict",
    signals: Optional[Dict[str, Any]] = None,
    panel_info: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Invokes the LLM to get (vote, reason).

    Falls back gracefully to persona logic if Ollama service is unreachable.
    """
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        vote = parse_vote(content)
        reason = parse_reason(content)
        return vote, reason
    except Exception as e:
        # Graceful fallback simulation when Ollama daemon is offline/unreachable
        return simulate_juror_fallback(role, user_prompt, signals, panel_info)


def simulate_juror_fallback(
    role: str,
    user_prompt: str,
    signals: Optional[Dict[str, Any]] = None,
    panel_info: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """Fallback simulation adhering strictly to juror personalities if Ollama is offline."""
    sig = signals or spam_signals(user_prompt)

    if role == "strict":
        if sig["links"] > 0 or sig["urgent_words"] > 0 or sig["money_words"] > 0:
            return "SPAM", "Suspicious link, payment or urgency detected"
        if sig["exclamations"] > 1 or sig["caps_ratio"] > 0.2:
            return "SPAM", "Excessive exclamations or uppercase letters"
        return "NOT_SPAM", "No obvious pressure or links detected"

    elif role == "friendly":
        # Friendly gives benefit of doubt to normal chat, reminders, receipts
        # Only obvious scams (prizes, gift cards, claim now, lottery) are SPAM
        lower_prompt = user_prompt.lower()
        obvious_scam_keywords = ["prize", "gift card", "claim now", "winner", "lottery", "free $"]
        if any(kw in lower_prompt for kw in obvious_scam_keywords) or sig["money_words"] >= 2:
            return "SPAM", "Obvious prize or money scam"
        if sig["score"] >= 5:
            return "SPAM", "Excessive danger signs present"
        return "NOT_SPAM", "Benefit of doubt given to reminder or chat"

    elif role == "evidence":
        score = sig["score"]
        if score >= 4:
            return "SPAM", f"Calculated score is {score}"
        elif score <= 1:
            return "NOT_SPAM", f"Calculated score is {score}"
        else:
            return "UNSURE", f"Calculated score is {score}"

    elif role == "judge":
        if sig["score"] >= 2:
            return "SPAM", "Message contains multiple warning indicators"
        return "NOT_SPAM", "Message appears safe overall"

    return "UNSURE", "Unable to determine"


# ============================================================================
# GRAPH NODES
# ============================================================================

def strict_node(state: VotingPanelState) -> Dict[str, str]:
    """Strict security guard: distrusts everything."""
    message = state["message"]
    user_prompt = f"Message to analyze:\n{message}"
    vote, reason = invoke_juror_llm(
        system_prompt=STRICT_PROMPT,
        user_prompt=user_prompt,
        role="strict",
        signals=spam_signals(message),
    )
    formatted = f"{vote} - {reason}"
    return {"strict": formatted}


def friendly_node(state: VotingPanelState) -> Dict[str, str]:
    """Friendly reviewer: gives benefit of doubt; scams are SPAM."""
    message = state["message"]
    user_prompt = f"Message to analyze:\n{message}"
    vote, reason = invoke_juror_llm(
        system_prompt=FRIENDLY_PROMPT,
        user_prompt=user_prompt,
        role="friendly",
        signals=spam_signals(message),
    )
    formatted = f"{vote} - {reason}"
    return {"friendly": formatted}


def evidence_node(state: VotingPanelState) -> Dict[str, str]:
    """Data Analyst: gets spam_signals(...) and follows strict score rule."""
    message = state["message"]
    signals = spam_signals(message)

    user_prompt = (
        f"Message to analyze:\n{message}\n\n"
        f"Counted Warning Signs:\n"
        f"- Links: {signals['links']}\n"
        f"- Urgent words: {signals['urgent_words']}\n"
        f"- Money words: {signals['money_words']}\n"
        f"- Exclamations: {signals['exclamations']}\n"
        f"- Capital letters ratio: {signals['caps_ratio']}\n"
        f"- Total Score: {signals['score']}\n\n"
        f"Rule: score >= 4 -> SPAM; score 0 or 1 -> NOT_SPAM; score 2 or 3 -> UNSURE.\n"
        f"Quote the score in your reason."
    )

    vote, reason = invoke_juror_llm(
        system_prompt=EVIDENCE_PROMPT,
        user_prompt=user_prompt,
        role="evidence",
        signals=signals,
    )
    formatted = f"{vote} - {reason}"
    return {"evidence": formatted}


def tally_node(state: VotingPanelState) -> Dict[str, Any]:
    """NO LLM.

    Majority -> verdict + decided_by='majority vote' ;
    otherwise leave verdict empty.
    """
    strict_vote = parse_vote(state.get("strict", ""))
    friendly_vote = parse_vote(state.get("friendly", ""))
    evidence_vote = parse_vote(state.get("evidence", ""))

    votes = [strict_vote, friendly_vote, evidence_vote]
    winner, counts = tally(votes)

    if winner:
        return {
            "counts": dict(counts),
            "verdict": winner,
            "decided_by": "majority vote",
        }
    else:
        return {
            "counts": dict(counts),
            "verdict": None,
            "decided_by": None,
        }


def judge_node(state: VotingPanelState) -> Dict[str, Any]:
    """LLM tie-breaker -> verdict, decided_by='judge (tie-break)'."""
    message = state["message"]
    user_prompt = (
        f"Original Message:\n{message}\n\n"
        f"The panel disagreed:\n"
        f"- Strict Guard: {state.get('strict', 'N/A')}\n"
        f"- Friendly Person: {state.get('friendly', 'N/A')}\n"
        f"- Data Analyst: {state.get('evidence', 'N/A')}\n\n"
        f"Make the final call: SPAM or NOT_SPAM."
    )

    vote, reason = invoke_juror_llm(
        system_prompt=JUDGE_PROMPT,
        user_prompt=user_prompt,
        role="judge",
        signals=spam_signals(message),
    )

    # Judge must make a binary decision
    if vote not in ("SPAM", "NOT_SPAM"):
        vote = "SPAM" if "SPAM" in vote else "NOT_SPAM"

    return {
        "verdict": vote,
        "decided_by": "judge (tie-break)",
    }


# ============================================================================
# ROUTING & GRAPH CONSTRUCTION
# ============================================================================

def route_tally(state: VotingPanelState) -> str:
    """Conditional edge from tally: verdict ? END : judge."""
    if state.get("verdict"):
        return END
    return "judge"


def create_voting_panel_graph():
    """Builds the Voting Panel LangGraph StateGraph."""
    workflow = StateGraph(VotingPanelState)

    # Register Nodes
    workflow.add_node("strict", strict_node)
    workflow.add_node("friendly", friendly_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("tally", tally_node)
    workflow.add_node("judge", judge_node)

    # FAN-OUT: START -> 3 jurors
    workflow.add_edge(START, "strict")
    workflow.add_edge(START, "friendly")
    workflow.add_edge(START, "evidence")

    # FAN-IN: add_edge([3 jurors], 'tally')
    workflow.add_edge(["strict", "friendly", "evidence"], "tally")

    # Conditional branch from tally
    workflow.add_conditional_edges(
        "tally",
        route_tally,
        {END: END, "judge": "judge"},
    )

    # Judge -> END
    workflow.add_edge("judge", END)

    return workflow.compile()


# ============================================================================
# CLI APPLICATION LOOP
# ============================================================================

def format_verdict_badge(verdict: Optional[str]) -> str:
    """Formats the verdict with the required icons."""
    if verdict == "SPAM":
        return "🚫 SPAM"
    elif verdict == "NOT_SPAM":
        return "✅ NOT_SPAM"
    return f"❓ {verdict}"


def run_cli():
    """CLI loop: ask 'Message to check:', print juror votes, tally, verdict."""
    print("=" * 64)
    print("      VOTING-PANEL MULTI-AGENT SPAM CLASSIFIER (CLI)")
    print("=" * 64)
    print("Type 'exit' or 'quit' to stop.\n")

    graph = create_voting_panel_graph()

    while True:
        try:
            message = input("Message to check: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not message:
            continue
        if message.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print("\nAnalyzing message with Voting Panel...")
        result = graph.invoke({"message": message})

        # Display Juror Votes
        print("\n" + "-" * 40)
        print("JUROR VOTES:")
        print(f"  Strict Security Guard : {result.get('strict')}")
        print(f"  Friendly Reviewer     : {result.get('friendly')}")
        print(f"  Data Analyst Evidence : {result.get('evidence')}")

        # Display Tally
        counts = result.get("counts", {})
        counts_str = ", ".join(f"{k}: {v}" for k, v in counts.items()) if counts else "No decisive votes"
        print("\n" + "-" * 40)
        print(f"TALLY: {counts_str}")

        # Display Verdict
        verdict = result.get("verdict")
        decided_by = result.get("decided_by")
        badge = format_verdict_badge(verdict)
        print("\n" + "-" * 40)
        print(f"VERDICT: {badge} (decided by {decided_by})")
        print("=" * 64 + "\n")


if __name__ == "__main__":
    run_cli()
