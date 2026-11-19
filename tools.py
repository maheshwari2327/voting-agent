"""Tools for the Voting Panel Spam Detector Agent.

Contains:
- spam_signals: Rule-based heuristics signal extractor.
- parse_vote: Robust vote parser from LLM response strings.
- tally: Voting aggregator that ignores UNSURE and finds >= 2 majorities.
"""

from collections import Counter
import re
from typing import Dict, List, Optional, Tuple, Any

# Urgency keywords and regex patterns
URGENT_PATTERN = re.compile(
    r"\b(urgent|urgently|immediately|hurry|act now|action required|limited time|expires?|expiring|instant|instantly|deadline|asap|alert|warning|final notice|last chance)\b",
    re.IGNORECASE,
)

# Financial / money / prize keywords and currency symbols
MONEY_WORDS_PATTERN = re.compile(
    r"\b(free|prize|winner|winning|cash|dollar|dollars|euro|euros|pound|pounds|credit|credits|pay|payment|bonus|reward|gift\s*card|claim|discount|win|won|lottery|refund|bitcoin|crypto|wire\s*transfer|bank\s*account)\b",
    re.IGNORECASE,
)
CURRENCY_SYMBOLS_PATTERN = re.compile(r"[\$€£¥₹]")

# URL / link pattern
LINK_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|org|net|edu|gov|io|co|xyz|info|biz|me|online|top|site|live|link|app)\b\S*",
    re.IGNORECASE,
)


def spam_signals(text: str) -> Dict[str, Any]:
    """Analyzes text for spam indicators and returns signal counts and composite score.

    Rule Scale:
      score >= 4 -> SPAM
      score <= 1 -> NOT_SPAM
      score 2 or 3 -> UNSURE
    """
    if not text:
        return {
            "links": 0,
            "urgent_words": 0,
            "money_words": 0,
            "exclamations": 0,
            "caps_ratio": 0.0,
            "score": 0,
        }

    # 1. Links count
    links_count = len(LINK_PATTERN.findall(text))

    # 2. Urgent words count
    urgent_count = len(URGENT_PATTERN.findall(text))

    # 3. Money words and currency signs count
    money_words_count = len(MONEY_WORDS_PATTERN.findall(text))
    currency_count = len(CURRENCY_SYMBOLS_PATTERN.findall(text))
    total_money_count = money_words_count + currency_count

    # 4. Exclamations count
    exclamations_count = text.count("!")

    # 5. Caps ratio
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars:
        caps_ratio = round(sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars), 2)
    else:
        caps_ratio = 0.0

    # 6. Composite Score calculation
    # - Links are high risk (+2 per link)
    # - Urgent keywords (+1 each)
    # - Money / prize indicators (+1 each)
    # - Multiple exclamations (+1 for 1-2, +2 for 3+)
    # - High capital letter ratio (+1 for >= 0.25, +2 for >= 0.50)
    score = 0
    score += links_count * 2
    score += urgent_count
    score += total_money_count

    if exclamations_count >= 3:
        score += 2
    elif exclamations_count >= 1:
        score += 1

    if caps_ratio >= 0.50:
        score += 2
    elif caps_ratio >= 0.25:
        score += 1

    return {
        "links": links_count,
        "urgent_words": urgent_count,
        "money_words": total_money_count,
        "exclamations": exclamations_count,
        "caps_ratio": caps_ratio,
        "score": score,
    }


def parse_vote(text: str) -> str:
    """Parses vote verdict from LLM output.

    Returns one of: 'SPAM', 'NOT_SPAM', 'UNSURE'.
    Safely reads the first words / structured labels of the reply.
    """
    if not text:
        return "UNSURE"

    clean_text = text.strip()

    # Check start of text (e.g. "NOT_SPAM - reason" or "SPAM: reason")
    first_token_match = re.match(
        r"^(NOT_SPAM|NOT\s+SPAM|NOT-SPAM|UNSURE|SPAM)\b",
        clean_text,
        re.IGNORECASE,
    )
    if first_token_match:
        token = first_token_match.group(1).upper()
        if "NOT" in token:
            return "NOT_SPAM"
        if "UNSURE" in token:
            return "UNSURE"
        return "SPAM"

    # Check for explicit "Vote: <VOTE>" line
    vote_line_match = re.search(
        r"Vote\s*:\s*(NOT_SPAM|NOT\s+SPAM|NOT-SPAM|UNSURE|SPAM)\b",
        clean_text,
        re.IGNORECASE,
    )
    if vote_line_match:
        token = vote_line_match.group(1).upper()
        if "NOT" in token:
            return "NOT_SPAM"
        if "UNSURE" in token:
            return "UNSURE"
        return "SPAM"

    # Search for keywords anywhere, prioritizing NOT_SPAM and UNSURE over SPAM
    if re.search(r"\b(NOT_SPAM|NOT\s+SPAM|NOT-SPAM)\b", clean_text, re.IGNORECASE):
        return "NOT_SPAM"
    if re.search(r"\bUNSURE\b", clean_text, re.IGNORECASE):
        return "UNSURE"
    if re.search(r"\bSPAM\b", clean_text, re.IGNORECASE):
        return "SPAM"

    return "UNSURE"


def parse_reason(text: str) -> str:
    """Extracts a short reason (< 12 words) from LLM output."""
    if not text:
        return "No reason provided"

    clean_text = text.strip()
    reason = ""

    # Check for "Reason: ..."
    reason_match = re.search(r"Reason\s*:\s*([^\n\r]+)", clean_text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Fallback to first line that doesn't define the vote
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        for line in lines:
            if not re.match(r"^(Vote|SPAM|NOT_SPAM|UNSURE)\b", line, re.IGNORECASE):
                reason = line
                break
        if not reason and lines:
            reason = lines[0]

    # Enforce under 12 words
    words = reason.split()
    if len(words) > 12:
        reason = " ".join(words[:12])

    return reason or "No reason provided"


def tally(votes: List[str]) -> Tuple[Optional[str], Counter]:
    """Tallies votes from jurors.

    Ignores 'UNSURE'.
    A winner needs >= 2 equal votes and no tie.
    Returns (winner_or_None, Counter).
    """
    # Filter to only valid decisive votes (ignore UNSURE)
    valid_votes = [v for v in votes if v in ("SPAM", "NOT_SPAM")]
    counts = Counter(valid_votes)

    if counts:
        top_vote, top_count = counts.most_common(1)[0]
        if top_count >= 2:
            return top_vote, counts

    return None, counts
