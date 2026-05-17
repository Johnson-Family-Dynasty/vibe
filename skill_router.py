from __future__ import annotations

import re
from dataclasses import dataclass

# Intent labels expected by prompting.select_skills.
INTENTS = {
    "tutoring",
    "planning",
    "general_qa",
    "writing_help",
    "brainstorming",
    "coding_help",
    "web_lookup",
    "repo_lookup",
}


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float


INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "tutoring": (
        r"\bhomework\b", r"\bquiz\b", r"\bexplain\b", r"\bteach\b", r"\bmath\b",
        r"\bscience\b", r"\breading\b", r"likely mode:\s*student",
    ),
    "planning": (
        r"\bplan\b", r"\bschedule\b", r"\btimeline\b", r"\btoday\b", r"\btomorrow\b",
        r"\bthis week\b", r"\bwhat should\b",
    ),
    "writing_help": (
        r"\bwrite\b", r"\bdraft\b", r"\brewrite\b", r"\bedit\b", r"\bannouncement\b",
        r"\brubric\b", r"\blesson\b",
    ),
    "brainstorming": (
        r"\bideas?\b", r"\bbrainstorm\b", r"\boptions\b", r"\bcreative\b",
    ),
    "coding_help": (
        r"\bcode\b", r"\bbug\b", r"\bdebug\b", r"\bfunction\b", r"\bpython\b",
    ),
    "web_lookup": (
        r"https?://", r"\bweather\b", r"\bforecast\b", r"\bnews\b", r"\blatest\b", r"\blook up\b",
    ),
    "repo_lookup": (
        r"\bgithub\b", r"\brepo\b", r"\blearning center\b", r"github\.com", r"\bcommit\b", r"\bpr\b",
    ),
    "general_qa": (
        r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bcan you\b",
    ),
}

INTENT_TO_SKILLS: dict[str, list[str]] = {
    "tutoring": ["tutor"],
    "planning": ["schedule"],
    "general_qa": [],
    "writing_help": ["teacher-assistant"],
    "brainstorming": ["teacher-assistant"],
    "coding_help": ["github"],
    "web_lookup": ["web-live"],
    "repo_lookup": ["github"],
}

# Small testable table for quick regression checks.
ROUTING_EXAMPLES: list[tuple[str, list[str]]] = [
    ("Can you explain this math homework question?", ["tutor"]),
    ("Plan Caleb's schoolwork for tomorrow.", ["schedule"]),
    ("Draft a classroom announcement for parents.", ["teacher-assistant"]),
    ("Look up the latest weather forecast in Austin.", ["web-live"]),
    ("Check the repo and summarize open PRs.", ["github"]),
]


def classify_intent(message_text: str) -> IntentResult:
    text = (message_text or "").strip()
    if not text:
        return IntentResult(intent="general_qa", confidence=0.0)

    best_intent = "general_qa"
    best_hits = 0
    second_hits = 0

    for intent, patterns in INTENT_PATTERNS.items():
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        if hits > best_hits:
            second_hits = best_hits
            best_hits = hits
            best_intent = intent
        elif hits > second_hits:
            second_hits = hits

    if best_hits == 0:
        return IntentResult(intent="general_qa", confidence=0.0)

    margin = best_hits - second_hits
    confidence = min(1.0, 0.45 + (0.2 * best_hits) + (0.1 * margin))
    return IntentResult(intent=best_intent, confidence=round(confidence, 2))
