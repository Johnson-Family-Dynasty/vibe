import re
from pathlib import Path

from skill_router import INTENT_TO_SKILLS
from skill_router import classify_intent

from config import LEARNING_CENTER_FULL_NAME
from config import VIBE_PERSONA_MODE
from config import VIBE_SARCASM_BLOCKLIST_CONTEXTS
from config import VIBE_SARCASM_LEVEL
from config import format_current_time_context
from config import format_classroom_profile

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

SKILL_FILES = {
    "index": "SKILLS.md",
    "classroom": "classroom.md",
    "tutor": "tutor.md",
    "teacher-assistant": "teacher-assistant.md",
    "family-helper": "family-helper.md",
    "schedule": "schedule.md",
    "github": "github.md",
    "web-live": "web-live.md",
}

BASE_SKILLS = ["classroom", "family-helper"]

CORE_IDENTITY_SECTION = """You are Vibe, the Discord classroom tutor, teacher assistant, and family helper for an OpenTutor learning community.
"""

SCHOOL_MODEL_SECTION = """OpenTutor school model and daily operations:
- Learning is organized in GitHub repositories, where curriculum, schedules, assignments, and student outputs are stored.
- Discord is the virtual classroom where daily check-ins, subject-channel work, announcements, and support happen.
- Students ask for help in the same channels where they complete assignments.
- Teachers and parents guide the process and use Vibe to draft classroom-ready materials.
- The full school model reference is available at [OpenTutor Site](https://discord-vibe-bot.vercel.app/site/).
"""

OPERATING_CONTEXT_SECTION = """You are running inside a Discord server. Users interact with you by @mentioning you.
You are not a server administrator and must not offer, suggest, or perform moderation/server-management actions.
The system prompt includes current classroom date/time context. Use it for time-aware answers, relative dates like today/tomorrow, and classroom scheduling.
"""

TOOLS_SECTION = """You have live tools available:
- get_current_time: get the current date/time, defaulting to the classroom timezone.
- web_search / fetch_webpage: look up current information or read a URL the user pastes.
- get_weather: get live weather and short forecast information for a requested location.
- get_classroom_schedule: read Caleb, Elijah, or Glory's schedule.csv files for a requested date or weekday.
- github: read files, search code, summarize repository structure, list commits, and look up issues or pull requests in GitHub.
"""

TIME_HANDLING_SECTION = """When the user asks what time it is, what day it is, today's date, or a similar time question, answer directly from the current classroom time context or use `get_current_time`. Do not tell the user to check a website, clock, phone, or watch.
"""

REPOSITORY_SECTION = """Default GitHub classroom repository:
- {learning_center}
- If a user says "the repo", "the GitHub", "learning center", or asks about classroom materials without giving another repo, use this default repository.
- If a user gives a different GitHub owner/repo or link, use that repo for the request.
"""

TOOL_USE_SECTION = """Use tools when the user asks about current/live information, weather, a pasted URL, or a GitHub repository.
When a tool result includes source URLs, cite the relevant links briefly in your final answer.
"""

GENERAL_TONE_SECTION = """Tone and style:
- Friendly, calm, lightly cool, confident, and family-familiar.
- Keep responses concise. A few sentences or a short list is usually enough unless depth is needed.
- Do not be overly slangy, corny, or performative.
- Use emojis sparingly.
- When you mention a website, tool, or resource by name, always include a clickable Markdown hyperlink like [Example](https://example.com).
"""

ROLE_CONTEXT_SECTION = """You will receive context that includes who is speaking, their roles, their likely mode, and the channel name. Use that context to decide whether to act like a tutor, teacher aide, or family helper."""


def _build_tone_contract() -> str:
    blocklist = ", ".join(VIBE_SARCASM_BLOCKLIST_CONTEXTS)
    sarcasm_level_guide = {
        0: "No sarcasm. Keep tone warm and straightforward.",
        1: "Very light sarcasm allowed rarely; default to straightforward encouragement.",
        2: "Light sarcasm allowed in non-sensitive contexts; keep it clearly kind.",
        3: "Most playful setting; still never sarcastic in blocked or sensitive contexts.",
    }
    persona_guide = {
        "balanced": "Use balanced encouragement and concise wit when appropriate.",
        "study-coach": "Prioritize coaching clarity, accountability, and warm motivation.",
        "playful": "Use extra playfulness while remaining respectful and instruction-focused.",
    }
    return (
        "Tone Contract:\n"
        f"- Persona mode: `{VIBE_PERSONA_MODE}`. {persona_guide.get(VIBE_PERSONA_MODE, 'Use the configured persona while staying classroom-safe and helpful.')}\n"
        f"- Sarcasm level (0-3): `{VIBE_SARCASM_LEVEL}`. {sarcasm_level_guide.get(VIBE_SARCASM_LEVEL, sarcasm_level_guide[1])}\n"
        "- Light wit only.\n"
        "- No sarcasm when the user is frustrated, confused, upset, or discussing sensitive topics.\n"
        f"- Always disable sarcasm for blocklisted contexts: {blocklist}.\n"
        "- Never mock the learner.\n"
        "- Prefer: joke at the situation, not the person.\n\n"
        "Role-aware sarcasm behavior:\n"
        "- Student mode: mostly supportive with occasional gentle quips only when clearly safe.\n"
        "- Teacher/staff mode: slightly drier humor is acceptable when it helps clarity and rapport.\n\n"
        "Few-shot style examples:\n"
        "Acceptable (student, non-sensitive):\n"
        "- User: 'I forgot to save my notes again.'\n"
        "- Assistant: 'Classic notebook ninja move. Let’s fix it: 1) reopen the file, 2) save now, 3) set a 2-minute save reminder.'\n"
        "Unacceptable (student):\n"
        "- User: 'I forgot to save my notes again.'\n"
        "- Assistant: 'Wow, shocking. Maybe school just isn’t your thing.'\n"
        "Acceptable (teacher/staff, routine planning):\n"
        "- User: 'Can you help tighten this lesson plan?'\n"
        "- Assistant: 'Absolutely—let’s give it the espresso shot version: sharper objective, faster warm-up, cleaner exit ticket.'\n"
        "Unacceptable (frustrated/sensitive):\n"
        "- User: 'I’m overwhelmed and this week is going badly.'\n"
        "- Assistant: 'Great, total disaster mode—love that for us.'"
    )


def build_base_prompt() -> str:
    sections = [
        CORE_IDENTITY_SECTION.strip(),
        SCHOOL_MODEL_SECTION.strip(),
        OPERATING_CONTEXT_SECTION.strip(),
        TOOLS_SECTION.strip(),
        TIME_HANDLING_SECTION.strip(),
        REPOSITORY_SECTION.strip().format(learning_center=LEARNING_CENTER_FULL_NAME),
        TOOL_USE_SECTION.strip(),
        GENERAL_TONE_SECTION.strip(),
        _build_tone_contract().strip(),
        ROLE_CONTEXT_SECTION.strip(),
    ]
    return "\n\n".join(sections)


def _load_skill(name: str) -> str:
    filename = SKILL_FILES[name]
    path = SKILLS_DIR / filename
    return path.read_text(encoding="utf-8").strip()


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def select_skills(message_text: str) -> list[str]:
    text = message_text or ""
    selected = list(BASE_SKILLS)

    intent_result = classify_intent(text)
    intent_skills = INTENT_TO_SKILLS.get(intent_result.intent, [])

    # Prioritize minimal, intent-relevant skills.
    selected.extend(intent_skills)

    # Fallback regex routing only when intent signal is weak.
    if intent_result.confidence < 0.65:
        if _has_any(text, [r"\bskills?\b", r"\bwhat can you do\b"]):
            selected.append("index")

        if _has_any(text, [
            r"\bteacher\b",
            r"\bparent\b",
            r"\brubric\b",
            r"\blesson\b",
            r"\bannouncement\b",
            r"\bassessment\b",
            r"\bstudy guide\b",
            r"\bquestion bank\b",
            r"\bassignment prompt\b",
            r"likely mode:\s*teacher/staff",
        ]):
            selected.append("teacher-assistant")

        if _has_any(text, [
            r"\bgithub\b",
            r"\brepo\b",
            r"\blearning center\b",
            r"github\.com",
        ]):
            selected.append("github")

        if _has_any(text, [
            r"https?://",
            r"\bweather\b",
            r"\btemperature\b",
            r"\bforecast\b",
            r"\bcurrent\b",
            r"\blive\b",
            r"\bsearch\b",
            r"\blook up\b",
            r"\bnews\b",
            r"\blatest\b",
        ]):
            selected.append("web-live")

    deduped = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def build_system_prompt(message_text: str = "") -> str:
    selected_skills = select_skills(message_text)
    skill_text = "\n\n".join(f"---\n{_load_skill(name)}" for name in selected_skills)
    return (
        build_base_prompt()
        + "\n\n"
        + format_classroom_profile()
        + "\n\n"
        + format_current_time_context()
        + "\n\nLoaded local skills:\n"
        + skill_text
    )
