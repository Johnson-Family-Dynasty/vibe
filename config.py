import os
import re
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# RUNTIME CONFIG
# ──────────────────────────────────────────────
MAX_HISTORY = 10
COOLDOWN_SECONDS = 5
MAX_RESPONSE_LEN = 1900
MAX_TOOL_RESULT = 3000
MAX_SEARCH_RESULTS = 5
MAX_GITHUB_RESULTS = 10
MAX_GITHUB_TREE = 150
MAX_TOOL_ROUNDS = 8
HTTP_TIMEOUT = 12

LEARNING_CENTER_OWNER = os.getenv("LEARNING_CENTER_GITHUB_OWNER", "Johnson-Family-Dynasty")
LEARNING_CENTER_REPO = os.getenv("LEARNING_CENTER_GITHUB_REPO", "learning-center")
LEARNING_CENTER_FULL_NAME = f"{LEARNING_CENTER_OWNER}/{LEARNING_CENTER_REPO}"

CLASSROOM_NAME = os.getenv("CLASSROOM_NAME", "JFD Learning Center")
CLASSROOM_GROUP_NAME = os.getenv("CLASSROOM_GROUP_NAME", "Johnson Family Dynasty")
CLASSROOM_TIMEZONE = os.getenv("CLASSROOM_TIMEZONE", "America/Chicago")

REQUEST_HEADERS = {
    "User-Agent": "OpenTutor-VibeBot/1.0 (+https://discord-vibe-bot.vercel.app/site/)",
}

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **REQUEST_HEADERS,
    **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
}

# ──────────────────────────────────────────────
# CLASSROOM PROFILE
# ──────────────────────────────────────────────
CLASSROOM_STUDENTS = [
    {
        "name": "Caleb",
        "aliases": ["caleb", "creativegt"],
        "birthday": date(2012, 12, 3),
        "workspace": "students/caleb/",
        "schedule_path": "students/caleb/schedule.csv",
        "notes": "Oldest student. Give him more ownership, clear standards, and real-world skill challenges.",
    },
    {
        "name": "Elijah",
        "aliases": ["elijah"],
        "birthday": date(2016, 5, 9),
        "workspace": "students/elijah/",
        "schedule_path": "students/elijah/schedule.csv",
        "notes": "Middle student. Keep explanations concrete, encouraging, and broken into doable next steps.",
    },
    {
        "name": "Glory",
        "aliases": ["glory"],
        "birthday": date(2021, 2, 7),
        "workspace": "students/glory/",
        "schedule_path": "students/glory/schedule.csv",
        "notes": "Youngest student. Favor playful, experiential, oral, visual, and parent-guided learning.",
    },
]

CLASSROOM_TEACHERS = [
    {
        "name": "Josh",
        "family_role": "Dad",
        "teaching_role": "Primary teacher for hard skills and book knowledge: math, STEM, social studies, structured reading, research, technical skills, and GitHub workflows.",
    },
    {
        "name": "Lonisa",
        "family_role": "Mom",
        "teaching_role": "Primary teacher for experiential learning: hands-on projects, outings, observation, creativity, life skills, discussion, and practical application.",
    },
]

VIBE_FAMILY_ROLE = (
    "Vibe is the friendly AI buddy and helper inside the JFD group: tutor, teacher aide, "
    "GitHub learning-center guide, productivity helper, idea sparring partner, and upbeat family assistant."
)


def classroom_timezone():
    try:
        return ZoneInfo(CLASSROOM_TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now().astimezone().tzinfo


def now_in_classroom_timezone() -> datetime:
    return datetime.now(classroom_timezone())


def today_in_classroom_timezone() -> date:
    return now_in_classroom_timezone().date()


def format_current_time_context(now: Optional[datetime] = None) -> str:
    now = now or now_in_classroom_timezone()
    timezone_name = getattr(now.tzinfo, "key", None) or now.tzname() or CLASSROOM_TIMEZONE
    return (
        "Current classroom time:\n"
        f"- Local date: {now.strftime('%A, %B')} {now.day}, {now.year}\n"
        f"- Local time: {now.strftime('%I:%M %p').lstrip('0')} {now.tzname() or ''}\n"
        f"- Timezone: {timezone_name}\n"
        f"- ISO timestamp: {now.isoformat()}"
    )


def calculate_age(birthday: date, today=None) -> int:
    today = today or today_in_classroom_timezone()
    birthday_this_year = birthday.replace(year=today.year)
    return today.year - birthday.year - (today < birthday_this_year)


def student_matches(student: dict, text: str) -> bool:
    lowered = (text or "").lower()
    candidates = [student["name"], *student.get("aliases", [])]
    return any(re.search(rf"\b{re.escape(candidate.lower())}\b", lowered) for candidate in candidates)


def format_classroom_profile() -> str:
    student_lines = []
    for student in CLASSROOM_STUDENTS:
        age = calculate_age(student["birthday"])
        birthday = f"{student['birthday'].strftime('%B')} {student['birthday'].day}, {student['birthday'].year}"
        student_lines.append(
            f"- {student['name']}: student, age {age}, birthday {birthday}, "
            f"workspace `{student['workspace']}`, schedule `{student['schedule_path']}`. "
            f"Aliases: {', '.join(student['aliases'])}. {student['notes']}"
        )

    teacher_lines = [
        f"- {teacher['name']}: {teacher['family_role']}. {teacher['teaching_role']}"
        for teacher in CLASSROOM_TEACHERS
    ]

    return (
        "JFD classroom profile:\n"
        f"- Group: {CLASSROOM_GROUP_NAME}\n"
        f"- Classroom: {CLASSROOM_NAME}\n"
        f"- Family helper role: {VIBE_FAMILY_ROLE}\n\n"
        "Students:\n"
        + "\n".join(student_lines)
        + "\n\nPrimary teachers:\n"
        + "\n".join(teacher_lines)
        + "\n\nLearning-center repo model from the README:\n"
        f"- `{LEARNING_CENTER_FULL_NAME}` is the central homeschool hub for materials, assignments, projects, student work, teacher prompts, reports, and dashboards.\n"
        "- Students work in their own folders under `students/<name>/` and commit quizzes, assignments, notes, and projects back there.\n"
        "- Shared materials live in `resources/`, assignment specs live in `assignments/`, teacher AI prompts live in `teachers/ai-assistants/`, reports live in `teachers/reports/`, and dashboards/live views live in `teachers/sites/`.\n"
        "- The daily flow is: log in, check the student folder or weekly timeline, work from assignments/resources, ask for help in Discord, then save and commit completed work.\n"
    )


def describe_known_classroom_member(display_name: str) -> str:
    name = (display_name or "").lower()

    def has_name(part: str) -> bool:
        return re.search(rf"\b{re.escape(part.lower())}\b", name) is not None

    for student in CLASSROOM_STUDENTS:
        if student_matches(student, name):
            return (
                f"{student['name']} — JFD student, age {calculate_age(student['birthday'])}, "
                f"workspace `{student['workspace']}`"
            )

    for teacher in CLASSROOM_TEACHERS:
        teacher_name = teacher["name"].lower()
        family_role = teacher["family_role"].lower()
        if has_name(teacher_name) or has_name(family_role):
            return f"{teacher['name']} — {teacher['family_role']} and primary teacher"

    return "unknown"


def infer_user_mode(member_roles: list[str], known_member: str = "unknown") -> str:
    lowered_roles = [role.lower() for role in member_roles]

    if any(keyword in role for role in lowered_roles for keyword in ("teacher", "parent", "staff", "mod", "moderator")):
        return "teacher/staff"

    if any(keyword in role for role in lowered_roles for keyword in ("student", "learner", "kid")):
        return "student"

    if "JFD student" in known_member:
        return "student"

    if "primary teacher" in known_member:
        return "teacher/staff"

    return "unknown"
