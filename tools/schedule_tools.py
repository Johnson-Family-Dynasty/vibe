import csv
import io
from datetime import date, datetime, timedelta
from urllib.parse import quote

from config import CLASSROOM_STUDENTS
from config import student_matches
from config import today_in_classroom_timezone
from tools.common import truncate
from tools.github_tools import decode_github_file, github_get, github_target


def _resolve_schedule_students(students=None):
    if not students:
        return CLASSROOM_STUDENTS, []

    if isinstance(students, str):
        students = [students]

    resolved = []
    unknown = []
    for raw_name in students:
        match = next((student for student in CLASSROOM_STUDENTS if student_matches(student, raw_name)), None)
        if match and match not in resolved:
            resolved.append(match)
        elif not match:
            unknown.append(raw_name)

    return (resolved or CLASSROOM_STUDENTS), unknown


def _format_date(value: date, include_year: bool = True) -> str:
    label = f"{value.strftime('%A')}, {value.strftime('%B')} {value.day}"
    if include_year:
        label += f", {value.year}"
    return label


def _format_short_date(value: date) -> str:
    return f"{value.strftime('%a')} {value.strftime('%b')} {value.day}"


def _parse_schedule_date(raw_date: str):
    raw_date = (raw_date or "").strip().strip('"')
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw_date, fmt).date()
        except ValueError:
            pass
    return None


def _resolve_schedule_target_date(weekday=None, target_date=None):
    if target_date:
        parsed = _parse_schedule_date(target_date)
        if parsed:
            return parsed, _format_date(parsed)

    today = today_in_classroom_timezone()
    text = (weekday or "").strip().lower()
    if not text:
        return None, "upcoming schedule rows"

    if text in {"today", "now"}:
        return today, _format_date(today)
    if text == "tomorrow":
        target = today + timedelta(days=1)
        return target, _format_date(target)

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for name, index in weekdays.items():
        if name in text:
            days_ahead = (index - today.weekday()) % 7
            if days_ahead == 0 and "next" in text:
                days_ahead = 7
            target = today + timedelta(days=days_ahead)
            return target, _format_date(target)

    return None, f"unrecognized date/weekday: {weekday or target_date}"


def _read_student_schedule(student: dict, owner: str, repo: str):
    path = student["schedule_path"]
    data = github_get(f"/repos/{owner}/{repo}/contents/{quote(path, safe='/')}")
    content = decode_github_file(data)
    rows = []
    for row in csv.DictReader(io.StringIO(content)):
        row_date = _parse_schedule_date(row.get("Date", ""))
        if row_date:
            rows.append({
                "date": row_date,
                "study_area": (row.get("Study Area") or "").strip(),
                "task": (row.get("Task") or "").strip(),
                "status": (row.get("Status") or "").strip(),
            })
    return rows, data.get("html_url", f"https://github.com/{owner}/{repo}/blob/main/{path}")


def tool_get_classroom_schedule(students=None, weekday=None, target_date=None, study_area=None) -> str:
    owner, repo = github_target()
    selected_students, unknown = _resolve_schedule_students(students)
    target, target_label = _resolve_schedule_target_date(weekday, target_date)
    area_filter = (study_area or "").lower().strip()

    lines = [
        f"Repository: {owner}/{repo}",
        f"Schedule lookup: {target_label}",
        f"Requested students: {', '.join(students) if isinstance(students, list) else (students or 'all')}",
    ]
    if unknown:
        lines.append(f"Unknown student aliases ignored: {', '.join(unknown)}")

    today = today_in_classroom_timezone()
    for student in selected_students:
        rows, source_url = _read_student_schedule(student, owner, repo)
        if target:
            matches = [row for row in rows if row["date"] == target]
        else:
            matches = [row for row in rows if today <= row["date"] <= today + timedelta(days=7)]
            matches = matches[:8]

        if area_filter:
            matches = [row for row in matches if area_filter in row["study_area"].lower()]

        lines.append(f"\n{student['name']} ({student['workspace']})")
        lines.append(f"Source: {source_url}")
        if not matches:
            lines.append("No matching schedule rows found.")
            continue

        for row in matches:
            date_label = _format_short_date(row["date"])
            lines.append(
                f"- {date_label} | {row['study_area']}: {row['task']} "
                f"[{row['status'] or 'No status'}]"
            )

    return truncate("\n".join(lines), 5000)
