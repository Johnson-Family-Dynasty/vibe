from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from config import CLASSROOM_TIMEZONE, now_in_classroom_timezone


def tool_get_current_time(timezone: Optional[str] = None) -> str:
    requested_timezone = (timezone or "").strip() or CLASSROOM_TIMEZONE
    try:
        now = datetime.now(ZoneInfo(requested_timezone))
    except (ZoneInfoNotFoundError, ValueError):
        now = now_in_classroom_timezone()
        requested_timezone = CLASSROOM_TIMEZONE

    timezone_name = getattr(now.tzinfo, "key", None) or now.tzname() or requested_timezone
    return (
        f"Current date/time for {timezone_name}:\n"
        f"- Date: {now.strftime('%A, %B')} {now.day}, {now.year}\n"
        f"- Time: {now.strftime('%I:%M %p').lstrip('0')} {now.tzname() or ''}\n"
        f"- ISO timestamp: {now.isoformat()}"
    )
